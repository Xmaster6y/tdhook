import pytest
import torch
from tensordict import TensorDict

from tests.composition_conformance import assert_conformance
from tdhook.attribution import LRP
from tdhook.concepts import (
    ChannelConditionedLRP,
    ConceptSelection,
    _uniform_selection,
    concept_channel_gradient_callback,
)
from tdhook.workflow import Workflow


def _workflow(direction="positive"):
    return Workflow(
        LRP(
            input_modules=["linear1"],
            attribution_key=("attributions", "concept_examples"),
            warn_on_missing_rule=False,
        ),
        ConceptSelection(("attributions", "concept_examples", "linear1"), direction=direction),
        ChannelConditionedLRP(
            LRP(warn_on_missing_rule=False),
            condition_module="linear1",
        ),
    )


def _artifacts(seed=0):
    torch.manual_seed(seed)
    return TensorDict(
        {
            "input": torch.randn(4, 10),
            "concept_labels": torch.tensor([1, 1, 0, 0]),
        },
        batch_size=[4],
    )


def test_concept_attribution_workflow_is_declared_inspectable_and_matches_frozen_fixture(get_model):
    model = get_model(seed=42)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    module_types_before = {name: type(module) for name, module in model.named_modules()}
    artifacts = _artifacts()
    workflow = _workflow()

    plan = workflow.plan(model, artifacts)
    assert_conformance(
        "test_concept_attribution_workflow_is_declared_inspectable_and_matches_frozen_fixture",
        plan,
        status="supported",
    )
    first = workflow(model, artifacts.clone())
    second = workflow(model, artifacts.clone())

    assert plan.model_passes == 2
    assert [execution.steps for execution in plan.executions] == [
        ("0:LRP",),
        ("1:ConceptSelection",),
        ("2:ChannelConditionedLRP",),
    ]
    selection = first["metrics", "concept_selection"]
    assert set(selection.keys()) == {"positive_mean", "negative_mean", "scores", "channel", "direction", "score"}
    assert selection["direction"].unique().item() == 1
    assert selection["channel"].unique().item() == selection["scores"][0].argmax().item()
    assert selection["channel"].unique().item() == 8
    expected = torch.tensor(
        [
            [
                -0.026217487,
                -0.034608621,
                0.026935985,
                -0.020520344,
                0.030811844,
                -0.012047096,
                0.029041190,
                -0.039444517,
                -0.025619673,
                0.015477733,
            ],
            [
                -0.058280926,
                -0.076934248,
                0.059878133,
                -0.045616295,
                0.068494089,
                -0.026780445,
                0.064557955,
                -0.087684333,
                -0.056952000,
                0.034406677,
            ],
            [0.0] * 10,
            [0.0] * 10,
        ]
    )
    conditioned = first["attributions", "conditioned", "input"]
    torch.testing.assert_close(conditioned, expected)
    torch.testing.assert_close(conditioned, second["attributions", "conditioned", "input"])
    assert first["attributions", "concept_examples", "linear1"].shape == (4, 20)
    assert conditioned.shape == artifacts["input"].shape
    assert conditioned.device == artifacts["input"].device
    assert {name: type(module) for name, module in model.named_modules()} == module_types_before
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])


def test_concept_selection_can_be_swapped_without_changing_conditioned_method(default_test_model):
    result = _workflow(direction="negative")(default_test_model, _artifacts(seed=1))

    selection = result["metrics", "concept_selection"]
    assert selection["direction"].unique().item() == -1
    assert selection["channel"].unique().item() == selection["scores"][0].argmin().item()
    assert result["attributions", "conditioned", "input"].shape == (4, 10)


def test_concept_selection_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="direction"):
        ConceptSelection("relevance", direction="sideways")
    with pytest.raises(TypeError, match="nested keys"):
        ConceptSelection(object())

    operator = ConceptSelection("relevance", labels_key="labels")

    class Artifacts:
        def __init__(self, relevances, labels):
            self.values = {"relevance": relevances, "labels": labels}
            self.batch_dims = 1
            self.batch_size = torch.Size([2])

        def get(self, key):
            return self.values[key]

    cases = [
        (Artifacts(object(), torch.tensor([1, 0])), TypeError),
        (Artifacts(torch.ones(2), torch.tensor([1, 0])), ValueError),
        (Artifacts(torch.ones(2, 3), torch.tensor([1])), ValueError),
        (Artifacts(torch.ones(2, 3), torch.tensor([1, 1])), ValueError),
        (Artifacts(torch.ones(2, 3), torch.tensor([-1, 1])), ValueError),
    ]
    for artifacts, error in cases:
        with pytest.raises(error):
            operator(artifacts)


def test_concept_selection_preserves_batch_shape_and_uses_signed_spatial_relevance():
    operator = ConceptSelection("relevance", labels_key="labels")
    artifacts = TensorDict(
        {
            # Channel 0 cancels spatially, while channel 1 is positive.
            "relevance": torch.tensor(
                [
                    [[[[1.0, -1.0]], [[1.0, 1.0]]], [[[1.0, -1.0]], [[1.0, 1.0]]]],
                    [[[[0.0, 0.0]], [[0.0, 0.0]]], [[[0.0, 0.0]], [[0.0, 0.0]]]],
                ]
            ),
            "labels": torch.tensor([[1, 1], [0, 0]]),
        },
        batch_size=[2, 2],
    )

    result = operator(artifacts)

    selection = result[operator.out_key]
    assert selection.batch_size == torch.Size([2, 2])
    assert selection["channel"].unique().item() == 1
    assert selection["scores"].shape == (2, 2, 2)

    empty_channels = TensorDict(
        {"relevance": torch.ones(2, 0), "labels": torch.tensor([1, 0])},
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="non-empty channel"):
        operator(empty_channels)


def _conditioned_artifacts(channel, direction):
    return TensorDict(
        {
            "input": torch.ones(2, 10),
            "metrics": {"concept_selection": {"channel": channel, "direction": direction}},
        },
        batch_size=[2],
    )


def test_concept_conditioning_rejects_invalid_selection_artifacts(default_test_model):
    with pytest.raises(TypeError, match="LRP method"):
        ChannelConditionedLRP(torch.nn.Identity(), condition_module="linear1")
    with pytest.raises(ValueError, match="condition_module"):
        ChannelConditionedLRP(LRP(), condition_module="")
    with pytest.raises(TypeError, match="nested keys"):
        ChannelConditionedLRP(LRP(), condition_module="linear1", selection_key=object())
    with pytest.raises(ValueError, match="already has"):
        ChannelConditionedLRP(
            LRP(output_grad_callbacks={"linear1": lambda grad_output, **_: grad_output}),
            condition_module="linear1",
        )

    method = ChannelConditionedLRP(LRP(warn_on_missing_rule=False), condition_module="linear1")
    invalid = [
        _conditioned_artifacts(torch.empty(2, 0, dtype=torch.long), torch.ones(2, 0, dtype=torch.long)),
        _conditioned_artifacts(torch.tensor([0, 1]), torch.ones(2, dtype=torch.long)),
    ]
    for artifacts in invalid:
        with pytest.raises((TypeError, ValueError)):
            Workflow(method)(default_test_model, artifacts)

    with pytest.raises(ValueError, match="provide"):
        _uniform_selection(TensorDict({"channel": torch.zeros(2)}, batch_size=[2]))
    with pytest.raises(TypeError, match="must be tensors"):
        _uniform_selection(TensorDict({"channel": "zero", "direction": "positive"}, batch_size=[]))


def test_conditioned_method_preserves_base_initialisation_and_guards_hook_order(default_test_model):
    calls = []

    def initialise(inputs, additional):
        calls.append(additional["metrics", "concept_selection", "channel"].clone())
        return inputs

    method = ChannelConditionedLRP(
        LRP(init_attr_inputs=initialise, warn_on_missing_rule=False),
        condition_module="linear1",
    )
    artifacts = _conditioned_artifacts(torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    Workflow(method)(default_test_model, artifacts)
    assert len(calls) == 2

    with method.prepare(default_test_model) as prepared:
        bound = method._prepared[prepared.td_module]
        callback = bound._output_grad_callbacks["linear1"]
        with pytest.raises(RuntimeError, match="not loaded"):
            callback((torch.ones(2, 20),))


def test_concept_channel_gradient_callback_validates_its_selection_and_shape():
    with pytest.raises(ValueError, match="non-negative"):
        concept_channel_gradient_callback(-1, 1)
    with pytest.raises(ValueError, match="either -1 or 1"):
        concept_channel_gradient_callback(0, 0)
    callback = concept_channel_gradient_callback(2, 1)
    with pytest.raises(ValueError, match="invalid"):
        callback((torch.ones(1, 2, 2),))
    with pytest.raises(ValueError, match="invalid"):
        callback((torch.tensor(1.0),))
