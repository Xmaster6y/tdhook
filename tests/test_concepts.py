import torch
import pytest
from tensordict import TensorDict

from tdhook.artifacts import ArtifactRegistry
from tdhook.attribution import LRP
from tdhook.concepts import ChannelConditionedLRPStage, ConceptSelectionStage, concept_channel_gradient_callback
from tdhook.pipeline import Pipeline
from tdhook.stages import AttributionStage


def _workflow(direction="positive"):
    return Pipeline(
        [
            AttributionStage(
                "concept-relevances",
                LRP(input_modules=["linear1"], warn_on_missing_rule=False),
                attribution_key=("attributions", "concept_examples"),
                legacy_attribution_key=("attr", "linear1"),
            ),
            ConceptSelectionStage("select-concept", direction=direction),
            ChannelConditionedLRPStage(
                "conditioned-relevance",
                LRP(warn_on_missing_rule=False),
                condition_module="linear1",
            ),
        ]
    )


def test_concept_attribution_workflow_is_declared_inspectable_and_matches_frozen_fixture(get_model):
    model = get_model(seed=42)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    module_types_before = {name: type(module) for name, module in model.named_modules()}
    torch.manual_seed(0)
    artifacts = TensorDict(
        {
            "inputs": {
                "input": torch.randn(4, 10),
                "concept_labels": torch.tensor([1, 1, 0, 0]),
            }
        },
        batch_size=[4],
    )
    registry = ArtifactRegistry()
    pipeline = _workflow()
    pipeline.artifact_registry = registry

    plan = pipeline.plan(artifacts)
    first = pipeline.run(model, artifacts.clone(), model_id="tiny-linear", seed=0)
    second = pipeline.run(model, artifacts.clone(), model_id="tiny-linear", seed=0)

    assert plan.model_passes == 2
    assert [run.stages for run in plan.runs] == [
        ("concept-relevances",),
        ("select-concept",),
        ("conditioned-relevance",),
    ]
    selection = first.artifacts[("metrics", "concept_selection")]
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
    torch.testing.assert_close(first.artifacts[("attributions", "conditioned")], expected)
    torch.testing.assert_close(
        first.artifacts[("attributions", "conditioned")], second.artifacts[("attributions", "conditioned")]
    )
    assert first.artifacts[("attributions", "concept_examples")].shape == (4, 20)
    assert first.artifacts[("attributions", "conditioned")].shape == artifacts[("inputs", "input")].shape
    assert first.artifacts[("attributions", "conditioned")].device == artifacts[("inputs", "input")].device
    registry.require_fresh(("metrics", "concept_selection"), generation=2)
    registry.require_fresh(("attributions", "conditioned"), generation=2)
    with pytest.raises(ValueError, match="already owned"):
        registry.claim(("attributions", "conditioned"), "another-stage", generation=2)
    assert {name: type(module) for name, module in model.named_modules()} == module_types_before
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])
    assert [record.parents for record in first.provenance] == [
        (("inputs", "input"),),
        (("attributions", "concept_examples"), ("inputs", "concept_labels")),
        (("inputs", "input"), ("metrics", "concept_selection")),
    ]


def test_concept_selection_can_be_swapped_without_changing_conditioned_stage(default_test_model):
    torch.manual_seed(1)
    artifacts = TensorDict(
        {
            "inputs": {
                "input": torch.randn(4, 10),
                "concept_labels": torch.tensor([1, 1, 0, 0]),
            }
        },
        batch_size=[4],
    )
    result = _workflow(direction="negative").run(default_test_model, artifacts)

    selection = result.artifacts[("metrics", "concept_selection")]
    assert selection["direction"].unique().item() == -1
    assert selection["channel"].unique().item() == selection["scores"][0].argmin().item()
    assert result.artifacts[("attributions", "conditioned")].shape == (4, 10)


def test_concept_selection_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="direction"):
        ConceptSelectionStage("select", direction="sideways")

    stage = ConceptSelectionStage("select")

    class Artifacts:
        def __init__(self, relevances, labels):
            self.values = {stage.relevance_key: relevances, stage.labels_key: labels}
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
            stage.run(torch.nn.Identity(), artifacts)


def test_concept_selection_preserves_batch_shape_and_uses_signed_spatial_relevance():
    stage = ConceptSelectionStage("select")
    artifacts = TensorDict(
        {
            "attributions": {
                # Channel 0 cancels spatially, while channel 1 is positive.
                "concept_examples": torch.tensor(
                    [
                        [[[[1.0, -1.0]], [[1.0, 1.0]]], [[[1.0, -1.0]], [[1.0, 1.0]]]],
                        [[[[0.0, 0.0]], [[0.0, 0.0]]], [[[0.0, 0.0]], [[0.0, 0.0]]]],
                    ]
                )
            },
            "inputs": {"concept_labels": torch.tensor([[1, 1], [0, 0]])},
        },
        batch_size=[2, 2],
    )

    result = stage.run(torch.nn.Identity(), artifacts)

    selection = result[stage.selection_key]
    assert selection.batch_size == torch.Size([2, 2])
    assert selection["channel"].unique().item() == 1
    assert selection["scores"].shape == (2, 2, 2)

    empty_channels = TensorDict(
        {
            "attributions": {"concept_examples": torch.ones(2, 0)},
            "inputs": {"concept_labels": torch.tensor([1, 0])},
        },
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="non-empty channel"):
        stage.run(torch.nn.Identity(), empty_channels)


def _conditioned_artifacts(channel, direction):
    return TensorDict(
        {
            "inputs": {"input": torch.ones(2, 10)},
            "metrics": {"concept_selection": {"channel": channel, "direction": direction}},
        },
        batch_size=[2],
    )


def test_concept_conditioning_rejects_invalid_selection_artifacts(monkeypatch):
    with pytest.raises(ValueError, match="condition_module"):
        ChannelConditionedLRPStage("conditioned", LRP(), condition_module="")

    stage = ChannelConditionedLRPStage("conditioned", LRP(), condition_module="linear1")
    invalid = [
        TensorDict({"inputs": {"input": torch.ones(2, 10)}}, batch_size=[2]),
        TensorDict(
            {"inputs": {"input": torch.ones(2, 10)}, "metrics": {"concept_selection": torch.ones(2)}}, batch_size=[2]
        ),
        _conditioned_artifacts(torch.empty(2, 0, dtype=torch.long), torch.ones(2, 0, dtype=torch.long)),
        _conditioned_artifacts(torch.tensor([0, 1]), torch.ones(2, dtype=torch.long)),
    ]
    for artifacts in invalid:
        with pytest.raises((TypeError, ValueError)):
            stage.run(torch.nn.Identity(), artifacts)

    class Selection:
        def __contains__(self, key):
            return True

        def get(self, key):
            return object()

    class Artifacts:
        def get(self, key):
            return Selection()

    monkeypatch.setattr("tdhook.concepts.TensorDictBase", Selection)
    with pytest.raises(TypeError, match="must be tensors"):
        stage.run(torch.nn.Identity(), Artifacts())
    monkeypatch.undo()

    duplicate = ChannelConditionedLRPStage(
        "conditioned",
        LRP(output_grad_callbacks={"linear1": lambda grad_output, **_: grad_output}),
        condition_module="linear1",
    )
    with pytest.raises(ValueError, match="already has"):
        duplicate.run(torch.nn.Identity(), _conditioned_artifacts(torch.zeros(2, dtype=torch.long), torch.ones(2)))


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
