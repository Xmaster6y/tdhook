import torch
import pytest
from tensordict import TensorDict

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


def test_concept_attribution_workflow_is_declared_inspectable_and_deterministic(default_test_model):
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
    pipeline = _workflow()

    plan = pipeline.plan(artifacts)
    first = pipeline.run(default_test_model, artifacts.clone(), model_id="tiny-linear", seed=0)
    second = pipeline.run(default_test_model, artifacts.clone(), model_id="tiny-linear", seed=0)

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
    torch.testing.assert_close(
        first.artifacts[("attributions", "conditioned")], second.artifacts[("attributions", "conditioned")]
    )
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

        def get(self, key):
            return self.values[key]

    cases = [
        (Artifacts(object(), torch.tensor([1, 0])), TypeError),
        (Artifacts(torch.ones(2), torch.tensor([1, 0])), ValueError),
        (Artifacts(torch.ones(2, 3), torch.tensor([1])), ValueError),
        (Artifacts(torch.ones(2, 3), torch.tensor([1, 1])), ValueError),
    ]
    for artifacts, error in cases:
        with pytest.raises(error):
            stage.run(torch.nn.Identity(), artifacts)


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
