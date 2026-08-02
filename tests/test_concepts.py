import torch
from tensordict import TensorDict

from tdhook.attribution import LRP
from tdhook.concepts import ChannelConditionedLRPStage, ConceptSelectionStage
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
