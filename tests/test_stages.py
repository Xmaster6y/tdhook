import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from tdhook.artifacts import ArtifactRegistry
from tdhook.latent import ActivationCaching
from tdhook.pipeline import Pipeline, TransformStage
from tdhook.stages import (
    ActivationCachingStage,
    AttributionStage,
    ProbingStage,
    WeightInterventionStage,
    capability_for_stage,
)
from tdhook.contexts import HookingContextFactory


def _artifacts():
    return TensorDict({"inputs": {"input": torch.ones(2, 3)}}, batch_size=[2])


def test_activation_caching_stage_executes_a_real_method_and_publishes_its_cache():
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    result = Pipeline([ActivationCachingStage("cache", ActivationCaching("0"))]).run(model, _artifacts())

    assert "0" in result.artifacts[("activations", "cache")]
    assert result.provenance[0].method == "ActivationCaching"


def test_builtin_stages_publish_method_accurate_contracts_without_callbacks():
    # A TensorDictModule lets the unmodified factory route the legacy output
    # name each public method already uses.
    attribution_model = TensorDictModule(nn.Identity(), in_keys=["input"], out_keys=["attr"])
    attr_result = Pipeline([AttributionStage("attr", HookingContextFactory())]).run(attribution_model, _artifacts())
    assert torch.equal(attr_result.artifacts[("attributions", "values")], torch.ones(2, 3))

    results = object()
    probe_result = Pipeline([ProbingStage("probe", HookingContextFactory(), results)]).run(
        TensorDictModule(nn.Identity(), in_keys=["input"], out_keys=["ignored"]), _artifacts()
    )
    assert probe_result.artifacts[("probes", "results")] == results

    output_result = Pipeline([WeightInterventionStage("intervene", HookingContextFactory())]).run(
        TensorDictModule(nn.Identity(), in_keys=["input"], out_keys=["output"]), _artifacts()
    )
    assert torch.equal(output_result.artifacts[("outputs", "model")], torch.ones(2, 3))
    assert ("interventions", "weights") not in output_result.artifacts.keys(include_nested=True)


def test_pipeline_claims_and_checks_artifact_generation_during_execution(default_test_model):
    registry = ArtifactRegistry()
    pipeline = Pipeline(
        [
            TransformStage(
                "copy",
                lambda td: td.set(("outputs", "value"), td[("inputs", "input")]),
                required_keys=[("inputs", "input")],
                provided_keys=[("outputs", "value")],
            )
        ],
        artifact_registry=registry,
    )

    result = pipeline.run(default_test_model, _artifacts())
    registry.require_fresh(("outputs", "value"), generation=1)
    assert torch.equal(result.artifacts[("outputs", "value")], torch.ones(2, 3))


def test_capabilities_are_executable_contracts():
    assert capability_for_stage("ActivationCaching").provided_keys == (("activations", "cache"),)
    assert capability_for_stage("WeightIntervention").provided_keys == (("outputs", "model"),)
