import pytest
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from tdhook.artifacts import ArtifactRegistry
from tdhook.attribution import ActivationMaximisation, IntegratedGradients
from tdhook.latent import ActivationCaching, Probing
from tdhook.pipeline import Pipeline, TransformStage
from tdhook.stages import (
    ActivationCachingStage,
    AttributionStage,
    ProbingStage,
    WeightInterventionStage,
    activation_caching_stage,
    attribution_stage,
    capability_for_stage,
    probing_stage,
    weight_intervention_stage,
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
    attribution_model = TensorDictModule(nn.Identity(), in_keys=["input"], out_keys=[("attr", "input")])
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


def test_shared_registry_ignores_undeclared_retained_artifacts(default_test_model):
    registry = ArtifactRegistry()
    first = Pipeline(
        [
            TransformStage(
                "first",
                lambda td: td.set(("outputs", "retained"), td[("inputs", "input")]),
                required_keys=[("inputs", "input")],
                provided_keys=[("outputs", "retained")],
            )
        ],
        artifact_registry=registry,
    )
    retained = first.run(default_test_model, _artifacts()).artifacts
    second = Pipeline(
        [
            TransformStage(
                "second",
                lambda td: td.set(("metrics", "value"), td[("inputs", "input")].sum(-1)),
                required_keys=[("inputs", "input")],
                provided_keys=[("metrics", "value")],
            )
        ],
        artifact_registry=registry,
    )

    assert "value" in second.run(default_test_model, retained).artifacts["metrics"]


def test_capabilities_are_executable_contracts():
    assert capability_for_stage("ActivationCaching").provided_keys == (("activations", "cache"),)
    assert capability_for_stage("WeightIntervention").provided_keys == (("outputs", "model"),)
    assert capability_for_stage("Attribution", factory=IntegratedGradients(n_steps=2)).model_passes == 1
    with pytest.raises(ValueError, match="concrete factory"):
        capability_for_stage("Attribution")


def test_public_stage_factories_build_the_typed_stages():
    assert isinstance(activation_caching_stage("cache", ActivationCaching("0")), ActivationCachingStage)
    assert isinstance(attribution_stage("attr", HookingContextFactory()), AttributionStage)
    assert isinstance(probing_stage("probe", HookingContextFactory(), object()), ProbingStage)
    assert isinstance(weight_intervention_stage("intervene", HookingContextFactory()), WeightInterventionStage)


def test_attribution_stage_maps_baseline_and_additional_inputs_and_reports_passes(default_test_model):
    factory = IntegratedGradients(
        n_steps=2,
        compute_convergence_delta=True,
        additional_init_keys=["label"],
        init_attr_targets=lambda outputs, extra: outputs,
    )
    stage = AttributionStage("integrated-gradients", factory)
    assert stage.required_keys == (("inputs", "input"), ("inputs", "baseline"), ("inputs", "label"))
    assert stage.capability.model_passes == 3

    artifacts = TensorDict(
        {
            "inputs": {
                "input": torch.ones(2, 10),
                "baseline": torch.zeros(2, 10),
                "label": torch.zeros(2, dtype=torch.long),
            }
        },
        batch_size=[2],
    )
    result = Pipeline([stage]).run(default_test_model, artifacts)
    assert result.artifacts[("attributions", "values")].shape == (2, 10)
    assert AttributionStage("maximise", ActivationMaximisation(["linear1"], n_steps=4)).capability.model_passes == 4


def test_probing_stage_maps_configured_auxiliary_inputs(default_test_model):
    observed = []

    class RecordingProbe:
        def step(self, data, **kwargs):
            observed.append(kwargs)

    factory = Probing("linear1", lambda *_: RecordingProbe(), additional_keys=["labels", "step_type"])
    stage = ProbingStage("probe", factory, object())
    artifacts = TensorDict(
        {"inputs": {"input": torch.ones(2, 10), "labels": torch.ones(2), "step_type": "fit"}},
        batch_size=[2],
    )
    Pipeline([stage]).run(default_test_model, artifacts)
    assert stage.required_keys == (("inputs", "input"), ("inputs", "labels"), ("inputs", "step_type"))
    assert observed[0]["labels"].equal(torch.ones(2))
    assert observed[0]["step_type"] == "fit"


def test_activation_caching_stage_retains_configured_cache(default_test_model):
    cache = TensorDict()
    factory = ActivationCaching("linear1", cache=cache, clear_cache=False)
    Pipeline([ActivationCachingStage("cache", factory)]).run(
        default_test_model, TensorDict({"inputs": {"input": torch.ones(2, 10)}}, batch_size=[2])
    )
    assert "linear1" in cache
