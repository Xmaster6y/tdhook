import pytest
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from tests.composition_conformance import assert_conformance
from tdhook.artifacts import ArtifactRegistry
from tdhook.dimension import (
    ActivationSampleStage,
    DimensionEstimationStage,
    DimensionSummaryStage,
    channel_conditioned_samples,
    conditioned_dimension_pipeline,
    spatial_conditioned_samples,
)
from tdhook.latent import ActivationCaching
from tdhook.latent.dimension_estimation import (
    CaPcaDimensionEstimator,
    LocalKnnDimensionEstimator,
    LocalPcaDimensionEstimator,
    TwoNnDimensionEstimator,
)
from tdhook.pipeline import Pipeline


@pytest.fixture
def activation_fixture():
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1) / 10 + 0.1)
    inputs = torch.arange(8 * 3 * 2 * 2, dtype=torch.float32).reshape(8, 3, 2, 2) / 20
    return model, inputs


def test_conditioned_dimension_pipeline_has_one_capture_pass_and_frozen_channel_fixture(activation_fixture):
    model, inputs = activation_fixture
    hooks_before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    registry = ArtifactRegistry()
    pipeline = conditioned_dimension_pipeline(
        ActivationCaching("0"), "0", channel_conditioned_samples, TwoNnDimensionEstimator()
    )
    pipeline.artifact_registry = registry
    artifacts = TensorDict({"inputs": {"input": inputs}}, batch_size=[len(inputs)])

    result = pipeline.run(model, artifacts, model_id="frozen-conv", seed=0)
    assert_conformance(
        "test_conditioned_dimension_pipeline_has_one_capture_pass_and_frozen_channel_fixture",
        result.plan,
        status="supported",
    )

    assert [(run.stages, run.model_passes) for run in result.plan.runs] == [
        (("capture-activations",), 1),
        (("select-samples",), 0),
        (("estimate-dimension",), 0),
        (("summarize-dimension",), 0),
    ]
    torch.testing.assert_close(result.artifacts[("metrics", "dimension")].data, torch.full((4,), 3.0))
    assert result.artifacts[("activations", "cache")]["0"].shape == (8, 4, 2, 2)
    assert result.artifacts[("activations", "samples")].data.shape == (4, 8, 4)
    assert result.artifacts[("metrics", "dimension")].data.device == inputs.device
    summary = result.artifacts[("metrics", "dimension_summary")].data
    assert summary["count"].item() == 4
    torch.testing.assert_close(summary["mean"], torch.tensor(3.0))
    torch.testing.assert_close(summary["std"], torch.tensor(0.0))
    assert [item.parents for item in result.provenance] == [
        (("inputs", "input"),),
        (("activations", "cache"),),
        (("activations", "samples"),),
        (("metrics", "dimension"),),
    ]
    for key in (
        ("activations", "cache"),
        ("activations", "samples"),
        ("metrics", "dimension"),
        ("metrics", "dimension_summary"),
    ):
        registry.require_fresh(key, generation=1)
    with pytest.raises(ValueError, match="already owned"):
        registry.claim(("metrics", "dimension"), "another-stage", generation=1)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert (
        sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules()) == hooks_before
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity is optional")
def test_conditioned_dimension_pipeline_matches_cpu_on_cuda(activation_fixture):
    cpu_model, cpu_inputs = activation_fixture
    cpu_result = conditioned_dimension_pipeline(
        ActivationCaching("0"), "0", channel_conditioned_samples, TwoNnDimensionEstimator()
    ).run(cpu_model, TensorDict({"inputs": {"input": cpu_inputs}}, batch_size=[len(cpu_inputs)]))

    cuda_model = cpu_model.to("cuda")
    cuda_inputs = cpu_inputs.to("cuda")
    cuda_result = conditioned_dimension_pipeline(
        ActivationCaching("0"), "0", channel_conditioned_samples, TwoNnDimensionEstimator()
    ).run(cuda_model, TensorDict({"inputs": {"input": cuda_inputs}}, batch_size=[len(cuda_inputs)]))

    assert cuda_result.plan.model_passes == 1
    assert cuda_result.artifacts[("metrics", "dimension")].data.device.type == "cuda"
    torch.testing.assert_close(
        cuda_result.artifacts[("metrics", "dimension")].data.cpu(),
        cpu_result.artifacts[("metrics", "dimension")].data,
    )


def test_channel_and_spatial_selectors_match_the_notebook_reshapes(activation_fixture):
    model, inputs = activation_fixture
    activations = model(inputs)

    assert channel_conditioned_samples(activations).shape == (4, 8, 4)
    assert spatial_conditioned_samples(activations).shape == (4, 8, 4)
    torch.testing.assert_close(
        channel_conditioned_samples(activations), activations.permute(1, 0, 2, 3).reshape(4, 8, 4)
    )
    torch.testing.assert_close(
        spatial_conditioned_samples(activations), activations.permute(2, 3, 0, 1).reshape(4, 8, 4)
    )


@pytest.mark.parametrize(
    ("selector", "activations", "message"),
    [
        (channel_conditioned_samples, torch.ones(2, 3), "Channel-conditioned"),
        (spatial_conditioned_samples, torch.ones(2, 3, 4), "Spatial-conditioned"),
    ],
)
def test_conditioned_selectors_reject_incompatible_activation_shapes(selector, activations, message):
    with pytest.raises(ValueError, match=message):
        selector(activations)


def test_multiple_cached_layers_can_feed_independent_conditioned_slices(activation_fixture):
    model, inputs = activation_fixture
    first = model(inputs)
    artifacts = TensorDict({"activations": {"cache": {"first": first, "second": first + 1}}}, batch_size=[len(inputs)])
    pipeline = Pipeline(
        [
            ActivationSampleStage(
                "first-layer", "first", channel_conditioned_samples, sample_key=("activations", "first")
            ),
            ActivationSampleStage(
                "second-layer", "second", spatial_conditioned_samples, sample_key=("activations", "second")
            ),
        ]
    )

    result = pipeline.run(model, artifacts)

    assert result.plan.model_passes == 0
    assert result.artifacts[("activations", "first")].data.shape == (4, 8, 4)
    assert result.artifacts[("activations", "second")].data.shape == (4, 8, 4)


@pytest.mark.parametrize(
    "estimator",
    [
        TwoNnDimensionEstimator(),
        LocalKnnDimensionEstimator(k=2),
        LocalPcaDimensionEstimator(k=2),
        CaPcaDimensionEstimator(k=2),
    ],
)
def test_existing_estimators_are_swappable_artifact_only_stages(estimator):
    samples = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]], dtype=torch.float32
    )
    artifacts = TensorDict({"activations": {"samples": NonTensorData(samples, batch_size=[2])}}, batch_size=[2])
    pipeline = Pipeline([DimensionEstimationStage("estimate", estimator)])

    result = pipeline.run(nn.Identity(), artifacts)

    dimensions = result.artifacts[("metrics", "dimension")].data
    assert dimensions.shape[0] == 1
    assert result.plan.model_passes == 0


def test_dimension_stage_accepts_an_estimator_with_nested_internal_keys():
    samples = torch.randn(1, 8, 4)
    artifacts = TensorDict(
        {"activations": {"samples": NonTensorData(samples, batch_size=[])}},
        batch_size=[],
    )
    estimator = TwoNnDimensionEstimator(
        in_key=("operator", "samples"),
        out_key=("operator", "dimension"),
    )

    result = DimensionEstimationStage("estimate", estimator).run(nn.Identity(), artifacts)

    assert result.get(("metrics", "dimension")).data.shape == (1,)


def test_dimension_stages_reject_invalid_artifacts_and_estimator_contracts():
    artifacts = TensorDict(
        {"activations": {"cache": {"invalid": NonTensorData("not-a-tensor", batch_size=[])}}}, batch_size=[]
    )
    with pytest.raises(TypeError, match="Cached activation"):
        ActivationSampleStage("invalid-cache", "invalid", lambda value: value).run(nn.Identity(), artifacts)

    tensor_artifacts = TensorDict({"activations": {"cache": {"value": torch.ones(3, 2, 2)}}}, batch_size=[])
    with pytest.raises(TypeError, match="must return a tensor"):
        ActivationSampleStage("invalid-transform", "value", lambda value: "not-a-tensor").run(
            nn.Identity(), tensor_artifacts
        )
    with pytest.raises(ValueError, match="points, features"):
        ActivationSampleStage("invalid-shape", "value", lambda value: value[0, 0]).run(nn.Identity(), tensor_artifacts)

    with pytest.raises(TypeError, match="in_key and out_key"):
        DimensionEstimationStage("invalid-estimator", nn.Identity())

    samples = TensorDict({"activations": {"samples": NonTensorData("not-a-tensor", batch_size=[])}}, batch_size=[])
    with pytest.raises(TypeError, match="Estimator samples"):
        DimensionEstimationStage("invalid-samples", TwoNnDimensionEstimator()).run(nn.Identity(), samples)
    samples.set(("activations", "samples"), NonTensorData(torch.ones(2), batch_size=[]))
    with pytest.raises(ValueError, match="points, features"):
        DimensionEstimationStage("one-dimensional-samples", TwoNnDimensionEstimator()).run(nn.Identity(), samples)


def test_summary_can_preserve_condition_axes_and_ignores_non_finite_values():
    dimensions = torch.tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, 6.0]])
    artifacts = TensorDict({"metrics": {"dimension": NonTensorData(dimensions, batch_size=[2])}}, batch_size=[2])
    result = Pipeline([DimensionSummaryStage("summary", dims=-1)]).run(nn.Identity(), artifacts)

    summary = result.artifacts[("metrics", "dimension_summary")].data
    torch.testing.assert_close(summary["count"], torch.tensor([2, 3]))
    torch.testing.assert_close(summary["mean"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(summary["std"], torch.tensor([1.0, (8 / 3) ** 0.5]))


def test_summary_handles_scalar_dimensions_and_marks_empty_slices_undefined():
    scalar = TensorDict({"metrics": {"dimension": NonTensorData(torch.tensor(2.5), batch_size=[])}}, batch_size=[])
    scalar_summary = Pipeline([DimensionSummaryStage("scalar-summary")]).run(nn.Identity(), scalar)
    assert scalar_summary.artifacts[("metrics", "dimension_summary")].data["count"].item() == 1
    torch.testing.assert_close(
        scalar_summary.artifacts[("metrics", "dimension_summary")].data["mean"], torch.tensor(2.5)
    )

    empty = TensorDict(
        {"metrics": {"dimension": NonTensorData(torch.full((2, 3), float("nan")), batch_size=[])}}, batch_size=[]
    )
    empty_summary = Pipeline([DimensionSummaryStage("empty-summary", dims=-1)]).run(nn.Identity(), empty)
    summary = empty_summary.artifacts[("metrics", "dimension_summary")].data
    torch.testing.assert_close(summary["count"], torch.zeros(2, dtype=torch.long))
    assert torch.isnan(summary["mean"]).all()
    assert torch.isnan(summary["std"]).all()


def test_summary_rejects_non_tensor_dimensions_and_invalid_reduction_axes():
    artifacts = TensorDict({"metrics": {"dimension": NonTensorData("not-a-tensor", batch_size=[])}}, batch_size=[])
    with pytest.raises(TypeError, match="Dimensions"):
        DimensionSummaryStage("invalid-dimensions").run(nn.Identity(), artifacts)

    artifacts.set(("metrics", "dimension"), NonTensorData(torch.ones(2, 2), batch_size=[]))
    with pytest.raises(ValueError, match="unique valid dimensions"):
        DimensionSummaryStage("invalid-axes", dims=(0, 0)).run(nn.Identity(), artifacts)
    with pytest.raises(ValueError, match="unique valid dimensions"):
        DimensionSummaryStage("empty-axes", dims=()).run(nn.Identity(), artifacts)
