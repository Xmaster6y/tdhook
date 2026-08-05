"""Declared, artifact-only intrinsic-dimension analysis stages.

The stages in this module turn cached activations into estimator inputs without
encoding a model, board, or plotting convention.  They deliberately run after
``ActivationCachingStage``: the only model pass belongs to activation capture.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.tensorclass import NonTensorData
from tensordict.utils import NestedKey

from tdhook.latent import ActivationCaching
from tdhook.pipeline import Pipeline, PipelineKey, Stage
from tdhook.stages import ActivationCachingStage


def _store_shape_neutral(artifacts: TensorDictBase, key: PipelineKey, value: object) -> None:
    """Store analysis output without imposing the model batch shape on it."""
    artifacts.set(key, NonTensorData(value, batch_size=artifacts.batch_size))


def _shape_neutral_value(value: object) -> object:
    return value.data if isinstance(value, NonTensorData) else value


def channel_conditioned_samples(activations: torch.Tensor) -> torch.Tensor:
    """Arrange ``(samples, channels, ...)`` activations as ``(channels, samples, features)``.

    This is useful when each channel is a condition and the remaining trailing
    dimensions form that condition's feature vector.  It is not tied to a
    particular model architecture or domain.
    """
    if activations.ndim < 3:
        raise ValueError("Channel-conditioned activations need shape (samples, channels, ...)")
    return activations.movedim(1, 0).flatten(start_dim=2)


def spatial_conditioned_samples(activations: torch.Tensor) -> torch.Tensor:
    """Arrange ``(samples, channels, height, width)`` activations by spatial location.

    The result has shape ``(height * width, samples, channels)``.  Consumers
    can use it for image, board, or any other two-dimensional activation grid.
    """
    if activations.ndim != 4:
        raise ValueError("Spatial-conditioned activations need shape (samples, channels, height, width)")
    samples, channels, height, width = activations.shape
    return activations.permute(2, 3, 0, 1).reshape(height * width, samples, channels)


class ActivationSampleStage(Stage):
    """Select one cached activation and reshape it into estimator datasets."""

    def __init__(
        self,
        name: str,
        activation_key: PipelineKey,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        cache_key: PipelineKey = ("activations", "cache"),
        sample_key: PipelineKey = ("activations", "samples"),
    ) -> None:
        super().__init__(
            name,
            required_keys=[cache_key],
            provided_keys=[sample_key],
            effects=["activation_selection"],
            method_id="ActivationSample",
        )
        self.activation_key = activation_key
        self.transform = transform
        self.cache_key = cache_key
        self.sample_key = sample_key

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        activations = artifacts.get(self.cache_key).get(self.activation_key)
        if not isinstance(activations, torch.Tensor):
            raise TypeError(f"Cached activation {self.activation_key!r} must be a tensor")
        samples = self.transform(activations)
        if not isinstance(samples, torch.Tensor):
            raise TypeError("Activation sample transform must return a tensor")
        if samples.ndim < 2:
            raise ValueError("Estimator samples must have shape (..., points, features)")
        # Condition axes (channels, positions, or layers) need not align with
        # the model input batch.  NonTensorData preserves the tensor exactly
        # while retaining TensorDict ownership and provenance.
        _store_shape_neutral(artifacts, self.sample_key, samples)
        return artifacts


class DimensionEstimationStage(Stage):
    """Run an existing TensorDict intrinsic-dimension estimator on an artifact.

    ``TwoNnDimensionEstimator``, ``LocalKnnDimensionEstimator``,
    ``LocalPcaDimensionEstimator``, and ``CaPcaDimensionEstimator`` all use
    this adapter unchanged.  Their existing ``in_key`` and ``out_key`` remain
    private to the stage; callers use stable artifact paths instead.
    """

    def __init__(
        self,
        name: str,
        estimator: TensorDictModuleBase,
        *,
        sample_key: PipelineKey = ("activations", "samples"),
        dimension_key: PipelineKey = ("metrics", "dimension"),
    ) -> None:
        in_key = getattr(estimator, "in_key", None)
        out_key = getattr(estimator, "out_key", None)
        if not isinstance(in_key, NestedKey) or not isinstance(out_key, NestedKey):
            raise TypeError("Dimension estimator stages require native TensorDict in_key and out_key attributes")
        super().__init__(
            name,
            required_keys=[sample_key],
            provided_keys=[dimension_key],
            effects=["dimension_estimation"],
            method_id=type(estimator).__name__,
        )
        self.estimator = estimator
        self.sample_key = sample_key
        self.dimension_key = dimension_key
        self._in_key = in_key
        self._out_key = out_key

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        samples = _shape_neutral_value(artifacts.get(self.sample_key))
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"Estimator samples at {self.sample_key!r} must be a tensor")
        if samples.ndim < 2:
            raise ValueError("Estimator samples must have shape (..., points, features)")
        storage = TensorDict({self._in_key: samples}, batch_size=[])
        result = self.estimator(storage)
        _store_shape_neutral(artifacts, self.dimension_key, result.get(self._out_key))
        return artifacts


class DimensionSummaryStage(Stage):
    """Publish finite-value count, mean, and standard deviation for dimensions."""

    def __init__(
        self,
        name: str,
        *,
        dimension_key: PipelineKey = ("metrics", "dimension"),
        summary_key: PipelineKey = ("metrics", "dimension_summary"),
        dims: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            name,
            required_keys=[dimension_key],
            provided_keys=[summary_key],
            effects=["dimension_summary"],
            method_id="DimensionSummary",
        )
        self.dimension_key = dimension_key
        self.summary_key = summary_key
        self.dims = dims

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        dimensions = _shape_neutral_value(artifacts.get(self.dimension_key))
        if not isinstance(dimensions, torch.Tensor):
            raise TypeError(f"Dimensions at {self.dimension_key!r} must be a tensor")
        reduce_dims = (
            tuple(range(dimensions.ndim))
            if self.dims is None
            else ((self.dims,) if isinstance(self.dims, int) else tuple(self.dims))
        )
        reduce_dims = tuple(dim if dim >= 0 else dimensions.ndim + dim for dim in reduce_dims)
        if (
            (not reduce_dims and self.dims is not None)
            or len(set(reduce_dims)) != len(reduce_dims)
            or any(dim < 0 or dim >= dimensions.ndim for dim in reduce_dims)
        ):
            raise ValueError("Summary dimensions must be unique valid dimensions")
        finite = torch.isfinite(dimensions)
        if not reduce_dims:
            count = finite.to(dtype=torch.long)
            nan = torch.full_like(dimensions, float("nan"))
            summary = TensorDict(
                {
                    "count": count,
                    "mean": torch.where(finite, dimensions, nan),
                    "std": torch.where(finite, torch.zeros_like(dimensions), nan),
                },
                batch_size=[],
            )
            _store_shape_neutral(artifacts, self.summary_key, summary)
            return artifacts
        values = torch.where(finite, dimensions, torch.zeros_like(dimensions))
        count = finite.sum(dim=reduce_dims, keepdim=True)
        divisor = count.clamp_min(1)
        nan = torch.full_like(values.sum(dim=reduce_dims, keepdim=True), float("nan"))
        mean = torch.where(count > 0, values.sum(dim=reduce_dims, keepdim=True) / divisor, nan)
        centered = torch.where(finite, dimensions - mean, torch.zeros_like(dimensions))
        variance = torch.where(count > 0, centered.square().sum(dim=reduce_dims, keepdim=True) / divisor, nan)
        count, mean, variance = (value.squeeze(dim=reduce_dims) for value in (count, mean, variance))
        summary = TensorDict({"count": count, "mean": mean, "std": variance.sqrt()}, batch_size=[])
        _store_shape_neutral(artifacts, self.summary_key, summary)
        return artifacts


def conditioned_dimension_pipeline(
    cache: ActivationCaching,
    activation_key: PipelineKey,
    transform: Callable[[torch.Tensor], torch.Tensor],
    estimator: TensorDictModuleBase,
    *,
    input_key: PipelineKey = ("inputs", "input"),
    cache_key: PipelineKey = ("activations", "cache"),
    sample_key: PipelineKey = ("activations", "samples"),
    dimension_key: PipelineKey = ("metrics", "dimension"),
    summary_key: PipelineKey = ("metrics", "dimension_summary"),
    summary_dims: int | Sequence[int] | None = None,
) -> Pipeline:
    """Build the standard capture, selection, estimation, and summary pipeline."""
    return Pipeline(
        [
            ActivationCachingStage("capture-activations", cache, input_key=input_key, cache_key=cache_key),
            ActivationSampleStage(
                "select-samples", activation_key, transform, cache_key=cache_key, sample_key=sample_key
            ),
            DimensionEstimationStage(
                "estimate-dimension", estimator, sample_key=sample_key, dimension_key=dimension_key
            ),
            DimensionSummaryStage(
                "summarize-dimension", dimension_key=dimension_key, summary_key=summary_key, dims=summary_dims
            ),
        ]
    )


__all__ = [
    "ActivationSampleStage",
    "DimensionEstimationStage",
    "DimensionSummaryStage",
    "channel_conditioned_samples",
    "conditioned_dimension_pipeline",
    "spatial_conditioned_samples",
]
