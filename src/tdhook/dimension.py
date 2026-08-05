"""TensorDict-native operators for conditioned intrinsic-dimension analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.tensorclass import NonTensorData
from tensordict.utils import NestedKey

from tdhook._types import is_nested_key, join_keys
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow


def _store_shape_neutral(artifacts: TensorDictBase, key: NestedKey, value: object) -> None:
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


class ActivationSamples(TensorDictModuleBase):
    """Select one cached activation and reshape it into estimator datasets."""

    def __init__(
        self,
        activation_key: NestedKey,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        cache_key: NestedKey = ("activations", "cache"),
        out_key: NestedKey = ("activations", "samples"),
    ) -> None:
        super().__init__()
        if not is_nested_key(activation_key):
            raise TypeError("activation_key must be a TensorDict nested key")
        if not is_nested_key(cache_key) or not is_nested_key(out_key):
            raise TypeError("cache_key and out_key must be TensorDict nested keys")
        self.activation_key = activation_key
        self.transform = transform
        self.cache_key = cache_key
        self.out_key = out_key
        self.in_keys = [join_keys(cache_key, activation_key)]
        self.out_keys = [out_key]

    def forward(self, artifacts: TensorDictBase) -> TensorDictBase:
        activations = artifacts.get(self.in_keys[0])
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
        _store_shape_neutral(artifacts, self.out_key, samples)
        return artifacts


class DimensionEstimation(TensorDictModuleBase):
    """Run an existing TensorDict intrinsic-dimension estimator on an artifact.

    ``TwoNnDimensionEstimator``, ``LocalKnnDimensionEstimator``,
    ``LocalPcaDimensionEstimator``, and ``CaPcaDimensionEstimator`` are used
    unchanged. Their configured keys remain internal to this remapping
    operator; callers use stable workflow keys instead.
    """

    def __init__(
        self,
        estimator: TensorDictModuleBase,
        *,
        in_key: NestedKey = ("activations", "samples"),
        out_key: NestedKey = ("metrics", "dimension"),
    ) -> None:
        super().__init__()
        estimator_in_key = getattr(estimator, "in_key", None)
        estimator_out_key = getattr(estimator, "out_key", None)
        if not is_nested_key(estimator_in_key) or not is_nested_key(estimator_out_key):
            raise TypeError("Dimension estimation requires native estimator in_key and out_key attributes")
        if not is_nested_key(in_key) or not is_nested_key(out_key):
            raise TypeError("in_key and out_key must be TensorDict nested keys")
        self.estimator = estimator
        self.in_key = in_key
        self.out_key = out_key
        self.estimator_in_key = estimator_in_key
        self.estimator_out_key = estimator_out_key
        self.in_keys = [in_key]
        self.out_keys = [out_key]

    def forward(self, artifacts: TensorDictBase) -> TensorDictBase:
        samples = _shape_neutral_value(artifacts.get(self.in_key))
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"Estimator samples at {self.in_key!r} must be a tensor")
        if samples.ndim < 2:
            raise ValueError("Estimator samples must have shape (..., points, features)")
        storage = TensorDict({self.estimator_in_key: samples}, batch_size=[])
        result = self.estimator(storage)
        _store_shape_neutral(artifacts, self.out_key, result.get(self.estimator_out_key))
        return artifacts


class DimensionSummary(TensorDictModuleBase):
    """Publish finite-value count, mean, and standard deviation for dimensions."""

    def __init__(
        self,
        *,
        in_key: NestedKey = ("metrics", "dimension"),
        out_key: NestedKey = ("metrics", "dimension_summary"),
        dims: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not is_nested_key(in_key) or not is_nested_key(out_key):
            raise TypeError("in_key and out_key must be TensorDict nested keys")
        self.in_key = in_key
        self.out_key = out_key
        self.dims = dims
        self.in_keys = [in_key]
        self.out_keys = [out_key]

    def forward(self, artifacts: TensorDictBase) -> TensorDictBase:
        dimensions = _shape_neutral_value(artifacts.get(self.in_key))
        if not isinstance(dimensions, torch.Tensor):
            raise TypeError(f"Dimensions at {self.in_key!r} must be a tensor")
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
            _store_shape_neutral(artifacts, self.out_key, summary)
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
        _store_shape_neutral(artifacts, self.out_key, summary)
        return artifacts


def conditioned_dimension_workflow(
    cache: ActivationCaching,
    activation_key: NestedKey,
    transform: Callable[[torch.Tensor], torch.Tensor],
    estimator: TensorDictModuleBase,
    *,
    cache_key: NestedKey = ("activations", "cache"),
    sample_key: NestedKey = ("activations", "samples"),
    dimension_key: NestedKey = ("metrics", "dimension"),
    summary_key: NestedKey = ("metrics", "dimension_summary"),
    summary_dims: int | Sequence[int] | None = None,
) -> Workflow:
    """Build the standard capture, selection, estimation, and summary workflow."""
    if cache.cache_key != cache_key:
        raise ValueError(
            f"ActivationCaching publishes {cache.cache_key!r}; configure cache_key={cache_key!r} on the method"
        )
    return Workflow(
        cache,
        ActivationSamples(activation_key, transform, cache_key=cache_key, out_key=sample_key),
        DimensionEstimation(estimator, in_key=sample_key, out_key=dimension_key),
        DimensionSummary(in_key=dimension_key, out_key=summary_key, dims=summary_dims),
    )


__all__ = [
    "ActivationSamples",
    "DimensionEstimation",
    "DimensionSummary",
    "channel_conditioned_samples",
    "conditioned_dimension_workflow",
    "spatial_conditioned_samples",
]
