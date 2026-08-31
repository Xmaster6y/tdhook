"""Input-dependent feature-circuit attribution.

This module implements the local decompositions used by CircuitLens without
depending on a transformer or transcoder implementation.  Model integrations
describe their sites with public :class:`~tdhook.targets.Target` objects;
:func:`attribute_feature_circuit` obtains activations and output gradients
through :class:`~tdhook.session.HookSession`.

All scores are evaluated at the observed input.  Autograd supplies the local
Jacobian, so piecewise nonlinearity gates are frozen at their observed state.
Attention scores additionally hold the observed attention pattern fixed and
decompose the value/output path by head and source token.  The result is a
local attribution, not an intervention or a finite-difference causal effect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import torch
from torch import Tensor, nn

from tdhook.session import CapturedTarget, HookSession
from tdhook.targets import Target


@dataclass(frozen=True)
class FeatureSite:
    """One transcoder-feature activation selected through a public target.

    ``position`` indexes the non-feature axes after the target has selected
    ``target.indices``.  It is required for the target feature when the
    selected tensor contains more than one scalar and is optional for upstream
    sites, whose remaining positions are attributed independently.
    """

    layer: int
    target: Target
    position: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.target.kind != "activation":
            raise ValueError("FeatureSite.target must be an activation target")

    def gradient_target(self) -> Target:
        """Return the matching output-gradient target."""

        return Target(
            self.target.module_path,
            "gradient",
            self.target.feature_axis,
            self.target.indices,
            output_path=self.target.output_path,
        )


@dataclass(frozen=True)
class AttentionSite:
    """Captured tensors needed for a frozen-pattern attention decomposition.

    The selected tensors must have shapes ``[heads, query, source]``,
    ``[source, heads, head_dim]``, and ``[query, model]`` respectively.  A
    leading singleton batch dimension is accepted. ``output_weight`` has shape
    ``[heads, head_dim, model]``.
    """

    layer: int
    pattern: Target
    values: Target
    output_gradient: Target
    output_weight: Tensor
    target_position: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.target_position < 0:
            raise ValueError("target_position must be non-negative")
        if self.pattern.kind != "activation" or self.values.kind != "activation":
            raise ValueError("attention pattern and values must be activation targets")
        if self.output_gradient.kind != "gradient":
            raise ValueError("attention output_gradient must be a gradient target")


@dataclass(frozen=True)
class AttributionConventions:
    """Serializable statement of the attribution semantics."""

    jacobian: str = "local autograd Jacobian at the observed input"
    attention: str = "observed attention pattern frozen; value/output path decomposed"
    nonlinearities: str = "observed local derivatives used; piecewise gates frozen"
    score: str = "activation times local gradient"


@dataclass(frozen=True)
class FeatureContributor:
    """Contribution from one upstream feature and non-feature position."""

    layer: int
    feature_index: int
    position: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class AttentionContributor:
    """Contribution from one attention head and source token."""

    layer: int
    head_index: int
    source_token: int
    target_token: int
    score: float


@dataclass(frozen=True)
class LogitContributor:
    """Contribution from the active target feature to one output logit."""

    token_index: int
    score: float


@dataclass(frozen=True)
class CircuitLensArtifact:
    """JSON-serializable contributors and scores from one workflow run."""

    target_layer: int
    target_feature_index: int
    target_position: tuple[int, ...]
    target_activation: float
    upstream_features: tuple[FeatureContributor, ...]
    attention: tuple[AttentionContributor, ...]
    output_logits: tuple[LogitContributor, ...]
    conventions: AttributionConventions = AttributionConventions()

    def to_dict(self) -> dict[str, object]:
        """Return a nested representation accepted by :func:`json.dumps`."""

        return asdict(self)


def feature_contributions(
    activations: Tensor,
    gradients: Tensor,
    *,
    layer: int,
    feature_indices: Sequence[int],
    feature_axis: int = -1,
) -> tuple[FeatureContributor, ...]:
    """Compute activation-times-gradient scores for upstream features."""

    if activations.shape != gradients.shape:
        raise ValueError("feature activations and gradients must have the same shape")
    axis = _normalized_axis(feature_axis, activations.ndim)
    if activations.shape[axis] != len(feature_indices):
        raise ValueError("feature_indices must match the selected feature axis")
    scores = (activations.detach() * gradients.detach()).to(device="cpu", dtype=torch.float64)
    contributors: list[FeatureContributor] = []
    for coordinate in _coordinates(scores.shape):
        feature_offset = coordinate[axis]
        position = coordinate[:axis] + coordinate[axis + 1 :]
        contributors.append(
            FeatureContributor(layer, int(feature_indices[feature_offset]), position, float(scores[coordinate]))
        )
    return tuple(contributors)


def attention_contributions(
    pattern: Tensor,
    values: Tensor,
    output_weight: Tensor,
    output_gradient: Tensor,
    *,
    layer: int,
    target_position: int,
    head_indices: Sequence[int] | None = None,
) -> tuple[AttentionContributor, ...]:
    """Decompose a frozen attention output by head and source token.

    The score for head ``h`` and source ``s`` is
    ``pattern[h, q, s] * <values[s, h] @ W_O[h], d target / d attn_out[q]>``.
    """

    pattern = _without_singleton_batch("pattern", pattern, 3)
    values = _without_singleton_batch("values", values, 3)
    output_gradient = _without_singleton_batch("output_gradient", output_gradient, 2)
    if output_weight.ndim != 3:
        raise ValueError("output_weight must have shape [heads, head_dim, model]")
    heads, queries, sources = pattern.shape
    if values.shape[:2] != (sources, heads):
        raise ValueError("values must have shape [source, heads, head_dim]")
    if output_weight.shape[:2] != (heads, values.shape[2]):
        raise ValueError("output_weight head and head_dim axes do not match values")
    if output_gradient.shape != (queries, output_weight.shape[2]):
        raise ValueError("output_gradient must have shape [query, model]")
    if target_position >= queries:
        raise IndexError("target_position is outside the attention query axis")
    if head_indices is None:
        head_indices = tuple(range(heads))
    if len(head_indices) != heads:
        raise ValueError("head_indices must contain one index per selected head")

    q = target_position
    projected_values = torch.einsum("shd,hdm->shm", values.detach(), output_weight.detach())
    scores = torch.einsum("hs,shm,m->hs", pattern.detach()[:, q, :], projected_values, output_gradient.detach()[q]).to(
        device="cpu", dtype=torch.float64
    )
    return tuple(
        AttentionContributor(layer, int(head_indices[head]), source, q, float(scores[head, source]))
        for head in range(heads)
        for source in range(sources)
    )


def logit_contributions(
    feature_activation: Tensor | float,
    logit_gradients: Tensor,
    *,
    token_indices: Sequence[int],
) -> tuple[LogitContributor, ...]:
    """Compute feature-activation times each selected logit's local gradient."""

    activation = torch.as_tensor(feature_activation).detach()
    if activation.numel() != 1:
        raise ValueError("feature_activation must contain one scalar")
    gradients = logit_gradients.detach().reshape(-1)
    if gradients.numel() != len(token_indices):
        raise ValueError("token_indices must contain one index per logit gradient")
    scores = (activation.reshape(()) * gradients).to(device="cpu", dtype=torch.float64)
    return tuple(
        LogitContributor(int(index), float(score)) for index, score in zip(token_indices, scores, strict=True)
    )


def attribute_feature_circuit(
    model: nn.Module,
    *model_args: object,
    target_feature: FeatureSite,
    upstream_features: Sequence[FeatureSite] = (),
    attention_sites: Sequence[AttentionSite] = (),
    output_logits: Callable[[object], Tensor] | None = None,
    logit_indices: Sequence[int] = (),
    model_kwargs: Mapping[str, object] | None = None,
    top_k: int | None = None,
    positive_only: bool = False,
) -> CircuitLensArtifact:
    """Run one input-dependent CircuitLens attribution workflow.

    ``output_logits`` extracts a one-dimensional logit tensor from the model
    output. Only ``logit_indices`` are differentiated, avoiding a full-vocab
    Jacobian unless the caller explicitly requests it. Existing parameter
    gradients are restored after the workflow.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if len(target_feature.target.indices) != 1:
        raise ValueError("target_feature must select exactly one feature")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if logit_indices and output_logits is None:
        raise ValueError("output_logits is required when logit_indices are requested")
    kwargs = {} if model_kwargs is None else dict(model_kwargs)
    saved_gradients = _save_parameter_gradients(model)

    try:
        with HookSession(model) as session:
            target_activation = session.capture(target_feature.target, detach=False)
            target_gradient = session.capture(target_feature.gradient_target())
            upstream_captures = [
                (site, session.capture(site.target), session.capture(site.gradient_target()))
                for site in upstream_features
            ]
            attention_captures = [
                (
                    site,
                    session.capture(site.pattern),
                    session.capture(site.values),
                    session.capture(site.output_gradient),
                )
                for site in attention_sites
            ]

            output = model(*model_args, **kwargs)
            live_target = _site_scalar(_captured(target_activation, "target feature"), target_feature)
            retain_graph = bool(logit_indices)
            live_target.backward(retain_graph=retain_graph)

            upstream = tuple(
                contributor
                for site, activations, gradients in upstream_captures
                for contributor in feature_contributions(
                    _captured(activations, "upstream activation"),
                    _captured(gradients, "upstream gradient"),
                    layer=site.layer,
                    feature_indices=site.target.indices,
                    feature_axis=site.target.feature_axis,
                )
            )
            attention = tuple(
                contributor
                for site, pattern, values, gradient in attention_captures
                for contributor in attention_contributions(
                    _captured(pattern, "attention pattern"),
                    _captured(values, "attention values"),
                    site.output_weight,
                    _captured(gradient, "attention output gradient"),
                    layer=site.layer,
                    target_position=site.target_position,
                    head_indices=site.pattern.indices,
                )
            )

            logit_gradients: list[Tensor] = []
            if logit_indices:
                logits = output_logits(output)  # type: ignore[misc]
                if not isinstance(logits, Tensor) or logits.ndim != 1:
                    raise ValueError("output_logits must return a one-dimensional tensor")
                start = len(target_gradient.values)
                for offset, token_index in enumerate(logit_indices):
                    if token_index < -logits.numel() or token_index >= logits.numel():
                        raise IndexError(f"logit index {token_index} is out of bounds")
                    logits[token_index].backward(retain_graph=offset + 1 < len(logit_indices))
                captured_logit_gradients = target_gradient.values[start:]
                if len(captured_logit_gradients) != len(logit_indices):
                    raise RuntimeError("target feature gradient was not captured once per output logit")
                logit_gradients = [
                    _site_scalar(gradient, target_feature).detach() for gradient in captured_logit_gradients
                ]

        target_value = _site_scalar(_captured(target_activation, "target feature"), target_feature).detach()
        logits_artifact = logit_contributions(
            target_value,
            torch.stack(logit_gradients) if logit_gradients else torch.empty(0),
            token_indices=logit_indices,
        )
        target_position = _site_position(_captured(target_activation, "target feature"), target_feature)
        return CircuitLensArtifact(
            target_layer=target_feature.layer,
            target_feature_index=target_feature.target.indices[0],
            target_position=target_position,
            target_activation=float(target_value.cpu()),
            upstream_features=_rank(upstream, top_k=top_k, positive_only=positive_only),
            attention=_rank(attention, top_k=top_k, positive_only=positive_only),
            output_logits=_rank(logits_artifact, top_k=top_k, positive_only=positive_only),
        )
    finally:
        _restore_parameter_gradients(saved_gradients)


def _captured(capture: CapturedTarget, name: str) -> Tensor:
    if capture.value is None:
        raise RuntimeError(f"{name} target was not reached during model execution")
    return capture.value


def _site_scalar(value: Tensor, site: FeatureSite) -> Tensor:
    if value.numel() == 1:
        return value.reshape(())
    if site.position is None:
        raise ValueError("FeatureSite.position is required when the selected target is not scalar")
    axis = _normalized_axis(site.target.feature_axis, value.ndim)
    if len(site.position) != value.ndim - 1:
        raise ValueError("FeatureSite.position must index every non-feature axis")
    coordinate = list(site.position)
    coordinate.insert(axis, 0)
    try:
        return value[tuple(coordinate)]
    except IndexError as exc:
        raise IndexError("FeatureSite.position is outside the selected activation") from exc


def _site_position(value: Tensor, site: FeatureSite) -> tuple[int, ...]:
    if value.numel() == 1:
        return tuple(0 for _ in range(value.ndim - 1)) if value.ndim > 1 else ()
    assert site.position is not None  # validated by _site_scalar
    return site.position


def _rank(items: Sequence[Any], *, top_k: int | None, positive_only: bool) -> tuple[Any, ...]:
    selected = [item for item in items if not positive_only or item.score > 0]
    selected.sort(key=lambda item: (-abs(item.score), -item.score, repr(item)))
    return tuple(selected if top_k is None else selected[:top_k])


def _normalized_axis(axis: int, ndim: int) -> int:
    normalized = axis if axis >= 0 else ndim + axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"feature_axis {axis} is out of bounds for a {ndim}-D tensor")
    return normalized


def _without_singleton_batch(name: str, value: Tensor, expected_ndim: int) -> Tensor:
    if value.ndim == expected_ndim + 1:
        if value.shape[0] != 1:
            raise ValueError(f"{name} only supports a singleton batch dimension")
        value = value[0]
    if value.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, with an optional singleton batch")
    return value


def _coordinates(shape: torch.Size) -> tuple[tuple[int, ...], ...]:
    if not shape:
        return ((),)
    return tuple(product(*(range(size) for size in shape)))


def _save_parameter_gradients(model: nn.Module) -> list[tuple[nn.Parameter, Tensor | None]]:
    return [
        (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in model.parameters()
    ]


def _restore_parameter_gradients(saved: Sequence[tuple[nn.Parameter, Tensor | None]]) -> None:
    for parameter, gradient in saved:
        parameter.grad = gradient


__all__ = [
    "AttentionContributor",
    "AttentionSite",
    "AttributionConventions",
    "CircuitLensArtifact",
    "FeatureContributor",
    "FeatureSite",
    "LogitContributor",
    "attention_contributions",
    "attribute_feature_circuit",
    "feature_contributions",
    "logit_contributions",
]
