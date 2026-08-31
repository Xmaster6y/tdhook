"""Input-invariant analysis of transcoder features.

The functions in this module operate on weight tensors rather than model or
transcoder classes.  Callers are responsible for extracting the tensors from
their implementation and, when desired, supplying a forward-pass validator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch


@dataclass(frozen=True)
class ProjectionCandidate:
    """A statistically unusual token or feature projection."""

    index: int
    score: float
    z_score: float
    label: str | None = None
    activation: float | None = None
    validated: bool | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class LayerFeatureCandidate:
    """An outlying contribution from an earlier transcoder feature."""

    layer: int
    feature_index: int
    score: float
    z_score: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class WeightLensArtifact:
    """Serializable output of input-invariant feature analysis."""

    feature_layer: int
    feature_index: int
    embedding_positive: tuple[ProjectionCandidate, ...]
    embedding_negative: tuple[ProjectionCandidate, ...]
    earlier_features: tuple[LayerFeatureCandidate, ...]
    output_positive: tuple[ProjectionCandidate, ...]
    output_negative: tuple[ProjectionCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a nested representation accepted by :func:`json.dumps`."""

        return asdict(self)


def project_encoder_to_embeddings(embedding: torch.Tensor, feature_encoder: torch.Tensor) -> torch.Tensor:
    """Project one encoder vector into token-embedding rows.

    ``embedding`` has shape ``[vocabulary, model]`` and ``feature_encoder``
    has shape ``[model]``.
    """

    _require_matrix("embedding", embedding)
    _require_vector("feature_encoder", feature_encoder)
    _require_equal_dimensions("embedding model dimension", embedding.shape[1], feature_encoder.shape[0])
    return embedding @ feature_encoder


def project_encoder_to_decoder_features(
    earlier_feature_decoder: torch.Tensor, feature_encoder: torch.Tensor
) -> torch.Tensor:
    """Project one encoder vector into an earlier decoder dictionary."""

    _require_matrix("earlier_feature_decoder", earlier_feature_decoder)
    _require_vector("feature_encoder", feature_encoder)
    _require_equal_dimensions("decoder model dimension", earlier_feature_decoder.shape[1], feature_encoder.shape[0])
    return earlier_feature_decoder @ feature_encoder


def project_decoder_to_logits(feature_decoder: torch.Tensor, unembedding: torch.Tensor) -> torch.Tensor:
    """Project one decoder vector into output-vocabulary logits.

    ``feature_decoder`` has shape ``[model]`` and ``unembedding`` has shape
    ``[model, vocabulary]``.
    """

    _require_vector("feature_decoder", feature_decoder)
    _require_matrix("unembedding", unembedding)
    _require_equal_dimensions("unembedding model dimension", feature_decoder.shape[0], unembedding.shape[0])
    return feature_decoder @ unembedding


def select_projection_outliers(
    scores: torch.Tensor,
    *,
    threshold: float,
    largest: bool,
    pool_size: int | None = None,
    max_candidates: int | None = None,
    labels: Sequence[str] | None = None,
) -> tuple[ProjectionCandidate, ...]:
    """Select one tail of a one-dimensional projection by absolute z-score.

    Z-scores use the sample standard deviation, matching the public WeightLens
    reference.  Constant and singleton projections contain no outliers.
    """

    _require_vector("scores", scores)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if pool_size is not None and pool_size < 2:
        raise ValueError("pool_size must be at least two")
    if labels is not None and len(labels) != scores.numel():
        raise ValueError("labels must contain one entry per score")

    detached = scores.detach().to(dtype=torch.float64)
    if detached.numel() < 2:
        return ()
    if pool_size is None or pool_size >= detached.numel():
        pool_indices = torch.arange(detached.numel(), device=detached.device)
    else:
        pool_indices = torch.argsort(detached, descending=largest, stable=True)[:pool_size]
    pool_scores = detached[pool_indices]
    standard_deviation = pool_scores.std()
    if not torch.isfinite(standard_deviation) or standard_deviation == 0:
        return ()

    z_scores = (pool_scores - pool_scores.mean()) / standard_deviation
    mask = z_scores > threshold if largest else z_scores < -threshold
    pool_positions = mask.nonzero(as_tuple=True)[0]
    pool_positions = sorted(
        (int(position) for position in pool_positions),
        key=lambda position: (
            (-float(pool_scores[position]), int(pool_indices[position]))
            if largest
            else (float(pool_scores[position]), int(pool_indices[position]))
        ),
    )
    if max_candidates is not None:
        pool_positions = pool_positions[:max_candidates]
    return tuple(
        ProjectionCandidate(
            index=int(pool_indices[position]),
            score=float(pool_scores[position]),
            z_score=float(z_scores[position]),
            label=None if labels is None else labels[int(pool_indices[position])],
        )
        for position in pool_positions
    )


def validate_token_candidates(
    candidates: Sequence[ProjectionCandidate],
    validator: Callable[[Sequence[int]], torch.Tensor | Sequence[float]],
    *,
    activation_threshold: float = 0.0,
) -> tuple[ProjectionCandidate, ...]:
    """Attach forward-pass activations to token candidates.

    ``validator`` receives all candidate token IDs and returns one scalar
    activation per ID.  This keeps batching, BOS handling, and model execution
    outside TDHook core.
    """

    if not candidates:
        return ()
    activations = torch.as_tensor(validator([candidate.index for candidate in candidates])).detach().cpu()
    if activations.ndim != 1 or activations.numel() != len(candidates):
        raise ValueError("validator must return one scalar activation per candidate")
    return tuple(
        replace(
            candidate,
            activation=float(activation),
            validated=bool(activation > activation_threshold),
        )
        for candidate, activation in zip(candidates, activations, strict=True)
    )


def analyze_input_invariant_feature(
    *,
    feature_layer: int,
    feature_index: int,
    embedding: torch.Tensor,
    unembedding: torch.Tensor,
    feature_encoders: Sequence[torch.Tensor],
    feature_decoders: Sequence[torch.Tensor],
    token_outlier_threshold: float = 5.5,
    feature_outlier_threshold: float = 4.0,
    token_pool_size: int | None = 1000,
    feature_pool_size: int | None = 100,
    max_candidates: int | None = None,
    token_labels: Sequence[str] | None = None,
    token_validator: Callable[[Sequence[int]], torch.Tensor | Sequence[float]] | None = None,
    validation_threshold: float = 0.0,
) -> WeightLensArtifact:
    """Analyze one transcoder feature using input-invariant weight projections.

    Encoder matrices have shape ``[model, features]`` and decoder matrices
    shape ``[features, model]``.  The two sequences must describe the same
    ordered layers.  Only decoder dictionaries before ``feature_layer`` are
    considered as possible upstream features.
    """

    if feature_layer < 0 or feature_layer >= len(feature_encoders):
        raise IndexError("feature_layer is outside feature_encoders")
    if len(feature_encoders) != len(feature_decoders):
        raise ValueError("feature_encoders and feature_decoders must contain the same layers")

    encoder_matrix = feature_encoders[feature_layer]
    decoder_matrix = feature_decoders[feature_layer]
    _require_matrix("feature encoder", encoder_matrix)
    _require_matrix("feature decoder", decoder_matrix)
    if feature_index < 0 or feature_index >= encoder_matrix.shape[1] or feature_index >= decoder_matrix.shape[0]:
        raise IndexError("feature_index is outside the selected transcoder")

    encoder = encoder_matrix[:, feature_index]
    decoder = decoder_matrix[feature_index]
    embedding_scores = project_encoder_to_embeddings(embedding, encoder)
    output_scores = project_decoder_to_logits(decoder, unembedding)

    selection_kwargs: dict[str, Any] = {
        "threshold": token_outlier_threshold,
        "pool_size": token_pool_size,
        "max_candidates": max_candidates,
        "labels": token_labels,
    }
    embedding_positive = select_projection_outliers(embedding_scores, largest=True, **selection_kwargs)
    embedding_negative = select_projection_outliers(embedding_scores, largest=False, **selection_kwargs)
    output_positive = select_projection_outliers(output_scores, largest=True, **selection_kwargs)
    output_negative = select_projection_outliers(output_scores, largest=False, **selection_kwargs)

    if token_validator is not None:
        embedding_positive = validate_token_candidates(
            embedding_positive, token_validator, activation_threshold=validation_threshold
        )

    earlier_features: list[LayerFeatureCandidate] = []
    for layer, earlier_decoder in enumerate(feature_decoders[:feature_layer]):
        scores = project_encoder_to_decoder_features(earlier_decoder, encoder)
        for candidate in select_projection_outliers(
            scores,
            threshold=feature_outlier_threshold,
            largest=True,
            pool_size=feature_pool_size,
            max_candidates=max_candidates,
        ):
            earlier_features.append(
                LayerFeatureCandidate(
                    layer=layer,
                    feature_index=candidate.index,
                    score=candidate.score,
                    z_score=candidate.z_score,
                )
            )
    earlier_features.sort(key=lambda candidate: (-candidate.score, candidate.layer, candidate.feature_index))

    return WeightLensArtifact(
        feature_layer=feature_layer,
        feature_index=feature_index,
        embedding_positive=embedding_positive,
        embedding_negative=embedding_negative,
        earlier_features=tuple(earlier_features),
        output_positive=output_positive,
        output_negative=output_negative,
    )


def _require_vector(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")


def _require_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")


def _require_equal_dimensions(name: str, left: int, right: int) -> None:
    if left != right:
        raise ValueError(f"{name} mismatch: {left} != {right}")
