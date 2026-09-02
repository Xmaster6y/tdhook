"""Sampling and clustering utilities for CircuitLens artifacts.

Circuit signatures follow the CircuitLens paper: upstream features are
identified by ``(layer, feature)`` and attention contributors by
``(layer, head, relative source position)``.  Clustering uses Jaccard distance
between the resulting sets.  Scikit-learn is imported only when DBSCAN is
requested, so artifact construction, filtering, and distance computation keep
working without the optional dependency.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal

import torch

from .circuit_lens import CircuitLensArtifact


@dataclass(frozen=True, order=True)
class CircuitContributor:
    """Stable identity of one contributor in a circuit signature."""

    kind: Literal["attention", "feature"]
    layer: int
    index: int
    relative_token: int | None = None


CircuitSignature = frozenset[CircuitContributor]


@dataclass(frozen=True)
class CircuitClusters:
    """Intermediate signatures, distances, and final DBSCAN labels."""

    signatures: tuple[CircuitSignature, ...]
    filtered_signatures: tuple[CircuitSignature, ...]
    distances: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]


def sample_activation_indices(
    activations: Sequence[float] | torch.Tensor,
    sample_size: int,
    *,
    seed: int = 0,
    bins: int = 20,
    alpha: float = 0.9,
) -> tuple[int, ...]:
    """Sample activation indices without replacement across quantile bins.

    Each activation in quantile bin ``b`` receives weight
    ``1 / count(b) ** alpha``.  A local CPU generator makes the result
    reproducible without changing PyTorch's global random state.
    """

    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an int")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise TypeError("bins must be an int")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    values = torch.as_tensor(activations, dtype=torch.float64, device="cpu").detach()
    if values.ndim != 1:
        raise ValueError("activations must be one-dimensional")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("activations must be finite")
    if sample_size > values.numel():
        raise ValueError("sample_size cannot exceed the number of activations")
    if sample_size == 0:
        return ()

    quantiles = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    boundaries = torch.quantile(values, quantiles)
    # Put values equal to repeated boundaries in the lower bin.  Sparse/ReLU
    # activations otherwise place the zero mass and positive tail together in
    # the last bin, eliminating the intended inverse-frequency weighting.
    bin_indices = torch.bucketize(values, boundaries[1:-1], right=False)
    counts = torch.bincount(bin_indices, minlength=bins).to(torch.float64)
    weights = counts[bin_indices].pow(-alpha)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(int(index) for index in torch.multinomial(weights, sample_size, False, generator=generator))


def build_circuit_signature(
    artifact: CircuitLensArtifact,
    *,
    min_abs_score: float = 0.0,
) -> CircuitSignature:
    """Build a contributor set from one :class:`CircuitLensArtifact`.

    Contributors with absolute score equal to or below ``min_abs_score`` are
    omitted.  Attention source positions are made relative to the target token
    so equivalent circuits at different absolute sequence positions match.
    """

    if not isfinite(min_abs_score) or min_abs_score < 0:
        raise ValueError("min_abs_score must be finite and non-negative")

    contributors = {
        CircuitContributor("feature", item.layer, item.feature_index)
        for item in artifact.upstream_features
        if abs(item.score) > min_abs_score
    }
    contributors.update(
        CircuitContributor("attention", item.layer, item.head_index, item.source_token - item.target_token)
        for item in artifact.attention
        if abs(item.score) > min_abs_score
    )
    return frozenset(contributors)


def filter_contributor_frequency(
    signatures: Sequence[CircuitSignature],
    *,
    min_frequency: float,
) -> tuple[CircuitSignature, ...]:
    """Keep contributors present in at least ``min_frequency`` of inputs."""

    if not isfinite(min_frequency) or not 0 <= min_frequency <= 1:
        raise ValueError("min_frequency must be between 0 and 1")
    if not signatures:
        return ()

    minimum_count = ceil(min_frequency * len(signatures))
    counts = Counter(contributor for signature in signatures for contributor in signature)
    retained = {contributor for contributor, count in counts.items() if count >= minimum_count}
    return tuple(frozenset(signature & retained) for signature in signatures)


def jaccard_distances(signatures: Sequence[CircuitSignature]) -> tuple[tuple[float, ...], ...]:
    """Return a square pairwise Jaccard-distance matrix.

    Two empty signatures have distance zero; one empty and one non-empty
    signature have distance one.
    """

    rows: list[tuple[float, ...]] = []
    for left in signatures:
        row: list[float] = []
        for right in signatures:
            union = left | right
            row.append(0.0 if not union else 1.0 - len(left & right) / len(union))
        rows.append(tuple(row))
    return tuple(rows)


def dbscan_circuit_signatures(
    signatures: Sequence[CircuitSignature],
    *,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[int, ...]:
    """Cluster signatures with DBSCAN over precomputed Jaccard distances."""

    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(min_samples, int) or isinstance(min_samples, bool):
        raise TypeError("min_samples must be an int")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if not signatures:
        return ()

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        raise ImportError("Circuit clustering requires scikit-learn; install tdhook[circuit-clustering]") from error

    raw_labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(
        jaccard_distances(signatures)
    )
    return _stable_cluster_labels(raw_labels, signatures)


def cluster_circuit_artifacts(
    artifacts: Iterable[CircuitLensArtifact],
    *,
    min_frequency: float = 0.05,
    min_abs_score: float = 0.0,
    eps: float = 0.5,
    min_samples: int = 5,
) -> CircuitClusters:
    """Build, frequency-filter, and cluster signatures from CircuitLens artifacts."""

    signatures = tuple(build_circuit_signature(item, min_abs_score=min_abs_score) for item in artifacts)
    filtered = filter_contributor_frequency(signatures, min_frequency=min_frequency)
    distances = jaccard_distances(filtered)
    labels = dbscan_circuit_signatures(filtered, eps=eps, min_samples=min_samples)
    return CircuitClusters(signatures, filtered, distances, labels)


def _stable_cluster_labels(labels: Sequence[int], signatures: Sequence[CircuitSignature]) -> tuple[int, ...]:
    cluster_keys: dict[int, tuple[tuple[CircuitContributor, ...], ...]] = {}
    for label in {int(value) for value in labels}:
        if label != -1:
            members = (
                tuple(sorted(signature)) for signature, value in zip(signatures, labels, strict=True) if value == label
            )
            cluster_keys[label] = tuple(sorted(members))
    ordered_clusters = sorted(cluster_keys.items(), key=lambda item: item[1])
    remapping = {label: stable for stable, (label, _) in enumerate(ordered_clusters)}
    return tuple(-1 if int(label) == -1 else remapping[int(label)] for label in labels)


__all__ = [
    "CircuitClusters",
    "CircuitContributor",
    "CircuitSignature",
    "build_circuit_signature",
    "cluster_circuit_artifacts",
    "dbscan_circuit_signatures",
    "filter_contributor_frequency",
    "jaccard_distances",
    "sample_activation_indices",
]
