from __future__ import annotations

import pytest
import torch

from tdhook.attribution import (
    AttentionContributor,
    CircuitContributor,
    CircuitLensArtifact,
    FeatureContributor,
    build_circuit_signature,
    cluster_circuit_artifacts,
    dbscan_circuit_signatures,
    filter_contributor_frequency,
    jaccard_distances,
    sample_activation_indices,
)


def artifact(
    *,
    features: tuple[FeatureContributor, ...] = (),
    attention: tuple[AttentionContributor, ...] = (),
) -> CircuitLensArtifact:
    return CircuitLensArtifact(2, 7, (0,), 1.0, features, attention, ())


def test_build_signature_consumes_circuit_lens_artifact_and_uses_relative_tokens():
    signature = build_circuit_signature(
        artifact(
            features=(FeatureContributor(1, 3, (8,), 0.5), FeatureContributor(1, 4, (8,), 0.0)),
            attention=(AttentionContributor(0, 2, 6, 8, -0.25),),
        )
    )

    assert signature == frozenset(
        {
            CircuitContributor("feature", 1, 3),
            CircuitContributor("attention", 0, 2, -2),
        }
    )


def test_empty_signatures_have_zero_jaccard_distance():
    assert build_circuit_signature(artifact()) == frozenset()
    assert jaccard_distances((frozenset(), frozenset())) == ((0.0, 0.0), (0.0, 0.0))


def test_frequency_filter_counts_presence_per_input():
    common = CircuitContributor("feature", 0, 1)
    rare = CircuitContributor("feature", 0, 2)
    signatures = (frozenset({common, rare}), frozenset({common}), frozenset())

    assert filter_contributor_frequency(signatures, min_frequency=2 / 3) == (
        frozenset({common}),
        frozenset({common}),
        frozenset(),
    )


def test_dbscan_marks_isolated_signature_as_noise_and_labels_are_stable():
    a = CircuitContributor("feature", 0, 1)
    b = CircuitContributor("feature", 0, 2)
    signatures = (frozenset({a}), frozenset({a}), frozenset({b}))

    assert dbscan_circuit_signatures(signatures, eps=0.01, min_samples=2) == (0, 0, -1)
    assert dbscan_circuit_signatures(signatures, eps=0.01, min_samples=2) == (0, 0, -1)


def test_cluster_ids_are_canonical_across_input_order():
    a = frozenset({CircuitContributor("feature", 0, 1)})
    z = frozenset({CircuitContributor("feature", 0, 9)})

    assert dbscan_circuit_signatures((z, z, a, a), eps=0.01, min_samples=2) == (1, 1, 0, 0)
    assert dbscan_circuit_signatures((a, z, a, z), eps=0.01, min_samples=2) == (0, 1, 0, 1)


def test_cluster_pipeline_exposes_filtered_signatures_and_distances():
    common = FeatureContributor(0, 1, (0,), 1.0)
    rare = FeatureContributor(0, 2, (0,), 1.0)

    result = cluster_circuit_artifacts(
        (artifact(features=(common, rare)), artifact(features=(common,))),
        min_frequency=1.0,
        eps=0.01,
        min_samples=2,
    )

    assert result.filtered_signatures == (result.filtered_signatures[0],) * 2
    assert result.distances == ((0.0, 0.0), (0.0, 0.0))
    assert result.labels == (0, 0)


def test_activation_sampling_is_seeded_without_replacement():
    activations = [float(index) for index in range(100)]

    first = sample_activation_indices(activations, 20, seed=42, bins=10)
    second = sample_activation_indices(activations, 20, seed=42, bins=10)

    assert first == second
    assert len(set(first)) == 20


def test_activation_sampling_separates_sparse_zero_mass_from_positive_tail(monkeypatch):
    captured_weights = None

    def capture_weights(weights, sample_size, replacement, *, generator):
        nonlocal captured_weights
        captured_weights = weights
        return torch.arange(sample_size)

    monkeypatch.setattr(torch, "multinomial", capture_weights)
    sample_activation_indices([0.0] * 100 + [1.0] * 5, 2)

    assert captured_weights is not None
    assert captured_weights[100] > captured_weights[0]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: sample_activation_indices([1.0], 2), "cannot exceed"),
        (lambda: sample_activation_indices([float("nan")], 1), "finite"),
        (lambda: filter_contributor_frequency((), min_frequency=1.1), "between 0 and 1"),
        (lambda: build_circuit_signature(artifact(), min_abs_score=-1), "non-negative"),
    ],
)
def test_invalid_clustering_inputs_are_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()
