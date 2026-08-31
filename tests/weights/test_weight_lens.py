import json

import pytest
import torch

from tdhook.weights import (
    ProjectionCandidate,
    analyze_input_invariant_feature,
    project_decoder_to_logits,
    project_encoder_to_decoder_features,
    project_encoder_to_embeddings,
    select_projection_outliers,
    validate_token_candidates,
)


def test_weight_projections_match_reference_formulas():
    embedding = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]])
    encoder = torch.tensor([2.0, -1.0])
    earlier_decoder = torch.tensor([[1.0, 0.0], [0.5, 2.0]])
    decoder = torch.tensor([1.5, -2.0])
    unembedding = torch.tensor([[1.0, 0.0, 2.0], [0.5, -1.0, 3.0]])

    torch.testing.assert_close(project_encoder_to_embeddings(embedding, encoder), embedding @ encoder)
    torch.testing.assert_close(
        project_encoder_to_decoder_features(earlier_decoder, encoder), earlier_decoder @ encoder
    )
    torch.testing.assert_close(project_decoder_to_logits(decoder, unembedding), decoder @ unembedding)


def test_outlier_selection_is_deterministic_and_labelled():
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, -9.0])

    positive = select_projection_outliers(
        scores, threshold=1.5, largest=True, labels=["a", "b", "c", "d", "winner", "loser"]
    )
    negative = select_projection_outliers(scores, threshold=1.5, largest=False)

    assert [(candidate.index, candidate.label) for candidate in positive] == [(4, "winner")]
    assert [candidate.index for candidate in negative] == [5]
    assert positive[0].z_score > 1.5
    assert negative[0].z_score < -1.5


def test_outlier_selection_can_match_reference_top_k_pooling():
    scores = torch.tensor([100.0, 10.0, 9.0, 8.0, 7.0, -100.0])

    positive = select_projection_outliers(scores, threshold=1.0, largest=True, pool_size=5)
    negative = select_projection_outliers(scores, threshold=1.0, largest=False, pool_size=5)

    assert [candidate.index for candidate in positive] == [0]
    assert [candidate.index for candidate in negative] == [5]


def test_validation_attaches_forward_activations():
    candidates = (
        ProjectionCandidate(index=7, score=3.0, z_score=2.0),
        ProjectionCandidate(index=9, score=2.0, z_score=1.5),
    )

    validated = validate_token_candidates(candidates, lambda token_ids: [token_ids[0] - 6.0, -0.5])

    assert [(candidate.activation, candidate.validated) for candidate in validated] == [(1.0, True), (-0.5, False)]


def test_selected_public_feature_agrees_with_reference_implementation():
    # A toy selected feature using the public WeightLens tensor conventions.
    embedding = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [8.0, 0.0], [-7.0, 0.0]])
    unembedding = embedding.T.clone()
    feature_encoders = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0], [0.0]]),
    )
    feature_decoders = (
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [9.0, 0.0], [-8.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )

    artifact = analyze_input_invariant_feature(
        feature_layer=1,
        feature_index=0,
        embedding=embedding,
        unembedding=unembedding,
        feature_encoders=feature_encoders,
        feature_decoders=feature_decoders,
        token_outlier_threshold=1.25,
        feature_outlier_threshold=1.25,
        token_labels=["zero", "x", "y", "positive", "negative"],
        token_validator=lambda token_ids: [1.0 if token_id == 3 else -1.0 for token_id in token_ids],
    )

    # These are the three projections used by the reference implementation.
    torch.testing.assert_close(embedding @ feature_encoders[1][:, 0], torch.tensor([0.0, 1.0, 0.0, 8.0, -7.0]))
    torch.testing.assert_close(
        feature_decoders[0] @ feature_encoders[1][:, 0], torch.tensor([0.0, 0.0, 0.0, 9.0, -8.0])
    )
    torch.testing.assert_close(feature_decoders[1][0] @ unembedding, torch.tensor([0.0, 1.0, 0.0, 8.0, -7.0]))
    assert [(candidate.index, candidate.validated) for candidate in artifact.embedding_positive] == [(3, True)]
    assert [candidate.index for candidate in artifact.embedding_negative] == [4]
    assert [(candidate.layer, candidate.feature_index) for candidate in artifact.earlier_features] == [(0, 3)]
    assert [candidate.index for candidate in artifact.output_positive] == [3]
    assert [candidate.index for candidate in artifact.output_negative] == [4]
    json.dumps(artifact.to_dict(), sort_keys=True)


def test_analysis_rejects_incompatible_shapes_and_validator_results():
    with pytest.raises(ValueError, match="model dimension mismatch"):
        project_encoder_to_embeddings(torch.ones(3, 2), torch.ones(3))

    candidate = ProjectionCandidate(index=0, score=1.0, z_score=2.0)
    with pytest.raises(ValueError, match="one scalar activation"):
        validate_token_candidates((candidate,), lambda _: torch.ones(2))
