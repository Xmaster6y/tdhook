import json

import pytest
import torch

from tdhook.weights import (
    LayerFeatureCandidate,
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


def test_candidate_artifacts_are_serializable():
    projection = ProjectionCandidate(index=2, score=3.0, z_score=4.0, label="token")
    feature = LayerFeatureCandidate(layer=1, feature_index=2, score=3.0, z_score=4.0)

    assert projection.to_dict()["label"] == "token"
    assert feature.to_dict() == {"layer": 1, "feature_index": 2, "score": 3.0, "z_score": 4.0}


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
        input_token_labels=["zero", "x", "y", "positive", "negative"],
        output_token_labels=["zero", "x", "y", "positive", "negative"],
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


def test_analysis_supports_distinct_vocabularies_without_autograd_graphs():
    embedding = torch.tensor([[0.0, 0.0], [10.0, 0.0], [-9.0, 0.0]], requires_grad=True)
    unembedding = torch.tensor([[0.0, 12.0, 0.0, -11.0], [0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    encoders = (torch.tensor([[1.0], [0.0]], requires_grad=True),)
    decoders = (torch.tensor([[1.0, 0.0]], requires_grad=True),)
    grad_enabled = []

    artifact = analyze_input_invariant_feature(
        feature_layer=0,
        feature_index=0,
        embedding=embedding,
        unembedding=unembedding,
        feature_encoders=encoders,
        feature_decoders=decoders,
        token_outlier_threshold=0.9,
        input_token_labels=["zero", "input-positive", "input-negative"],
        output_token_labels=["zero", "output-positive", "zero-2", "output-negative"],
        token_validator=lambda token_ids: grad_enabled.append(torch.is_grad_enabled()) or torch.ones(len(token_ids)),
    )

    assert artifact.embedding_positive[0].label == "input-positive"
    assert artifact.output_positive[0].label == "output-positive"
    assert grad_enabled == [False]
    assert all(tensor.grad is None for tensor in (embedding, unembedding, *encoders, *decoders))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": -1.0}, "threshold"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"pool_size": 1}, "pool_size"),
        ({"labels": ["too", "short"]}, "labels"),
    ],
)
def test_outlier_selection_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_projection_outliers(torch.arange(3.0), largest=True, **({"threshold": 0.0} | kwargs))


def test_outlier_selection_handles_degenerate_inputs_and_candidate_limits():
    assert select_projection_outliers(torch.tensor([1.0]), threshold=0.0, largest=True) == ()
    assert select_projection_outliers(torch.ones(3), threshold=0.0, largest=True) == ()

    candidates = select_projection_outliers(
        torch.tensor([0.0, 3.0, 4.0, 5.0]), threshold=0.0, largest=True, max_candidates=1
    )
    assert [candidate.index for candidate in candidates] == [3]
    assert validate_token_candidates((), lambda _: pytest.fail("validator should not run")) == ()


def test_analysis_rejects_invalid_feature_coordinates_and_layer_collections():
    valid_encoder = torch.ones(2, 1)
    valid_decoder = torch.ones(1, 2)
    common = {
        "embedding": torch.ones(3, 2),
        "unembedding": torch.ones(2, 3),
        "feature_encoders": (valid_encoder,),
        "feature_decoders": (valid_decoder,),
    }

    with pytest.raises(IndexError, match="feature_layer"):
        analyze_input_invariant_feature(feature_layer=1, feature_index=0, **common)
    with pytest.raises(ValueError, match="same layers"):
        analyze_input_invariant_feature(
            feature_layer=0,
            feature_index=0,
            **(common | {"feature_decoders": (valid_decoder, valid_decoder)}),
        )
    with pytest.raises(IndexError, match="feature_index"):
        analyze_input_invariant_feature(feature_layer=0, feature_index=1, **common)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: project_encoder_to_embeddings(torch.ones(2), torch.ones(2)),
        lambda: project_encoder_to_embeddings(torch.ones(2, 2), torch.ones(1, 2)),
        lambda: project_encoder_to_decoder_features(torch.ones(2), torch.ones(2)),
        lambda: project_decoder_to_logits(torch.ones(1, 2), torch.ones(2, 2)),
        lambda: project_decoder_to_logits(torch.ones(2), torch.ones(2)),
        lambda: select_projection_outliers(torch.ones(2, 2), threshold=0.0, largest=True),
    ],
)
def test_public_projection_apis_reject_wrong_ranks(operation):
    with pytest.raises(ValueError, match="dimensional"):
        operation()
