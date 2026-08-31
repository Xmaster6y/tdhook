"""
Module for weight analysis methods.
"""

from .adapters import Adapters
from .pruning import Pruning
from .task_vectors import TaskVectors
from .weight_lens import (
    LayerFeatureCandidate,
    ProjectionCandidate,
    WeightLensArtifact,
    analyze_input_invariant_feature,
    project_decoder_to_logits,
    project_encoder_to_decoder_features,
    project_encoder_to_embeddings,
    select_projection_outliers,
    validate_token_candidates,
)

__all__ = [
    "Adapters",
    "Pruning",
    "TaskVectors",
    "LayerFeatureCandidate",
    "ProjectionCandidate",
    "WeightLensArtifact",
    "analyze_input_invariant_feature",
    "project_decoder_to_logits",
    "project_encoder_to_decoder_features",
    "project_encoder_to_embeddings",
    "select_projection_outliers",
    "validate_token_candidates",
]

# TODO: Implment crosscoders
# TODO: Implement circuits tracer from Anthropic
