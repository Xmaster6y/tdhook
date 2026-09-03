"""
Module for attribution methods.
"""

from .activation_maximisation import ActivationMaximisation
from .circuit_clustering import (
    CircuitClusters,
    CircuitContributor,
    CircuitSignature,
    build_circuit_signature,
    cluster_circuit_artifacts,
    dbscan_circuit_signatures,
    filter_contributor_frequency,
    jaccard_distances,
    sample_activation_indices,
)
from .circuit_lens import (
    AttentionContributor,
    AttentionSite,
    AttributionConventions,
    CircuitLensArtifact,
    FeatureContributor,
    FeatureSite,
    LogitContributor,
    attention_contributions,
    attribute_feature_circuit,
    feature_contributions,
    logit_contributions,
)
from .grad_cam import GradCAM
from .guided_backpropagation import GuidedBackpropagation
from .integrated_gradients import IntegratedGradients
from .lrp import LRP
from .saliency import Saliency

__all__ = [
    "LRP",
    "ActivationMaximisation",
    "AttentionContributor",
    "AttentionSite",
    "AttributionConventions",
    "CircuitClusters",
    "CircuitContributor",
    "CircuitLensArtifact",
    "CircuitSignature",
    "FeatureContributor",
    "FeatureSite",
    "GradCAM",
    "GuidedBackpropagation",
    "IntegratedGradients",
    "LogitContributor",
    "Saliency",
    "attention_contributions",
    "attribute_feature_circuit",
    "build_circuit_signature",
    "cluster_circuit_artifacts",
    "dbscan_circuit_signatures",
    "feature_contributions",
    "filter_contributor_frequency",
    "jaccard_distances",
    "logit_contributions",
    "sample_activation_indices",
]

# TODO: Implement Occlusion
