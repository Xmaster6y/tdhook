"""
Module for attribution methods.
"""

# Import order is intentional: legacy modules import Saliency from this package.
# ruff: noqa: I001

from .lrp import LRP
from .saliency import Saliency
from .grad_cam import GradCAM
from .guided_backpropagation import GuidedBackpropagation
from .activation_maximisation import ActivationMaximisation
from .integrated_gradients import IntegratedGradients
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

__all__ = [
    "LRP",
    "ActivationMaximisation",
    "AttentionContributor",
    "AttentionSite",
    "AttributionConventions",
    "CircuitLensArtifact",
    "FeatureContributor",
    "FeatureSite",
    "GradCAM",
    "GuidedBackpropagation",
    "IntegratedGradients",
    "LogitContributor",
    "Saliency",
    "attention_contributions",
    "attribute_feature_circuit",
    "feature_contributions",
    "logit_contributions",
]

# TODO: Implement Occlusion
