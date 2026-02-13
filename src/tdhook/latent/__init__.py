"""
Module for latent methods.
"""

from .activation_caching import ActivationCaching
from .activation_patching import ActivationPatching
from .dimension_estimation import LocalKnnDimensionEstimator, TwoNnDimensionEstimator
from .probing import Probing
from .steering_vectors import SteeringVectors, ActivationAddition

__all__ = [
    "ActivationAddition",
    "ActivationCaching",
    "ActivationPatching",
    "LocalKnnDimensionEstimator",
    "Probing",
    "SteeringVectors",
    "TwoNnDimensionEstimator",
]

# TODO: Implement ATP*
