"""
Module for latent methods.
"""

from .activation_caching import ActivationCaching
from .activation_patching import ActivationPatching
from .dimension_estimation import LocalKnnDimensionEstimator, TwoNnDimensionEstimator
from .probing import (
    BilinearProbe,
    BilinearProbeManager,
    LinearEstimator,
    LowRankBilinearEstimator,
    MeanDifferenceClassifier,
    Probing,
    Probe,
    ProbeManager,
)
from .steering_vectors import SteeringVectors, ActivationAddition

__all__ = [
    "ActivationAddition",
    "ActivationCaching",
    "ActivationPatching",
    "BilinearProbe",
    "BilinearProbeManager",
    "LinearEstimator",
    "LowRankBilinearEstimator",
    "LocalKnnDimensionEstimator",
    "MeanDifferenceClassifier",
    "Probe",
    "ProbeManager",
    "Probing",
    "SteeringVectors",
    "TwoNnDimensionEstimator",
]

# TODO: Implement ATP*
