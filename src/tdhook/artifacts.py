"""
Artifacts
"""

import torch
from typing import Any, Iterator
import abc

from tensordict import TensorDict


class CircuitMeta(abc.ABCMeta):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, TensorDict) and instance.dtype == torch.bool


class Circuit(TensorDict, metaclass=CircuitMeta):
    """Circuit artifact"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to(torch.bool, inplace=True)

    def get_circuit_weights(self, weights: TensorDict) -> TensorDict:
        """Get circuit weights"""
        return self * weights

    @staticmethod
    def merge(circuits: Iterator["Circuit"]) -> "Circuit":
        """Merge circuits"""
        return Circuit.stack(circuits).any(dim=0)

    @staticmethod
    def intersect(circuits: Iterator["Circuit"]) -> "Circuit":
        """Intersect circuits"""
        return Circuit.stack(circuits).all(dim=0)

    @staticmethod
    def disjoint(circuits: Iterator["Circuit"]) -> bool:
        """Disjoint predicate"""
        return Circuit.stack(circuits).all(dim=0).any()
