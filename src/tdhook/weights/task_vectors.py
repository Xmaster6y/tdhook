"""
Task vectors for weight interpretability
"""

from torch import nn
from typing import Optional, Iterable, Callable, Generator
from tensordict import TensorDict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ComputeAlphaConfig:
    """Configuration for computing alpha"""

    values: Iterable[float]
    get_test_accuracy: Callable[[nn.Module], float]
    get_control_adequacy: Callable[[nn.Module], bool]


@dataclass
class TaskVectorsConfig:
    """Configuration for task vector analysis"""

    compute_alpha_config: Optional[ComputeAlphaConfig] = None


class TaskVectors:
    """Task vectors analysis"""

    def __init__(
        self,
        config: TaskVectorsConfig,
        pretrained_module: nn.Module,
    ):
        self.config = config
        self._pretrained_module = pretrained_module
        self._pretrained_weights = TensorDict.from_module(pretrained_module)

    def get_task_vector(self, finetuned_module: nn.Module) -> TensorDict:
        """Compute task vector"""
        return TensorDict.from_module(finetuned_module) - self._pretrained_weights

    def get_forget_vector(self, finetuned_module: nn.Module) -> TensorDict:
        """Compute forget vector"""
        return -self.get_task_vector(finetuned_module)

    def get_weights(self, *vectors: TensorDict, alpha: Optional[float] = None) -> TensorDict:
        """Get weights"""
        if alpha is None:
            alpha = self.compute_alpha(sum(vectors))
        return self._pretrained_weights + sum(vectors) * alpha

    def compute_alpha(self, vector: TensorDict) -> float:
        """Compute alpha"""
        if self.config.compute_alpha_config is None:
            raise ValueError("compute_alpha_config is not set and alpha was not provided")

        adequate_values = []
        for value in self.config.compute_alpha_config.values:
            with self.with_applied_vectors(vector, alpha=value) as module:
                if self.config.compute_alpha_config.get_control_adequacy(module):
                    adequate_values.append((value, self.config.compute_alpha_config.get_test_accuracy(module)))
        if not adequate_values:
            raise ValueError("No value satisfies the control adequacy criterion")
        return max(adequate_values, key=lambda x: x[1])[0]

    @contextmanager
    def with_applied_vectors(
        self, *vectors: TensorDict, alpha: Optional[float] = None
    ) -> Generator[nn.Module, None, None]:
        """Apply vectors to model"""
        if alpha is None:
            alpha = self.compute_alpha(sum(vectors))
        with (self._pretrained_weights + sum(vectors) * alpha).to_module(self._pretrained_module):
            yield self._pretrained_module
