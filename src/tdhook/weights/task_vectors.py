"""
Task vectors for weight interpretability
"""

import torch
from torch import nn
from typing import Optional, Iterable, Callable, Generator, List
from tensordict import TensorDict
from contextlib import contextmanager

from tdhook.contexts import HookingContextFactory, HookingContext
from tdhook.module import HookedModule


class TaskVectorsContext(HookingContext):
    def __init__(
        self,
        *args,
        alphas: Iterable[float],
        get_test_accuracy: Callable[[nn.Module], float],
        get_control_adequacy: Callable[[nn.Module], bool],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alphas = alphas
        self.get_test_accuracy = get_test_accuracy
        self.get_control_adequacy = get_control_adequacy

    def compute_alpha(self, vector: TensorDict) -> float:
        """Compute alpha"""
        if self._hooked_module is None or not self._in_context:
            raise RuntimeError("Cannot compute alpha outside of context")

        adequate_values = []
        for value in self.alphas:
            with self._hooked_module.with_applied_vectors(vector, alpha=value) as module:
                if self.get_control_adequacy(module):
                    adequate_values.append((value, self.get_test_accuracy(module)))
        if not adequate_values:
            raise RuntimeError("No value satisfies the control adequacy criterion")
        return max(adequate_values, key=lambda x: x[1])[0]


class TaskVectorsModule(HookedModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._weights = TensorDict.from_module(self.module)

    @torch.no_grad()
    def get_task_vector(self, finetuned_module: nn.Module) -> TensorDict:
        """Compute task vector"""
        return TensorDict.from_module(finetuned_module) - self._weights

    @torch.no_grad()
    def get_forget_vector(self, finetuned_module: nn.Module) -> TensorDict:
        """Compute forget vector"""
        return -self.get_task_vector(finetuned_module)

    @torch.no_grad()
    def get_weights(self, *vectors: TensorDict, alpha: Optional[float] = None) -> TensorDict:
        """Get weights"""
        if alpha is None:
            if self.hooking_context is None or not isinstance(self.hooking_context, TaskVectorsContext):
                raise RuntimeError("Module was not prepared with TaskVectors")
            alpha = self.hooking_context.compute_alpha(sum(vectors))
        return self._weights + sum(vectors) * alpha

    @contextmanager
    def with_applied_vectors(
        self, *vectors: TensorDict, alpha: Optional[float] = None
    ) -> Generator[nn.Module, None, None]:
        """Apply vectors to model"""
        with self.get_weights(*vectors, alpha=alpha).to_module(self.module):
            yield self


class TaskVectors(HookingContextFactory):
    _hooking_context_class = TaskVectorsContext
    _hooked_module_class = TaskVectorsModule

    def __init__(
        self,
        alphas: Iterable[float],
        get_test_accuracy: Callable[[nn.Module], float],
        get_control_adequacy: Callable[[nn.Module], bool],
    ):
        self._context_kwargs = {
            "alphas": alphas,
            "get_test_accuracy": get_test_accuracy,
            "get_control_adequacy": get_control_adequacy,
        }

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ) -> "TaskVectorsContext":
        """
        Prepare the module for execution.
        """

        return self._hooking_context_class(self, module, in_keys, out_keys, **self._context_kwargs)
