import torch
from torch import nn
from typing import Optional, Iterable, Callable, Generator
from tensordict import TensorDict
from contextlib import contextmanager

from tdhook.methods import Method, BoundMethod
from tdhook.modules import BoundModule
from tdhook.runtime import HookProgram, HookSpec, temporary_module_state


class TaskVectorsBinding(BoundMethod):
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
        if self._bound_module is None or not self._in_context:
            raise RuntimeError("Cannot compute alpha outside of context")

        adequate_values = []
        for value in self.alphas:
            with self._bound_module.with_applied_vectors(vector, alpha=value) as module:
                if self.get_control_adequacy(module):
                    adequate_values.append((value, self.get_test_accuracy(module)))
        if not adequate_values:
            raise RuntimeError("No value satisfies the control adequacy criterion")
        return max(adequate_values, key=lambda x: x[1])[0]


class TaskVectorsModule(BoundModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._weights = TensorDict.from_module(self.module)
        self._applied_program = HookProgram()

    @property
    def applied_program(self) -> HookProgram:
        """Return the most recent temporary parameter-state program."""

        return self._applied_program

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
            if self.binding is None or not isinstance(self.binding, TaskVectorsBinding):
                raise RuntimeError("Module is not bound with TaskVectors")
            alpha = self.binding.compute_alpha(sum(vectors))
        return self._weights + sum(vectors) * alpha

    @contextmanager
    def with_applied_vectors(
        self, *vectors: TensorDict, alpha: Optional[float] = None
    ) -> Generator[nn.Module, None, None]:
        """Apply vectors to model"""
        with temporary_module_state(
            self.module,
            self.get_weights(*vectors, alpha=alpha),
            HookSpec("", "replace_parameters", None),
        ) as program:
            self._applied_program = program
            yield self


class TaskVectors(Method):
    """
    Task vectors :cite:`Ilharco2022EditingMW`.
    """

    _binding_class = TaskVectorsBinding
    _bound_module_class = TaskVectorsModule

    def __init__(
        self,
        alphas: Iterable[float],
        get_test_accuracy: Callable[[nn.Module], float],
        get_control_adequacy: Callable[[nn.Module], bool],
    ):
        super().__init__()
        self._binding_kwargs.update(
            {
                "alphas": alphas,
                "get_test_accuracy": get_test_accuracy,
                "get_control_adequacy": get_control_adequacy,
            }
        )
