"""Interactive capture and intervention through an explicit hook lifecycle."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal
import weakref

import torch
from torch import Tensor, nn

from tdhook.hooks import HookDirection, register_hook_to_module
from tdhook.targets import Target


HookOperation = Literal["capture", "replace"]


@dataclass(frozen=True)
class HookSpec:
    """One hook operation validated and installed by a :class:`HookSession`."""

    target: Target
    operation: HookOperation
    direction: HookDirection | None
    prepend: bool = False


@dataclass(frozen=True)
class HookProgram:
    """The ordered, model-free description of a session's installed operations."""

    hooks: tuple[HookSpec, ...] = ()


@dataclass
class CapturedTarget:
    """Mutable result populated when a capture hook observes its target."""

    value: Tensor | None = None


class HookSession:
    """Own an ordered set of temporary capture and replacement operations.

    The session keeps only a weak reference to the caller-owned model. Hooks
    and parameter replacements are active inside the context and are removed
    or restored in reverse registration order on every exit path.
    """

    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("HookSession requires a torch.nn.Module")
        self._model_ref = weakref.ref(model)
        self._stack: ExitStack | None = None
        self._specs: list[HookSpec] = []

    @property
    def program(self) -> HookProgram:
        """Return an immutable description of the operations installed this run."""

        return HookProgram(tuple(self._specs))

    def __enter__(self) -> "HookSession":
        if self._stack is not None:
            raise RuntimeError("Cannot enter a HookSession twice")
        self._model()
        self._specs.clear()
        self._stack = ExitStack()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stack = self._stack
        if stack is None:
            raise RuntimeError("Cannot exit a HookSession that is not active")
        self._stack = None
        return stack.__exit__(exc_type, exc_value, traceback)

    def capture(self, target: Target, *, prepend: bool = False) -> CapturedTarget:
        """Capture ``target`` while this session is active."""

        model, stack = self._active_state()
        module = target.validate(model)
        captured = CapturedTarget()
        direction = self._direction(target)

        if target.kind == "parameter":
            parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
            captured.value = target._select(parameter).detach().clone()
        else:

            def forward_hook(_module: nn.Module, _args: tuple[object, ...], value: Tensor | tuple[Tensor, ...]):
                captured.value = target._select(self._hook_tensor(value)).detach().clone()

            def gradient_hook(_module: nn.Module, values: tuple[Tensor, ...]):
                captured.value = target._select(self._hook_tensor(values)).detach().clone()

            handle = register_hook_to_module(
                module,
                forward_hook if target.kind == "activation" else gradient_hook,
                direction=direction,
                prepend=prepend,
            )
            stack.callback(handle.remove)

        self._specs.append(HookSpec(target, "capture", direction, prepend))
        return captured

    def replace(self, target: Target, value: Tensor | float | int, *, prepend: bool = False) -> None:
        """Replace ``target`` until the session exits."""

        model, stack = self._active_state()
        module = target.validate(model)
        direction = self._direction(target)

        if target.kind == "parameter":
            parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
            original = target._select(parameter).detach().clone()
            stack.callback(self._restore_parameter, target, parameter, original)
            with torch.no_grad():
                target._assign(parameter, value)
        else:

            def forward_hook(_module: nn.Module, _args: tuple[object, ...], output: Tensor | tuple[Tensor, ...]):
                replacement = self._hook_tensor(output).clone()
                target._assign(replacement, value)
                return (replacement,) if isinstance(output, tuple) else replacement

            def gradient_hook(_module: nn.Module, values: tuple[Tensor, ...]):
                replacement = self._hook_tensor(values).clone()
                target._assign(replacement, value)
                return (replacement,) + values[1:]

            handle = register_hook_to_module(
                module,
                forward_hook if target.kind == "activation" else gradient_hook,
                direction=direction,
                prepend=prepend,
            )
            stack.callback(handle.remove)

        self._specs.append(HookSpec(target, "replace", direction, prepend))

    def _active_state(self) -> tuple[nn.Module, ExitStack]:
        if self._stack is None:
            raise RuntimeError("HookSession operations require an active context")
        return self._model(), self._stack

    def _model(self) -> nn.Module:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("The model bound to this HookSession no longer exists")
        return model

    @staticmethod
    def _direction(target: Target) -> HookDirection | None:
        if target.kind == "activation":
            return "fwd"
        if target.kind == "gradient":
            return "bwd_pre"
        return None

    @staticmethod
    def _restore_parameter(target: Target, parameter: Tensor, original: Tensor) -> None:
        with torch.no_grad():
            target._assign(parameter, original)

    @staticmethod
    def _hook_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
        if isinstance(value, Tensor):
            return value
        if len(value) != 1 or not isinstance(value[0], Tensor):
            raise ValueError("HookSession targets require a hook value containing exactly one tensor")
        return value[0]


__all__ = ["CapturedTarget", "HookOperation", "HookProgram", "HookSession", "HookSpec"]
