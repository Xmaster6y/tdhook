"""Interactive capture and intervention through an explicit hook lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import copy
from dataclasses import dataclass
from typing import Literal
import weakref

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn

from tdhook.hooks import HookDirection
from tdhook.runtime import HookProgram, HookProgramBuilder, HookSpec
from tdhook.targets import OutputPathComponent, Target


HookOperation = Literal["capture", "replace"]


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
        self._builder: HookProgramBuilder | None = None
        self._program = HookProgram()

    @property
    def program(self) -> HookProgram:
        """Return an immutable description of the operations installed this run."""

        return self._builder.program if self._builder is not None else self._program

    def __enter__(self) -> "HookSession":
        if self._builder is not None:
            raise RuntimeError("Cannot enter a HookSession twice")
        self._model()
        self._program = HookProgram()
        self._builder = HookProgramBuilder()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        builder = self._builder
        if builder is None:
            raise RuntimeError("Cannot exit a HookSession that is not active")
        self._builder = None
        bound = builder.build()
        self._program = bound.program
        bound.remove()

    def capture(self, target: Target, *, prepend: bool = False) -> CapturedTarget:
        """Capture ``target`` while this session is active."""

        model, builder = self._active_state()
        module = target.validate(model)
        captured = CapturedTarget()
        direction = self._direction(target)
        spec = HookSpec(target.module_path, "capture", direction, prepend, target)

        if target.kind == "parameter":
            parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
            captured.value = target._select(parameter).detach().clone()
            builder.record(spec)
        else:

            def forward_hook(_module: nn.Module, _args: tuple[object, ...], value: object):
                captured.value = target._select(self._hook_tensor(value, target.output_path)).detach().clone()

            def gradient_hook(_module: nn.Module, values: tuple[Tensor | None, ...]):
                captured.value = target._select(self._hook_tensor(values, target.output_path)).detach().clone()

            builder.register(
                module,
                forward_hook if target.kind == "activation" else gradient_hook,
                spec,
            )

        return captured

    def replace(self, target: Target, value: Tensor | float | int, *, prepend: bool = False) -> None:
        """Replace ``target`` until the session exits."""

        model, builder = self._active_state()
        module = target.validate(model)
        direction = self._direction(target)
        spec = HookSpec(target.module_path, "replace", direction, prepend, target)

        if target.kind == "parameter":
            parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
            original = target._select(parameter).detach().clone()
            try:
                with torch.no_grad():
                    target._assign(parameter, value)
            except BaseException:
                self._restore_parameter(target, parameter, original)
                raise
            builder.record(spec, lambda: self._restore_live_parameter(target, original))
        else:

            def forward_hook(_module: nn.Module, _args: tuple[object, ...], output: object):
                path = self._resolved_output_path(output, target.output_path)
                replacement = self._hook_tensor(output, path).clone()
                target._assign(replacement, value)
                return self._replace_hook_tensor(output, path, replacement)

            def gradient_hook(_module: nn.Module, values: tuple[Tensor | None, ...]):
                path = self._resolved_output_path(values, target.output_path)
                replacement = self._hook_tensor(values, path).clone()
                target._assign(replacement, value)
                return self._replace_hook_tensor(values, path, replacement)

            builder.register(
                module,
                forward_hook if target.kind == "activation" else gradient_hook,
                spec,
            )

    def _active_state(self) -> tuple[nn.Module, HookProgramBuilder]:
        if self._builder is None:
            raise RuntimeError("HookSession operations require an active context")
        return self._model(), self._builder

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

    def _restore_live_parameter(self, target: Target, original: Tensor) -> None:
        module = target.validate(self._model())
        parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
        self._restore_parameter(target, parameter, original)

    @staticmethod
    def _resolved_output_path(value: object, path: tuple[OutputPathComponent, ...]) -> tuple[OutputPathComponent, ...]:
        if path or isinstance(value, Tensor):
            return path
        if isinstance(value, (tuple, list)) and len(value) == 1:
            return (0,)
        raise ValueError("Structured hook values require Target.output_path to select a tensor leaf")

    @classmethod
    def _hook_tensor(cls, value: object, path: tuple[OutputPathComponent, ...] = ()) -> Tensor:
        selected = value
        for component in cls._resolved_output_path(value, path):
            if isinstance(component, int) and isinstance(selected, (tuple, list)):
                try:
                    selected = selected[component]
                except IndexError as exc:
                    raise ValueError(f"output_path slot {component} is out of range") from exc
            elif isinstance(component, str) and isinstance(selected, Mapping):
                try:
                    selected = selected[component]
                except KeyError as exc:
                    raise ValueError(f"output_path key {component!r} is missing") from exc
            else:
                raise ValueError(f"output_path component {component!r} does not match the hook value structure")
        if not isinstance(selected, Tensor):
            raise ValueError("Target.output_path must select a tensor leaf")
        return selected

    @classmethod
    def _replace_hook_tensor(
        cls,
        value: object,
        path: tuple[OutputPathComponent, ...],
        replacement: Tensor,
    ) -> object:
        resolved = cls._resolved_output_path(value, path)
        if not resolved:
            return replacement
        component, *remainder = resolved
        if isinstance(component, int) and isinstance(value, tuple):
            items = list(value)
            items[component] = cls._replace_hook_tensor(items[component], tuple(remainder), replacement)
            return tuple(items)
        if isinstance(component, int) and isinstance(value, list):
            items = list(value)
            items[component] = cls._replace_hook_tensor(items[component], tuple(remainder), replacement)
            return items
        if isinstance(component, str) and isinstance(value, TensorDictBase):
            items = value.clone(recurse=False)
            items.set(component, cls._replace_hook_tensor(value.get(component), tuple(remainder), replacement))
            return items
        if isinstance(component, str) and isinstance(value, Mapping):
            try:
                current = value[component]
            except KeyError as exc:
                raise ValueError(f"output_path key {component!r} is missing") from exc
            items = copy.copy(value)
            if not isinstance(items, MutableMapping):
                raise ValueError(
                    f"mapping type {type(value).__name__} does not support structure-preserving replacement"
                )
            items[component] = cls._replace_hook_tensor(current, tuple(remainder), replacement)
            return items
        raise ValueError(f"output_path component {component!r} does not match the hook value structure")


__all__ = ["CapturedTarget", "HookOperation", "HookProgram", "HookSession", "HookSpec"]
