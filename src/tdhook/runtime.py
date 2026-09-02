"""Executable hook programs shared by interactive and prepared interfaces."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generator

from tensordict import TensorDict, TensorDictBase
from torch import nn

from tdhook.hooks import HookDirection, register_hook_to_module
from tdhook.paths import resolve_submodule_path
from tdhook.targets import Target


HOOK_DIRECTIONS = frozenset({"fwd", "bwd", "fwd_pre", "bwd_pre", "fwd_kwargs", "fwd_pre_kwargs"})


@dataclass(frozen=True)
class CaptureSource:
    """Reference from an operation to an earlier capture in the program."""

    hook_index: int
    detach: bool

    def __post_init__(self) -> None:
        if type(self.hook_index) is not int:
            raise TypeError("hook_index must be an integer")
        if self.hook_index < 0:
            raise ValueError("hook_index must be non-negative")
        if not isinstance(self.detach, bool):
            raise TypeError("detach must be a bool")


@dataclass(frozen=True)
class HookSpec:
    """Model-free description of one installed hook operation."""

    module_path: str
    operation: str
    direction: HookDirection | None
    prepend: bool = False
    target: Target | None = None
    source: CaptureSource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.module_path, str):
            raise TypeError("module_path must be a string")
        if not isinstance(self.operation, str):
            raise TypeError("operation must be a string")
        if not self.operation:
            raise ValueError("operation must be non-empty")
        if self.direction is not None and self.direction not in HOOK_DIRECTIONS:
            raise ValueError(f"invalid hook direction: {self.direction!r}")
        if not isinstance(self.prepend, bool):
            raise TypeError("prepend must be a bool")
        if self.target is not None and not isinstance(self.target, Target):
            raise TypeError("target must be a Target")
        if self.source is not None and not isinstance(self.source, CaptureSource):
            raise TypeError("source must be a CaptureSource")


@dataclass(frozen=True)
class HookProgram:
    """Ordered, model-free description of installed hook operations."""

    hooks: tuple[HookSpec, ...] = ()
    stopped_at: str | None = None

    def __post_init__(self) -> None:
        for index, spec in enumerate(self.hooks):
            if spec.source is None:
                continue
            if spec.source.hook_index >= index:
                raise ValueError("capture dependencies must refer to an earlier hook")
            if self.hooks[spec.source.hook_index].operation != "capture":
                raise ValueError("capture dependencies must refer to a capture hook")


class BoundHookProgram:
    """An installed hook program with deterministic reverse-order cleanup."""

    def __init__(
        self,
        program: HookProgram,
        cleanups: tuple[Callable[[], Any], ...],
        hook_failure_handlers: tuple[list[Callable[[], Any] | None], ...] = (),
    ):
        self.program = program
        self._cleanups = list(cleanups)
        self._hook_failure_handlers = hook_failure_handlers

    def on_hook_failure(self, callback: Callable[[], Any]) -> None:
        """Run ``callback`` if one of this program's hooks raises."""

        for handler in self._hook_failure_handlers:
            handler[0] = callback

    def remove(self) -> None:
        """Run every remaining cleanup in reverse registration order."""

        error = None
        while self._cleanups:
            cleanup = self._cleanups.pop()
            try:
                cleanup()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error


class HookProgramBuilder:
    """Install hooks while constructing their inspectable program."""

    def __init__(self):
        self._specs: list[HookSpec] = []
        self._cleanups: list[Callable[[], Any]] = []
        self._hook_failure_handlers: list[list[Callable[[], Any] | None]] = []
        self._stopped_at: str | None = None
        self._built = False

    @property
    def program(self) -> HookProgram:
        """Return the operations successfully registered so far."""

        return HookProgram(tuple(self._specs), self._stopped_at)

    def mark_stopped(self, module_path: str) -> None:
        """Record that execution reached a registered early-stop location."""

        self._ensure_open()
        if not isinstance(module_path, str):
            raise TypeError("module_path must be a string")
        self._stopped_at = module_path

    def register(self, module: nn.Module, hook: Callable, spec: HookSpec) -> None:
        """Install one hook and record ``spec`` only after registration succeeds."""

        self._ensure_open()
        if spec.direction is None:
            raise ValueError("registered hooks require a direction")
        failure_handler: list[Callable[[], Any] | None] = [None]

        @wraps(hook)
        def guarded_hook(*args: Any, **kwargs: Any) -> Any:
            try:
                return hook(*args, **kwargs)
            except BaseException:
                if failure_handler[0] is not None:
                    failure_handler[0]()
                raise

        handle = register_hook_to_module(module, guarded_hook, spec.direction, spec.prepend)
        self._cleanups.append(handle.remove)
        self._hook_failure_handlers.append(failure_handler)
        self._specs.append(spec)

    def register_path(
        self,
        root: nn.Module,
        hook: Callable,
        spec: HookSpec,
        *,
        relative_path: str = "",
    ) -> None:
        """Resolve ``spec.module_path`` below ``root`` and install its hook."""

        module = self.resolve_path(root, spec.module_path, relative_path=relative_path)
        if isinstance(module, nn.ModuleList):
            warnings.warn(
                f"You are hooking a ModuleList ({spec.module_path}), which will never be executed.",
                stacklevel=2,
            )
        self.register(module, hook, spec)

    def resolve_path(self, root: nn.Module, module_path: str, *, relative_path: str = "") -> nn.Module:
        """Resolve one executable module path without installing a hook."""

        self._ensure_open()
        relative_root = resolve_submodule_path(root, relative_path)
        module = resolve_submodule_path(relative_root, module_path)
        if not isinstance(module, nn.Module):
            raise TypeError(f"runtime path must resolve to a torch.nn.Module, got {type(module).__name__}")
        return module

    def record(self, spec: HookSpec, cleanup: Callable[[], Any] | None = None) -> None:
        """Record a non-hook operation, optionally with lifecycle cleanup."""

        self._ensure_open()
        if cleanup is not None and not callable(cleanup):
            raise TypeError("cleanup must be callable")
        if cleanup is not None:
            self._cleanups.append(cleanup)
        self._specs.append(spec)

    def build(self) -> BoundHookProgram:
        """Seal and return the installed program."""

        self._ensure_open()
        bound = BoundHookProgram(self.program, tuple(self._cleanups), tuple(self._hook_failure_handlers))
        self._cleanups.clear()
        self._built = True
        return bound

    def remove(self) -> None:
        """Abort an unsealed program and run all registered cleanup."""

        self._ensure_open()
        try:
            BoundHookProgram(self.program, tuple(self._cleanups), tuple(self._hook_failure_handlers)).remove()
        finally:
            self._cleanups.clear()
            self._hook_failure_handlers.clear()
            self._specs.clear()
            self._stopped_at = None

    def __enter__(self) -> "HookProgramBuilder":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._built:
            self.remove()

    def _ensure_open(self) -> None:
        if self._built:
            raise RuntimeError("HookProgramBuilder is already built")


@contextmanager
def temporary_module_state(
    module: nn.Module,
    state: TensorDictBase,
    spec: HookSpec,
) -> Generator[HookProgram, None, None]:
    """Apply TensorDict module state and restore the caller-owned module on exit."""

    if spec.direction is not None:
        raise ValueError("temporary module state requires a directionless HookSpec")
    original = TensorDict.from_module(module).clone()
    with HookProgramBuilder() as builder:
        builder.record(spec, lambda: original.to_module(module, inplace=True))
        state.to_module(module, inplace=True)
        bound = builder.build()
    try:
        yield bound.program
    finally:
        bound.remove()


__all__ = [
    "BoundHookProgram",
    "CaptureSource",
    "HookProgram",
    "HookProgramBuilder",
    "HookSpec",
    "temporary_module_state",
]
