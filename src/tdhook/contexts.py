from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import List, Optional, Generator, Dict
from torch import nn
from tensordict.nn import TensorDictModuleBase, TensorDictModule
from tensordict import TensorDict
from tensordict.utils import NestedKey

from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle
from tdhook._types import is_nested_key
from tdhook.execution import ExecutionSpec
from tdhook.runtime import BoundHookProgram, HookProgram, TargetOccurrenceEvidence


class HookingContext:
    """One model-specific method context with deterministic cleanup."""

    def __init__(
        self,
        factory: "HookingContextFactory",
        module: nn.Module,
        in_keys: Optional[List[NestedKey] | Dict[NestedKey, str]] = None,
        out_keys: Optional[List[NestedKey]] = None,
        pre_factories: Optional[List["HookingContextFactory"]] = None,
        *,
        hook_root: Optional[TensorDictModuleBase] = None,
        relative_path: Optional[str] = None,
    ):
        self._prepare = factory._prepare_module
        self._restore = factory._restore_module
        self._spawn = factory._spawn_hooked_module
        self._hook = factory._hook_module
        self._in_context = False
        self._handle = None
        self._install_started = False
        self._program = None
        self._occurrence_evidence: tuple[TargetOccurrenceEvidence, ...] = ()
        self._hooked_module: HookedModule | None = None
        self._pre_contexts: list[HookingContext] = []
        self._stack = None

        if hook_root is not None:
            if not isinstance(module, TensorDictModuleBase) or relative_path is None:
                raise TypeError("Internal method contexts require a TensorDict module and relative path")
            self._module = module
            self._hook_root = hook_root
            self._extra_relative_path = relative_path
        elif isinstance(module, TensorDictModuleBase):
            self._module = module
            self._hook_root = module
            self._extra_relative_path = ""
        else:
            self._module = TensorDictModule(module, in_keys or ["input"], out_keys or ["output"])
            self._hook_root = self._module
            self._extra_relative_path = "module"

        self._in_keys = self._module.in_keys
        self._out_keys = self._module.out_keys
        working_module = self._module
        for pre_factory in pre_factories or ():
            child = pre_factory._prepare(
                working_module,
                self._in_keys,
                self._out_keys,
                hook_root=self._hook_root,
                relative_path=self._extra_relative_path,
            )
            self._pre_contexts.append(child)
            working_module = child.module
        self._working_module = working_module
        method_module = self._prepare(
            self._working_module,
            self._in_keys,
            self._out_keys,
            self._extra_relative_path,
        )
        self._declared_module = self._spawn(method_module, self, self._extra_relative_path)

    def __enter__(self):
        if self._in_context:
            raise RuntimeError("Cannot enter a method context twice")
        self._in_context = True
        self._program = None
        self._occurrence_evidence = ()

        try:
            with ExitStack() as stack:
                for child in self._pre_contexts:
                    stack.enter_context(child)
                self._stack = stack.pop_all()
            self._hooked_module = self._declared_module
            self._install_started = True
            self._handle = self._hook(self._hooked_module)
            self._program = getattr(self._handle, "program", None)
            return self._hooked_module
        except BaseException:
            self._abort_enter()
            raise

    def _abort_enter(self) -> None:
        """Undo a partial context without allowing one cleanup failure to skip another."""
        try:
            if self._handle is not None:
                self._handle.remove()
        finally:
            try:
                if self._install_started:
                    self._restore(self._working_module, self._in_keys, self._out_keys, self._extra_relative_path)
            finally:
                try:
                    if self._stack is not None:
                        self._stack.__exit__(None, None, None)
                finally:
                    self._in_context = False
                    self._hooked_module = None
                    self._handle = None
                    self._install_started = False
                    self._stack = None

    @property
    def program(self) -> HookProgram | None:
        """Return the model-free hook program installed by this context."""

        return self._program

    @property
    def module(self) -> HookedModule:
        """Return the hooked module contract without entering the context."""

        return self._declared_module

    @property
    def occurrence_evidence(self) -> tuple[TargetOccurrenceEvidence, ...]:
        """Return validated target-occurrence evidence from this context."""

        if isinstance(self._handle, BoundHookProgram):
            return self._occurrence_evidence + self._renumber_occurrence_evidence(self._handle.occurrence_evidence)
        return self._occurrence_evidence

    def _renumber_occurrence_evidence(
        self,
        evidence: tuple[TargetOccurrenceEvidence, ...],
    ) -> tuple[TargetOccurrenceEvidence, ...]:
        """Continue root-pass numbering when a context rebinds its hooks."""

        next_pass: dict[tuple[object, ...], int] = {}
        for item in self._occurrence_evidence:
            key = (item.hook_index, item.target_path, item.operation, item.direction)
            next_pass[key] = max(next_pass.get(key, 0), item.root_pass + 1)
        return tuple(
            replace(
                item,
                root_pass=item.root_pass
                + next_pass.get((item.hook_index, item.target_path, item.operation, item.direction), 0),
            )
            for item in evidence
        )

    def _retain_occurrence_evidence(self, handle: object) -> None:
        if isinstance(handle, BoundHookProgram):
            self._occurrence_evidence += self._renumber_occurrence_evidence(handle.occurrence_evidence)

    def on_hook_failure(self, callback) -> None:
        """Register cleanup to run if a bound hook raises during execution."""

        if self._handle is None:
            raise RuntimeError("Hook failure cleanup is only available inside an active context")
        if not isinstance(self._handle, BoundHookProgram):
            raise TypeError("Hook failure cleanup requires a BoundHookProgram")
        self._handle.on_hook_failure(callback)

    @property
    def executes_model_directly(self) -> bool:
        """Whether the hooked wrapper executes the caller's TensorDict module unchanged."""

        if not self._in_context or self._hooked_module is None:
            raise RuntimeError("Direct-execution state is only available inside an active context")
        return self._hooked_module.td_module is self._module

    @property
    def model_in_keys(self) -> tuple[NestedKey, ...]:
        """Return the caller-owned model signature used by this context."""

        return tuple(self._in_keys)

    @property
    def model_out_keys(self) -> tuple[NestedKey, ...]:
        """Return the caller-owned model outputs used by this context."""

        return tuple(self._out_keys)

    def __exit__(self, exc_type, exc_value, traceback):
        cleanup_error = None
        try:
            if self._handle is not None:
                try:
                    self._handle.remove()
                except BaseException as error:
                    cleanup_error = error
                finally:
                    self._retain_occurrence_evidence(self._handle)
            try:
                self._restore(self._working_module, self._in_keys, self._out_keys, self._extra_relative_path)
            except BaseException as error:
                cleanup_error = cleanup_error or error
        finally:
            self._in_context = False
            self._hooked_module = None
            self._handle = None
            self._install_started = False
            if self._stack is not None:
                try:
                    self._stack.__exit__(exc_type, exc_value, traceback)
                except BaseException as error:
                    cleanup_error = cleanup_error or error
                self._stack = None
        if cleanup_error is not None:
            raise cleanup_error

    @contextmanager
    def disable_hooks(self) -> Generator[None, None, None]:
        if not self._in_context:
            raise RuntimeError("Cannot disable hooks outside an active context")
        self._handle.remove()
        self._retain_occurrence_evidence(self._handle)
        try:
            yield
        finally:
            self._handle = self._hook(self._hooked_module)
            self._program = getattr(self._handle, "program", None)

    @contextmanager
    def disable(self) -> Generator[nn.Module, None, None]:
        if not self._in_context:
            raise RuntimeError("Cannot disable a method outside an active context")
        with self.disable_hooks():
            try:
                yield self._restore(self._working_module, self._in_keys, self._out_keys, self._extra_relative_path)
            finally:
                self._hooked_module.module = self._prepare(
                    self._working_module, self._in_keys, self._out_keys, self._extra_relative_path
                )


class HookingContextWithCache(HookingContext):
    """A method context that owns or publishes an activation cache."""

    def __init__(self, *args, cache: Optional[TensorDict] = None, clear_cache: bool = True, **kwargs):
        self._cache = TensorDict() if cache is None else cache
        self._clear_cache = clear_cache
        super().__init__(*args, **kwargs)

    @property
    def cache(self) -> TensorDict:
        return self._cache

    def clear(self):
        self._cache.clear()

    def __enter__(self):
        if self._clear_cache:
            if self._cache.is_locked:
                raise ValueError("locked or memory-mapped caches require clear_cache=False")
            self.clear()
        return super().__enter__()


class HookingContextFactory:
    """Base class for a configured interpretability method."""

    _hooked_module_class = HookedModule
    _hooking_context_class = HookingContext

    def __init__(self):
        self._hooking_context_kwargs = {}
        self._hooked_module_kwargs = {}

    @property
    def execution_spec(self) -> ExecutionSpec:
        """Return model-execution requirements owned by this method."""

        return ExecutionSpec()

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[NestedKey] | Dict[NestedKey, str]] = None,
        out_keys: Optional[List[NestedKey]] = None,
    ) -> "HookingContext":
        """Build the managed context for ``module`` without installing hooks."""
        if isinstance(module, TensorDictModuleBase):
            if in_keys is not None:
                for key in in_keys:
                    if not is_nested_key(key):
                        raise ValueError(f"in_keys must be TensorDict nested keys, got {type(key)}")
                    if key not in module.in_keys:
                        raise ValueError(f"Key {key} not in module.in_keys")
            if out_keys is not None:
                for key in out_keys:
                    if not is_nested_key(key):
                        raise ValueError(f"out_keys must be TensorDict nested keys, got {type(key)}")
                    if key not in module.out_keys:
                        raise ValueError(f"Key {key} not in module.out_keys")

        return self._prepare(module, in_keys, out_keys)

    def _prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[NestedKey] | Dict[NestedKey, str]],
        out_keys: Optional[List[NestedKey]],
        *,
        hook_root: Optional[TensorDictModuleBase] = None,
        relative_path: Optional[str] = None,
    ) -> HookingContext:
        return self._hooking_context_class(
            self,
            module,
            in_keys,
            out_keys,
            hook_root=hook_root,
            relative_path=relative_path,
            **self._hooking_context_kwargs,
        )

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        """Construct this method's execution module without mutating caller state."""

        return module

    def _restore_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        return module

    def _spawn_hooked_module(
        self, prep_module: TensorDictModuleBase, context: "HookingContext", extra_relative_path: str
    ) -> HookedModule:
        kwargs = {
            **self._hooked_module_kwargs,
            "hook_root": context._hook_root,
            "relative_path": extra_relative_path,
        }
        return self._hooked_module_class(prep_module, hooking_context=context, **kwargs)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return MultiHookHandle()


__all__ = ["HookingContext", "HookingContextFactory"]
