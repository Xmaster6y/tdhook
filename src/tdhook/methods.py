from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import List, Optional, Generator, Dict
from torch import nn
from tensordict.nn import TensorDictModuleBase, TensorDictModule
from tensordict import TensorDict
from tensordict.utils import NestedKey

from tdhook.modules import BoundModule
from tdhook.hooks import MultiHookHandle
from tdhook._types import is_nested_key
from tdhook.execution import ExecutionSpec
from tdhook.runtime import BoundHookProgram, HookProgram, TargetOccurrenceEvidence


class BoundMethod:
    """One model-specific method binding with deterministic cleanup."""

    def __init__(
        self,
        method: "Method",
        module: nn.Module,
        in_keys: Optional[List[NestedKey] | Dict[NestedKey, str]] = None,
        out_keys: Optional[List[NestedKey]] = None,
        pre_methods: Optional[List["Method"]] = None,
    ):
        self._prepare = method._bind_module
        self._restore = method._restore_module
        self._spawn = method._spawn_bound_module
        self._hook = method._install_hooks
        self._in_context = False
        self._handle = None
        self._program = None
        self._occurrence_evidence: tuple[TargetOccurrenceEvidence, ...] = ()
        self._bound_module = None
        self._pre_methods = pre_methods or []
        self._stack = None

        if isinstance(module, TensorDictModuleBase):
            self._module = module
            self._extra_relative_path = ""
        else:
            self._module = TensorDictModule(module, in_keys or ["input"], out_keys or ["output"])
            self._extra_relative_path = "module"

        self._in_keys = self._module.in_keys
        self._out_keys = self._module.out_keys

    def __enter__(self):
        if self._in_context:
            raise RuntimeError("Cannot enter a method binding twice")
        self._in_context = True
        self._program = None
        self._occurrence_evidence = ()

        working_module = self._module
        module_bound = False
        try:
            with ExitStack() as stack:
                for method in self._pre_methods:
                    child = method.bind(working_module, self._in_keys, self._out_keys)
                    working_module = stack.enter_context(child)
                self._stack = stack.pop_all()
            prep_module = self._prepare(working_module, self._in_keys, self._out_keys, self._extra_relative_path)
            module_bound = True
            self._bound_module = self._spawn(prep_module, self, self._extra_relative_path)
            self._handle = self._hook(self._bound_module)
            self._program = getattr(self._handle, "program", None)
            return self._bound_module
        except BaseException:
            self._abort_enter(module_bound)
            raise

    def _abort_enter(self, module_bound: bool) -> None:
        """Undo a partial binding without allowing one cleanup failure to skip another."""
        try:
            if self._handle is not None:
                self._handle.remove()
        finally:
            try:
                if module_bound:
                    self._restore(self._module, self._in_keys, self._out_keys, self._extra_relative_path)
            finally:
                try:
                    if self._stack is not None:
                        self._stack.__exit__(None, None, None)
                finally:
                    self._in_context = False
                    self._bound_module = None
                    self._handle = None
                    self._stack = None

    @property
    def program(self) -> HookProgram | None:
        """Return the model-free hook program installed by this binding."""

        return self._program

    @property
    def occurrence_evidence(self) -> tuple[TargetOccurrenceEvidence, ...]:
        """Return validated target-occurrence evidence from this binding."""

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
            raise RuntimeError("Hook failure cleanup is only available inside an active binding")
        if not isinstance(self._handle, BoundHookProgram):
            raise TypeError("Hook failure cleanup requires a BoundHookProgram")
        self._handle.on_hook_failure(callback)

    @property
    def executes_model_directly(self) -> bool:
        """Whether the bound wrapper executes the caller's TensorDict module unchanged."""

        if not self._in_context or self._bound_module is None:
            raise RuntimeError("Direct-execution state is only available inside an active binding")
        return self._bound_module.td_module is self._module

    @property
    def model_in_keys(self) -> tuple[NestedKey, ...]:
        """Return the caller-owned model signature used by this binding."""

        return tuple(self._in_keys)

    @property
    def model_out_keys(self) -> tuple[NestedKey, ...]:
        """Return the caller-owned model outputs used by this binding."""

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
                self._restore(self._module, self._in_keys, self._out_keys, self._extra_relative_path)
            except BaseException as error:
                cleanup_error = cleanup_error or error
        finally:
            self._in_context = False
            self._bound_module = None
            self._handle = None
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
            raise RuntimeError("Cannot disable hooks outside an active binding")
        self._handle.remove()
        self._retain_occurrence_evidence(self._handle)
        try:
            yield
        finally:
            self._handle = self._hook(self._bound_module)
            self._program = getattr(self._handle, "program", None)

    @contextmanager
    def disable(self) -> Generator[nn.Module, None, None]:
        if not self._in_context:
            raise RuntimeError("Cannot disable a method outside an active binding")
        with self.disable_hooks():
            try:
                yield self._restore(
                    self._bound_module.module, self._in_keys, self._out_keys, self._extra_relative_path
                )
            finally:
                self._bound_module.module = self._prepare(
                    self._module, self._in_keys, self._out_keys, self._extra_relative_path
                )


class CachedBoundMethod(BoundMethod):
    """A method binding that owns or publishes an activation cache."""

    def __init__(self, *args, cache: Optional[TensorDict] = None, clear_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = TensorDict() if cache is None else cache
        self._clear_cache = clear_cache

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


class Method:
    """Base class for a configured interpretability method."""

    _bound_module_class = BoundModule
    _binding_class = BoundMethod

    def __init__(self):
        self._binding_kwargs = {}
        self._bound_module_kwargs = {}

    @property
    def execution_spec(self) -> ExecutionSpec:
        """Return model-execution requirements owned by this method."""

        return ExecutionSpec()

    def bind(
        self,
        module: nn.Module,
        in_keys: Optional[List[NestedKey] | Dict[NestedKey, str]] = None,
        out_keys: Optional[List[NestedKey]] = None,
    ) -> "BoundMethod":
        """Return the sole managed binding interface for ``module``."""
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

        return self._binding_class(self, module, in_keys, out_keys, **self._binding_kwargs)

    def _bind_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        return module

    def _restore_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        return module

    def _spawn_bound_module(
        self, prep_module: TensorDictModuleBase, binding: "BoundMethod", extra_relative_path: str
    ) -> BoundModule:
        kwargs = {
            **self._bound_module_kwargs,
            "hook_root": binding._module,
            "relative_path": extra_relative_path,
        }
        return self._bound_module_class(prep_module, binding=binding, **kwargs)

    def _install_hooks(self, module: BoundModule) -> MultiHookHandle:
        return MultiHookHandle()


__all__ = ["BoundMethod", "Method"]
