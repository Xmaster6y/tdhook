from contextlib import contextmanager
from contextlib import ExitStack
from functools import wraps
import inspect
import sys
from typing import Callable, List, Optional, Generator, Dict, Mapping
from torch import nn
from tensordict.nn import TensorDictModuleBase, TensorDictModule
from tensordict import TensorDict

from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle, merge_paths
from tdhook._types import UnraveledKey, is_nested_key
from tdhook.execution import ExecutionSpec
from tdhook.descriptions import ConfiguredStepDescription, configured_step_description
from tdhook.runtime import BoundHookProgram, HookProgram


class HookingContext:
    """
    Base class for hooking contexts.
    """

    def __init__(
        self,
        factory: "HookingContextFactory",
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey] | Dict[UnraveledKey, str]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
        pre_factories: Optional[List["HookingContextFactory"]] = None,
    ):
        self._prepare = factory._prepare_module
        self._restore = factory._restore_module
        self._spawn = factory._spawn_hooked_module
        self._hook = factory._hook_module
        self._in_context = False
        self._handle = None
        self._program = None
        self._hooked_module = None
        self._pre_factories = pre_factories or []
        self._stack = None

        if isinstance(module, TensorDictModuleBase):
            self._module = module
            self._extra_relative_path = ""
        else:
            self._module = TensorDictModule(module, in_keys or ["input"], out_keys or ["output"])
            self._extra_relative_path = "module"

        self._in_keys = self._module.in_keys
        self._out_keys = self._module.out_keys
        self._for_inspection = False

    def _enter(self, *, for_inspection: bool = False):
        if self._in_context:
            raise RuntimeError("Cannot enter context twice")
        self._in_context = True
        self._for_inspection = for_inspection
        self._program = None

        working_module = self._module
        prepared = False
        try:
            with ExitStack() as stack:
                for factory in self._pre_factories:
                    child = factory.prepare(working_module, self._in_keys, self._out_keys)
                    working_module = stack.enter_context(child.inspect() if for_inspection else child)
                self._stack = stack.pop_all()
            prep_module = self._prepare(working_module, self._in_keys, self._out_keys, self._extra_relative_path)
            prepared = True
            self._hooked_module = self._spawn(prep_module, self, self._extra_relative_path)
            self._handle = self._hook(self._hooked_module)
            self._program = getattr(self._handle, "program", None)
            return self._hooked_module
        except BaseException:
            self._abort_enter(prepared)
            raise

    def _abort_enter(self, prepared: bool) -> None:
        """Undo a partially-entered context without allowing one cleanup failure to skip another."""
        try:
            if self._handle is not None:
                self._handle.remove()
        finally:
            try:
                if prepared:
                    self._restore(self._module, self._in_keys, self._out_keys, self._extra_relative_path)
            finally:
                try:
                    if self._stack is not None:
                        self._stack.__exit__(None, None, None)
                finally:
                    self._in_context = False
                    self._for_inspection = False
                    self._hooked_module = None
                    self._handle = None
                    self._stack = None

    @property
    def program(self) -> HookProgram | None:
        """Return the model-free hook program installed by this context."""

        return self._program

    def on_hook_failure(self, callback) -> None:
        """Register cleanup to run if a bound hook raises during execution."""

        if self._handle is None:
            raise RuntimeError("Hook failure cleanup is only available inside the prepared context")
        if not isinstance(self._handle, BoundHookProgram):
            raise TypeError("Hook failure cleanup requires a BoundHookProgram")
        self._handle.on_hook_failure(callback)

    @property
    def for_inspection(self) -> bool:
        """Whether this binding exists only to discover execution facts."""

        return self._for_inspection

    @property
    def executes_model_directly(self) -> bool:
        """Whether the bound wrapper executes the caller's TensorDict module unchanged."""

        if not self._in_context or self._hooked_module is None:
            raise RuntimeError("Direct-execution state is only available inside the prepared context")
        return self._hooked_module.td_module is self._module

    @property
    def model_in_keys(self) -> tuple[UnraveledKey, ...]:
        """Return the caller-owned model signature used by this binding."""

        return tuple(self._in_keys)

    @property
    def model_out_keys(self) -> tuple[UnraveledKey, ...]:
        """Return the caller-owned model outputs used by this binding."""

        return tuple(self._out_keys)

    def __enter__(self):
        return self._enter()

    @contextmanager
    def inspect(self) -> Generator[TensorDictModuleBase, None, None]:
        """Bind temporarily for planning without consuming execution state."""

        prepared = self._enter(for_inspection=True)
        try:
            yield prepared
        finally:
            self.__exit__(*sys.exc_info())

    def __exit__(self, exc_type, exc_value, traceback):
        cleanup_error = None
        try:
            if self._handle is not None:
                try:
                    self._handle.remove()
                except BaseException as error:
                    cleanup_error = error
            try:
                self._restore(self._module, self._in_keys, self._out_keys, self._extra_relative_path)
            except BaseException as error:
                cleanup_error = cleanup_error or error
        finally:
            self._in_context = False
            self._for_inspection = False
            self._hooked_module = None
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
            raise RuntimeError("Cannot disable hooks outside of context")
        self._handle.remove()
        try:
            yield
        finally:
            self._handle = self._hook(self._hooked_module)
            self._program = getattr(self._handle, "program", None)

    @contextmanager
    def disable(self) -> Generator[nn.Module, None, None]:
        if not self._in_context:
            raise RuntimeError("Cannot disable context outside of context")
        with self.disable_hooks():
            try:
                yield self._restore(
                    self._hooked_module.module, self._in_keys, self._out_keys, self._extra_relative_path
                )
            finally:
                self._hooked_module.module = self._prepare(
                    self._module, self._in_keys, self._out_keys, self._extra_relative_path
                )


class HookingContextWithCache(HookingContext):
    """
    Hooking context with cache.
    """

    def __init__(self, *args, cache: Optional[TensorDict] = None, clear_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = TensorDict() if cache is None else cache
        self._clear_cache = clear_cache

    @property
    def cache(self) -> TensorDict:
        return self._cache

    def clear(self):
        self._cache.clear()

    def _enter(self, *, for_inspection: bool = False):
        if self._clear_cache and not for_inspection:
            self.clear()
        return super()._enter(for_inspection=for_inspection)

    def __enter__(self):
        return self._enter()


class HookingContextFactory:
    """
    Factory for creating hooking contexts.
    """

    _hooked_module_class = HookedModule
    _hooking_context_class = HookingContext

    def __init_subclass__(cls, **kwargs) -> None:
        """Capture each concrete method's declared constructor configuration."""

        super().__init_subclass__(**kwargs)
        initializer = cls.__dict__.get("__init__")
        if initializer is None:
            return
        signature = inspect.signature(initializer)

        @wraps(initializer)
        def configured_initializer(self, *args, **init_kwargs):
            bound = signature.bind(self, *args, **init_kwargs)
            bound.apply_defaults()
            initializer(self, *args, **init_kwargs)
            self._configured_step_parameters = {
                name: value
                for name, value in bound.arguments.items()
                if name != "self" and name not in {"args", "kwargs"}
            }

        cls.__init__ = configured_initializer

    def __init__(self):
        self._hooking_context_kwargs = {}
        self._hooked_module_kwargs = {}
        self._configured_step_parameters: dict[str, object] = {}

    def describe(
        self, *, callback_identifiers: Mapping[Callable[..., object], str] | None = None
    ) -> ConfiguredStepDescription:
        """Describe this configured method without serializing executable objects."""

        return configured_step_description(
            self,
            self._configured_step_parameters,
            self.execution_spec,
            callback_identifiers=callback_identifiers,
        )

    @property
    def execution_spec(self) -> ExecutionSpec:
        """Return model-execution requirements owned by this method."""

        return ExecutionSpec()

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey] | Dict[UnraveledKey, str]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
    ) -> "HookingContext":
        """Return the sole managed binding interface for ``module``."""
        if isinstance(module, TensorDictModuleBase):
            if in_keys is not None:
                for key in in_keys:
                    if not is_nested_key(key):
                        raise ValueError(f"in_keys must be unraveled, got {type(key)}")
                    if key not in module.in_keys:
                        raise ValueError(f"Key {key} not in module.in_keys")
            if out_keys is not None:
                for key in out_keys:
                    if not is_nested_key(key):
                        raise ValueError(f"out_keys must be unraveled, got {type(key)}")
                    if key not in module.out_keys:
                        raise ValueError(f"Key {key} not in module.out_keys")

        context = self._hooking_context_class(self, module, in_keys, out_keys, **self._hooking_context_kwargs)

        return context

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        return module

    def _restore_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        return module

    def _spawn_hooked_module(
        self, prep_module: TensorDictModuleBase, hooking_context: "HookingContext", extra_relative_path: str
    ) -> HookedModule:
        base_relative_path = self._hooked_module_kwargs.get("relative_path", "td_module")
        relative_path = merge_paths(base_relative_path, extra_relative_path)
        kwargs = {
            **self._hooked_module_kwargs,
            "relative_path": relative_path,
        }
        return self._hooked_module_class(prep_module, hooking_context=hooking_context, **kwargs)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return MultiHookHandle()
