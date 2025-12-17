from contextlib import contextmanager
from contextlib import ExitStack
from typing import List, Optional, Generator, Dict
from torch import nn
from tensordict.nn import TensorDictModuleBase, TensorDictModule
from tensordict import TensorDict

from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle, merge_paths
from tdhook._types import UnraveledKey


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

    def __enter__(self):
        if self._in_context:
            raise RuntimeError("Cannot enter context twice")
        self._in_context = True

        working_module = self._module
        with ExitStack() as stack:
            for factory in self._pre_factories:
                working_module = stack.enter_context(factory.prepare(working_module, self._in_keys, self._out_keys))
            self._stack = stack.pop_all()

        prep_module = self._prepare(working_module, self._in_keys, self._out_keys, self._extra_relative_path)
        self._hooked_module = self._spawn(prep_module, self, self._extra_relative_path)
        self._handle = self._hook(self._hooked_module)
        return self._hooked_module

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
        self._restore(self._module, self._in_keys, self._out_keys, self._extra_relative_path)
        self._in_context = False
        self._hooked_module = None
        self._handle = None
        self._stack.__exit__(exc_type, exc_value, traceback)

    @contextmanager
    def disable_hooks(self) -> Generator[None, None, None]:
        if not self._in_context:
            raise RuntimeError("Cannot disable hooks outside of context")
        self._handle.remove()
        try:
            yield
        finally:
            self._handle = self._hook(self._hooked_module)

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

    def __enter__(self):
        if self._clear_cache:
            self.clear()
        return super().__enter__()


class HookingContextFactory:
    """
    Factory for creating hooking contexts.
    """

    _hooked_module_class = HookedModule
    _hooking_context_class = HookingContext

    def __init__(self):
        self._hooking_context_kwargs = {}
        self._hooked_module_kwargs = {}

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey] | Dict[UnraveledKey, str]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
    ) -> "HookingContext":
        """
        Prepare the module for execution.
        """
        if isinstance(module, TensorDictModuleBase):
            if in_keys is not None:
                for key in in_keys:
                    if not isinstance(key, UnraveledKey):
                        raise ValueError(f"in_keys must be unraveled, got {type(key)}")
                    if key not in module.in_keys:
                        raise ValueError(f"Key {key} not in module.in_keys")
            if out_keys is not None:
                for key in out_keys:
                    if not isinstance(key, UnraveledKey):
                        raise ValueError(f"out_keys must be unraveled, got {type(key)}")
                    if key not in module.out_keys:
                        raise ValueError(f"Key {key} not in module.out_keys")

        return self._hooking_context_class(self, module, in_keys, out_keys, **self._hooking_context_kwargs)

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


class CompositeHookingContextFactory(HookingContextFactory):
    """
    Composite hooking context factory.
    """

    def __init__(self, *contexts: HookingContextFactory):
        super().__init__()
        self._contexts = contexts
        attributes = ("_spawn_hooked_module", "_hooking_context_class", "_hooked_module_class")
        composite_overriden = {
            attr: getattr(type(self), attr) != getattr(HookingContextFactory, attr) for attr in attributes
        }
        for context in contexts:
            for attr in attributes:
                if (
                    getattr(type(context), attr) != getattr(HookingContextFactory, attr)
                    and not composite_overriden[attr]
                ):
                    raise ValueError(
                        f"Cannot compose factories that override {attr}, consider subclassing this factory to override {attr}"
                    )

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        for context in self._contexts:
            module = context._prepare_module(module, in_keys, out_keys, extra_relative_path)
        return module

    def _restore_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        for context in reversed(self._contexts):
            module = context._restore_module(module, in_keys, out_keys, extra_relative_path)
        return module

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = [context._hook_module(module) for context in self._contexts]
        return MultiHookHandle(handles)
