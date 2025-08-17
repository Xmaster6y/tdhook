"""
Context
"""

from contextlib import contextmanager
from typing import List, Optional, Generator
from torch import nn
from tensordict.nn import TensorDictModuleBase, TensorDictModule

from tdhook.module import HookedModule
from tdhook.hooks import MultiHookHandle
from tdhook._types import UnraveledKey


class HookingContext:
    def __init__(
        self,
        factory: "HookingContextFactory",
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
    ):
        self._prepare = factory._prepare_module
        self._restore = factory._restore_module
        self._spawn = factory._spawn_hooked_module
        self._hook = factory._hook_module
        self._in_context = False
        self._handle = None
        self._hooked_module = None

        if isinstance(module, TensorDictModuleBase):
            self._module = module
            self._in_keys = in_keys or module.in_keys
            self._out_keys = out_keys or module.out_keys
        else:
            self._in_keys = in_keys or ["input"]
            self._out_keys = out_keys or ["output"]
            self._module = TensorDictModule(module, self._in_keys, self._out_keys)

    def __enter__(self):
        if self._in_context:
            raise RuntimeError("Cannot enter context twice")
        self._in_context = True
        prep_module = self._prepare(self._module, self._in_keys, self._out_keys)
        self._hooked_module = self._spawn(prep_module, self)
        self._handle = self._hook(self._hooked_module)
        return self._hooked_module

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
        self._restore(self._module, self._in_keys, self._out_keys)
        self._in_context = False
        del self._hooked_module  # TODO: check impact of this
        self._hooked_module = None
        self._handle = None

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
                yield self._restore(self._hooked_module.module, self._in_keys, self._out_keys)
            finally:
                self._hooked_module.module = self._prepare(self._module, self._in_keys, self._out_keys)


class HookingContextFactory:
    """
    Factory for creating hooking contexts.
    """

    _hooked_module_class = HookedModule
    _hooking_context_class = HookingContext

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey]] = None,
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

        return self._hooking_context_class(self, module, in_keys, out_keys)

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
    ) -> TensorDictModuleBase:
        return module

    def _restore_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        return module

    def _spawn_hooked_module(
        self, prep_module: TensorDictModuleBase, hooking_context: "HookingContext"
    ) -> HookedModule:
        return self._hooked_module_class(prep_module, hooking_context=hooking_context)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return MultiHookHandle()


class CompositeHookingContextFactory(HookingContextFactory):
    def __init__(self, *contexts: HookingContextFactory):
        self._contexts = contexts
        composite_overriden = type(self)._spawn_hooked_module != HookingContextFactory._spawn_hooked_module
        for context in contexts:
            context_overriden = type(context)._spawn_hooked_module != HookingContextFactory._spawn_hooked_module
            if context_overriden and not composite_overriden:
                raise ValueError(
                    "Cannot compose factories that override _spawn_hooked_module, consider subclassing this factory to override _spawn_hooked_module"
                )

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
    ) -> TensorDictModuleBase:
        for context in self._contexts:
            module = context._prepare_module(module, in_keys, out_keys)
        return module

    def _restore_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        for context in reversed(self._contexts):
            module = context._restore_module(module, in_keys, out_keys)
        return module

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = [context._hook_module(module) for context in self._contexts]
        return MultiHookHandle(handles)
