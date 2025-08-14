"""
Context
"""

from contextlib import contextmanager
from typing import List, Optional, Generator
from torch import nn
from tensordict.nn import TensorDictModule

from tdhook.module import HookedModule
from tdhook.hooks import MultiHookHandle


class HookingContext:
    def __init__(
        self,
        factory: "HookingContextFactory",
        module: nn.Module,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ):
        if isinstance(module, TensorDictModule):
            self._prepare = factory._prepare_td_module
            self._restore = factory._restore_td_module
            in_keys = in_keys or module.in_keys
            out_keys = out_keys or module.out_keys
        else:
            self._prepare = factory._prepare_module
            self._restore = factory._restore_module
            in_keys = in_keys or ["input"]
            out_keys = out_keys or ["output"]

        self._spawn = factory._spawn_hooked_module
        self._hook = factory._hook_module
        self._in_context = False
        self._handle = None
        self._hooked_module = None

        self._module = module
        self._in_keys = in_keys
        self._out_keys = out_keys

    def __enter__(self):
        if self._in_context:
            raise RuntimeError("Cannot enter context twice")
        self._in_context = True
        prep_module = self._prepare(self._module)
        self._hooked_module = self._spawn(prep_module, self._in_keys, self._out_keys, self)
        self._handle = self._hook(self._hooked_module)
        return self._hooked_module

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
        self._restore(self._module)
        self._in_context = False
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
                yield self._restore(self._hooked_module.module)
            finally:
                self._hooked_module.module = self._prepare(self._module)


class HookingContextFactory:
    """
    Factory for creating hooking contexts.
    """

    _hooked_module_class = HookedModule
    _hooking_context_class = HookingContext

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ) -> "HookingContext":
        """
        Prepare the module for execution.
        """

        return self._hooking_context_class(self, module, in_keys, out_keys)

    def _prepare_module(
        self,
        module: nn.Module,
    ) -> nn.Module:
        return module

    def _restore_module(self, module: nn.Module) -> nn.Module:
        return module

    def _prepare_td_module(
        self,
        td_module: TensorDictModule,
    ) -> TensorDictModule:
        td_module.module = self._prepare_module(td_module.module)
        return td_module

    def _restore_td_module(self, td_module: TensorDictModule) -> TensorDictModule:
        td_module.module = self._restore_module(td_module.module)
        return td_module

    def _spawn_hooked_module(
        self, prep_module: nn.Module, in_keys: List[str], out_keys: List[str], hooking_context: "HookingContext"
    ):
        if isinstance(prep_module, TensorDictModule):
            prep_module.in_keys = in_keys
            prep_module.out_keys = out_keys
            return self._hooked_module_class(prep_module, hooking_context=hooking_context)

        else:
            return self._hooked_module_class.from_module(
                prep_module, in_keys, out_keys, hooking_context=hooking_context
            )

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return MultiHookHandle()


class CompositeHookingContextFactory(HookingContextFactory):
    def __init__(self, *contexts: HookingContextFactory):
        self._contexts = contexts
        for context in contexts:
            if type(context)._spawn_hooked_module != HookingContextFactory._spawn_hooked_module:
                raise ValueError("Cannot compose factories that override _spawn_hooked_module")

    def _prepare_module(
        self,
        module: nn.Module,
    ) -> nn.Module:
        for context in self._contexts:
            module = context._prepare_module(module)
        return module

    def _restore_module(self, module: nn.Module) -> nn.Module:
        for context in reversed(self._contexts):
            module = context._restore_module(module)
        return module

    def _prepare_td_module(
        self,
        td_module: TensorDictModule,
    ) -> TensorDictModule:
        for context in self._contexts:
            td_module = context._prepare_td_module(td_module)
        return td_module

    def _restore_td_module(self, td_module: TensorDictModule) -> TensorDictModule:
        for context in reversed(self._contexts):
            td_module = context._restore_td_module(td_module)
        return td_module

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []
        for context in self._contexts:
            handles.append(context._hook_module(module))
        return MultiHookHandle(handles)
