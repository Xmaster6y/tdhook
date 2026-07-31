from contextlib import contextmanager
from contextlib import ExitStack
from typing import List, Optional, Generator, Dict, overload, Literal
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
        self._managed_by_context_manager = False

    def _enter(self, managed_by_context_manager: bool = True):
        if self._in_context:
            raise RuntimeError("Cannot enter context twice")
        self._in_context = True
        self._managed_by_context_manager = managed_by_context_manager

        working_module = self._module
        prepared = False
        try:
            with ExitStack() as stack:
                for factory in self._pre_factories:
                    working_module = stack.enter_context(
                        factory.prepare(working_module, self._in_keys, self._out_keys)
                    )
                self._stack = stack.pop_all()
            prep_module = self._prepare(working_module, self._in_keys, self._out_keys, self._extra_relative_path)
            prepared = True
            self._hooked_module = self._spawn(prep_module, self, self._extra_relative_path)
            self._handle = self._hook(self._hooked_module)
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
                    self._hooked_module = None
                    self._handle = None
                    self._stack = None

    def __enter__(self):
        return self._enter(managed_by_context_manager=True)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._handle is not None:
                self._handle.remove()
            self._restore(self._module, self._in_keys, self._out_keys, self._extra_relative_path)
        finally:
            self._in_context = False
            self._hooked_module = None
            self._handle = None
            if self._stack is not None:
                self._stack.__exit__(exc_type, exc_value, traceback)
                self._stack = None

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

    def _enter(self, managed_by_context_manager: bool = True):
        if self._clear_cache:
            self.clear()
        return super()._enter(managed_by_context_manager=managed_by_context_manager)

    def __enter__(self):
        return self._enter(managed_by_context_manager=True)


class HookingContextFactory:
    """
    Factory for creating hooking contexts.
    """

    _hooked_module_class = HookedModule
    _hooking_context_class = HookingContext

    def __init__(self):
        self._hooking_context_kwargs = {}
        self._hooked_module_kwargs = {}

    @overload
    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey] | Dict[UnraveledKey, str]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
        *,
        return_context: Literal[True] = True,
    ) -> "HookingContext": ...

    @overload
    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey] | Dict[UnraveledKey, str]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
        *,
        return_context: Literal[False],
    ) -> HookedModule: ...

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[UnraveledKey] | Dict[UnraveledKey, str]] = None,
        out_keys: Optional[List[UnraveledKey]] = None,
        *,
        return_context: bool = True,
    ) -> "HookingContext | HookedModule":
        """
        Prepare the module for execution.

        Args:
            module: The module to prepare.
            in_keys: Optional input keys.
            out_keys: Optional output keys.
            return_context: If True (default), returns a context manager. If False, returns the hooked module directly.

        Returns:
            If return_context is True, returns a HookingContext that can be used as a context manager.
            If return_context is False, returns the HookedModule directly (context is automatically entered).
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

        context = self._hooking_context_class(self, module, in_keys, out_keys, **self._hooking_context_kwargs)

        if return_context:
            return context
        else:
            return context._enter(managed_by_context_manager=False)

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
        self._validate_capabilities()

    def _validate_capabilities(self) -> None:
        """Reject capabilities that cannot be represented by one shared context.

        A same-run group owns one :class:`HookingContext` and one
        :class:`HookedModule`.  Factories that need a specialised context or
        wrapper therefore cannot be safely composed yet; failing before any
        module mutation is preferable to silently giving the child the wrong
        runtime object.
        """
        for context in self._contexts:
            if type(context)._hooking_context_class is not HookingContext:
                raise ValueError(
                    f"Cannot compose {type(context).__name__}: it requires the specialised "
                    f"{type(context)._hooking_context_class.__name__} capability. "
                    "Same-run composition currently supports factories using HookingContext."
                )
            if type(context)._hooked_module_class is not HookedModule:
                raise ValueError(
                    f"Cannot compose {type(context).__name__}: it requires the specialised "
                    f"{type(context)._hooked_module_class.__name__} capability. "
                    "Same-run composition currently supports factories using HookedModule."
                )
            if type(context)._spawn_hooked_module is not HookingContextFactory._spawn_hooked_module:
                raise ValueError(
                    f"Cannot compose {type(context).__name__}: it customises hooked-module spawning, "
                    "which is not a shared-context capability."
                )

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        prepared_contexts = []
        original_module = module
        try:
            for context in self._contexts:
                module = context._prepare_module(module, in_keys, out_keys, extra_relative_path)
                prepared_contexts.append(context)
            return module
        except BaseException:
            # A child can mutate the original module before raising.  Restore
            # the failing child as well as every successfully prepared child
            # in reverse order, just as on a normal context exit. Preserve the
            # original exception.
            for context in reversed([context, *prepared_contexts]):
                try:
                    context._restore_module(original_module, in_keys, out_keys, extra_relative_path)
                except BaseException:
                    pass
            raise

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
        handles = []
        try:
            for context in self._contexts:
                handles.append(context._hook_module(self._child_module(module)))
            return MultiHookHandle(handles)
        except BaseException:
            # Hooks registered by earlier children are live immediately.  Do
            # not leave them installed when a later child cannot be installed.
            MultiHookHandle(handles).remove()
            raise

    @staticmethod
    def _child_module(module: HookedModule) -> HookedModule:
        """Give a child a view whose relative root is the original module.

        Preparation often wraps the input in a ``TensorDictSequential``.  The
        original TensorDict module remains an object inside that final wrapper;
        locating it by identity means every child's paths continue to resolve
        as they did before composition, independent of the order of rewrites.
        """
        original = module.hooking_context._module
        for name, candidate in module.td_module.named_modules(remove_duplicate=False):
            if candidate is original:
                relative_path = merge_paths("td_module", name, module.hooking_context._extra_relative_path)
                return HookedModule(
                    module.td_module, hooking_context=module.hooking_context, relative_path=relative_path
                )
        raise RuntimeError("Cannot compose hooks: the prepared module no longer contains the original module")


# A clearer public name for the same-run composition API.  Keep the original
# name for backwards compatibility.
HookGroup = CompositeHookingContextFactory
