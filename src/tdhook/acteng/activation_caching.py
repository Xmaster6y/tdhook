"""
Activation caching
"""

from typing import Callable, Optional, List

from tensordict import TensorDict

from tdhook.module import HookedModule
from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection, MultiHookHandle


class ActivationCaching(HookingContextFactory):
    def __init__(
        self,
        key_pattern: str,
        relative: bool = True,
        cache: Optional[TensorDict] = None,
        callback: Optional[Callable] = None,
        directions: Optional[List[HookDirection]] = None,
    ):
        self._cache = TensorDict() if cache is None else cache

        self._key_pattern = key_pattern
        self._hook_manager = MultiHookManager(key_pattern, relative=relative)
        self._callback = callback
        self._directions = directions or ["fwd"]

    @property
    def cache(self) -> TensorDict:
        return self._cache

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self
            return HookFactory.make_caching_hook(name, self._cache, direction=direction, callback=self._callback)

        handles = []
        for direction in self._directions:
            handles.append(
                self._hook_manager.register_hook(
                    module, lambda name: hook_factory(name, direction), direction=direction
                )
            )

        return MultiHookHandle(handles)
