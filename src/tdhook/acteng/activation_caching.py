"""
Activation caching
"""

from typing import Callable, Optional, Generator, List
from contextlib import contextmanager

from torch import nn
from tensordict import TensorDict

from tdhook.contexts import BaseContext
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection, MultiHookHandle


class ActivationCaching(BaseContext):
    def __init__(
        self,
        key_pattern: str,
        cache: Optional[TensorDict] = None,
        callback: Optional[Callable] = None,
        directions: Optional[List[HookDirection]] = None,
    ):
        self.cache = cache or TensorDict()

        self._key_pattern = key_pattern
        self._hook_manager = MultiHookManager(key_pattern)
        self._callback = callback
        self._directions = directions or ["fwd"]

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    @contextmanager
    def _hook_module(self, module: nn.Module) -> Generator[None, None, None]:
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

        with MultiHookHandle(handles):
            yield
