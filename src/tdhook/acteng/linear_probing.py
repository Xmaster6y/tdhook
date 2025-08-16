"""
Linear probing
"""

from typing import Callable, Optional, List

from torch import nn
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection, MultiHookHandle, DIRECTION_TO_RETURN


class LinearProbing(HookingContextFactory):
    # TODO: refactor to use ActivationCaching and focus on minibatch training
    def __init__(
        self,
        key_pattern: str,
        probe_factory: Callable[[str, str], Callable],
        relative: bool = True,
        cache: Optional[TensorDict] = None,
        preprocess: Optional[Callable] = None,
        directions: Optional[List[HookDirection]] = None,
    ):
        self._cache = TensorDict() if cache is None else cache

        self._key_pattern = key_pattern
        self._hook_manager = MultiHookManager(key_pattern, relative=relative)
        self._probe_factory = probe_factory
        self._preprocess = preprocess
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

    def _hook_module(self, module: nn.Module) -> MultiHookHandle:
        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self
            probe = self._probe_factory(name, direction)
            if self._preprocess is not None:

                def callback(**kwargs):
                    data = self._preprocess(**kwargs, name=name, direction=direction)
                    return probe(data[DIRECTION_TO_RETURN[direction]])

            else:

                def callback(**kwargs):
                    return probe(kwargs[DIRECTION_TO_RETURN[direction]])

            return HookFactory.make_caching_hook(name, self._cache, direction=direction, callback=callback)

        handles = []
        for direction in self._directions:
            handles.append(
                self._hook_manager.register_hook(
                    module, lambda name: hook_factory(name, direction), direction=direction
                )
            )

        return MultiHookHandle(handles)
