from typing import Callable, Optional, List

from tensordict import TensorDict

from tdhook.modules import HookedModule
from tdhook.contexts import HookingContextFactory, HookingContextWithCache
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection, MultiHookHandle


class ActivationCaching(HookingContextFactory):
    """
    Maximally activating samples :cite:`Chen2020ConceptWF` and attention visualisation :cite:`Abnar2020QuantifyingAF`.
    """

    _hooking_context_class = HookingContextWithCache

    def __init__(
        self,
        key_pattern: str,
        relative: bool = True,
        cache: Optional[TensorDict] = None,
        callback: Optional[Callable] = None,
        directions: Optional[List[HookDirection]] = None,
        use_nested_keys: bool = False,
        clear_cache: bool = True,
    ):
        super().__init__()
        self._hooking_context_kwargs["cache"] = cache
        self._hooking_context_kwargs["clear_cache"] = clear_cache

        self._key_pattern = key_pattern
        self._relative = relative
        self._hook_manager = MultiHookManager(key_pattern)
        self._callback = callback
        self._directions = directions or ["fwd"]
        self._use_nested_keys = use_nested_keys or len(self._directions) > 1

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        cache = module.hooking_context.cache

        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self, cache
            key = (direction, name) if self._use_nested_keys else name
            return HookFactory.make_caching_hook(key, cache, direction=direction, callback=self._callback)

        handles = []
        for direction in self._directions:
            handles.append(
                self._hook_manager.register_hook(
                    module,
                    (lambda name: hook_factory(name, direction)),
                    direction=direction,
                    relative_path=module.relative_path if self._relative else None,
                )
            )

        return MultiHookHandle(handles)
