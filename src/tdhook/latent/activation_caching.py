from typing import Callable, Optional, List

from tensordict import TensorDict, TensorDictBase

from tdhook.modules import HookedModule
from tdhook.contexts import HookingContextFactory, HookingContextWithCache
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook._types import UnraveledKey


class ActivationCachingModule(HookedModule):
    """A prepared capture method that publishes a stable cache snapshot."""

    def __init__(self, *args, cache_key: UnraveledKey | None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_key = cache_key
        if cache_key is not None:
            if cache_key in self.out_keys:
                raise ValueError(f"Activation cache key {cache_key!r} collides with a model output key")
            self.out_keys = [*self.out_keys, cache_key]

    def finalize_tensordict(self, data: TensorDictBase) -> TensorDictBase:
        if self.cache_key is None:
            return data
        if self.hooking_context is None:
            raise RuntimeError("ActivationCachingModule requires a prepared hooking context")
        return data.set(self.cache_key, self.hooking_context.cache.copy())

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if isinstance(result, TensorDictBase):
            return self.finalize_tensordict(result)
        return result


class ActivationCaching(HookingContextFactory):
    """
    Maximally activating samples :cite:`Chen2020ConceptWF` and attention visualisation :cite:`Abnar2020QuantifyingAF`.
    """

    _hooking_context_class = HookingContextWithCache
    _hooked_module_class = ActivationCachingModule

    def __init__(
        self,
        key_pattern: str,
        relative: bool = True,
        cache: Optional[TensorDict] = None,
        callback: Optional[Callable] = None,
        directions: Optional[List[HookDirection]] = None,
        use_nested_keys: bool = False,
        clear_cache: bool = True,
        cache_key: UnraveledKey | None = "cache",
    ):
        super().__init__()
        if cache_key is not None and not isinstance(cache_key, UnraveledKey):
            raise TypeError("cache_key must be a TensorDict nested key or None")
        self._hooking_context_kwargs["cache"] = cache
        self._hooking_context_kwargs["clear_cache"] = clear_cache
        self._hooked_module_kwargs["cache_key"] = cache_key

        self._key_pattern = key_pattern
        self._relative = relative
        self._hook_manager = MultiHookManager(key_pattern)
        self._callback = callback
        self._directions = directions or ["fwd"]
        self._use_nested_keys = use_nested_keys or len(self._directions) > 1

    @property
    def cache_key(self) -> UnraveledKey | None:
        """Return the native TensorDict key used to publish captured activations."""

        return self._hooked_module_kwargs["cache_key"]

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        cache = module.hooking_context.cache

        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self, cache
            key = (direction, name) if self._use_nested_keys else name
            return HookFactory.make_caching_hook(key, cache, direction=direction, callback=self._callback)

        with HookProgramBuilder() as program:
            for direction in self._directions:
                for name, submodule in self._hook_manager.iter_modules(
                    module,
                    relative_path=module.relative_path if self._relative else None,
                ):
                    program.register(
                        submodule,
                        hook_factory(name, direction),
                        HookSpec(name, "capture", direction),
                    )

            return program.build()
