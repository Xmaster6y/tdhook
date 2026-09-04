from typing import Callable, Optional, List

from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey

from tdhook.modules import HookedModule
from tdhook.contexts import HookingContextFactory, HookingContextWithCache
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook.targets import Target
from tdhook._types import is_nested_key


def _key_path(key: NestedKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _keys_overlap(left: NestedKey, right: NestedKey) -> bool:
    left_path = _key_path(left)
    right_path = _key_path(right)
    common = min(len(left_path), len(right_path))
    return left_path[:common] == right_path[:common]


class ActivationCachingModule(HookedModule):
    """A bound capture method that publishes a shallow cache snapshot.

    Tensor leaves retain their native storage. In particular, publishing a
    caller-owned memory-mapped cache does not materialize its leaves in memory.
    """

    def __init__(self, *args, cache_key: NestedKey | None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_key = cache_key
        if cache_key is not None:
            if any(_keys_overlap(cache_key, output_key) for output_key in self.out_keys):
                raise ValueError(f"Activation cache key {cache_key!r} collides with a model output key")
            self.out_keys = [*self.out_keys, cache_key]

    def finalize_tensordict(self, data: TensorDictBase) -> TensorDictBase:
        if self.cache_key is None:
            return data
        if self.hooking_context is None:
            raise RuntimeError("ActivationCachingModule requires an active method context")
        return data.set(self.cache_key, self.hooking_context.cache.copy())


class ActivationCaching(HookingContextFactory):
    """Cache model activations in a caller-visible TensorDict.

    A locked or memory-mapped ``cache`` must contain every captured key with
    its final shape, dtype, and device, and must be paired with
    ``clear_cache=False``. Captures then update that storage in place. TDHook
    neither creates nor removes caller-owned memory-map paths.

    Maximally activating samples :cite:`Chen2020ConceptWF` and attention visualisation :cite:`Abnar2020QuantifyingAF`.
    """

    _hooking_context_class = HookingContextWithCache
    _hooked_module_class = ActivationCachingModule

    def __init__(
        self,
        key_pattern: str | Target,
        relative: bool = True,
        cache: Optional[TensorDict] = None,
        callback: Optional[Callable] = None,
        directions: Optional[List[HookDirection]] = None,
        use_nested_keys: bool = False,
        clear_cache: bool = True,
        cache_key: NestedKey | None = "cache",
    ):
        super().__init__()
        if isinstance(key_pattern, Target):
            if key_pattern.kind != "activation":
                raise ValueError("activation caching requires an activation Target")
            if relative is not True:
                raise ValueError("Target module paths are always relative to the caller-owned model")
            self._target = key_pattern
            resolved_pattern = None
        elif isinstance(key_pattern, str):
            self._target = None
            resolved_pattern = key_pattern
        else:
            raise TypeError("key_pattern must be a module pattern or Target")
        if cache_key is not None and not is_nested_key(cache_key):
            raise TypeError("cache_key must be a TensorDict nested key or None")
        if cache is not None and cache.is_locked and clear_cache:
            raise ValueError("locked or memory-mapped caches require clear_cache=False")
        self._hooking_context_kwargs["cache"] = cache
        self._hooking_context_kwargs["clear_cache"] = clear_cache
        self._hooked_module_kwargs["cache_key"] = cache_key

        self._key_pattern = key_pattern
        self._relative = relative
        self._hook_manager = MultiHookManager(resolved_pattern)
        self._callback = callback
        self._directions = directions or ["fwd"]
        if self._target is not None and self._directions != ["fwd"]:
            raise ValueError("Target activation caching currently supports only the forward direction")
        self._use_nested_keys = use_nested_keys or len(self._directions) > 1

    @property
    def cache_key(self) -> NestedKey | None:
        """Return the native TensorDict key used to publish captured activations."""

        return self._hooked_module_kwargs["cache_key"]

    @property
    def key_pattern(self) -> str | Target:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._target = None
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        cache = module.hooking_context.cache

        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self, cache
            key = (direction, name) if self._use_nested_keys else name

            def callback(**kwargs):
                value = kwargs["output"]
                if self._callback is not None:
                    value = self._callback(**kwargs)
                return self._target.select_output(value)

            return HookFactory.make_caching_hook(
                key,
                cache,
                direction=direction,
                callback=callback if self._target is not None else self._callback,
            )

        with HookProgramBuilder() as program:
            if self._target is not None:
                program.register_target(
                    module.hook_root,
                    hook_factory(self._target.module_path, "fwd"),
                    HookSpec(self._target.module_path, "capture", "fwd", target=self._target),
                    relative_path=module.relative_path,
                )
                return program.build()
            for direction in self._directions:
                for name, submodule in self._hook_manager.iter_modules(
                    module.hook_root,
                    relative_path=module.relative_path if self._relative else None,
                ):
                    program.register(
                        submodule,
                        hook_factory(name, direction),
                        HookSpec(name, "capture", direction, target=self._target),
                    )

            return program.build()
