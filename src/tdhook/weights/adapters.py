import inspect
from typing import Callable, Optional, List, Dict, Tuple

from torch import nn
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory, HookingContextWithCache
from tdhook.modules import HookedModule
from tdhook.hooks import CacheProxy, DIRECTION_TO_RETURN, DIRECTION_TO_TYPE, HookDirection, HookFactory
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec


class HookedModuleWithAdapters(HookedModule):
    def __init__(self, *args, adapters: Dict[str, nn.Module], **kwargs):
        super().__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict(adapters)


class Adapters(HookingContextFactory):
    """
    ROME :cite:`Meng2022LocatingAE`, sparse autoencoders :cite:`Cunningham2023SparseAF` and transcoders :cite:`Dunefsky2024TranscodersFI`.
    """

    _hooked_module_class = HookedModuleWithAdapters
    _hooking_context_class = HookingContextWithCache

    def __init__(
        self,
        adapters: Dict[str, Tuple[nn.Module, str, str]],
        cache_callback: Optional[Callable] = None,
        relative: bool = True,
        directions: Optional[List[HookDirection]] = None,
        cache: Optional[TensorDict] = None,
        clear_cache: bool = True,
    ):
        super().__init__()
        self._hooked_module_kwargs["adapters"] = {k: v[0] for k, v in adapters.items()}
        self._hooking_context_kwargs["clear_cache"] = clear_cache
        self._hooking_context_kwargs["cache"] = cache

        self._adapters = adapters
        self._cache_callback = cache_callback
        self._relative = relative
        self._directions = directions or ["fwd"]

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        cache = module.hooking_context.cache
        relative_path = module.relative_path if self._relative else ""

        def callback_factory(adapter, cache_proxy=None):
            def callback(**kwargs):
                if cache_proxy is not None:
                    adapter_input = cache_proxy.resolve()
                else:
                    adapter_input = kwargs.pop(DIRECTION_TO_RETURN[kwargs["direction"]])
                # Filter kwargs to only those accepted by the adapter
                signature_target = adapter.forward if isinstance(adapter, nn.Module) else adapter
                adapter_params = inspect.signature(signature_target).parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in adapter_params.values()
                )
                filtered_kwargs = (
                    kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in adapter_params}
                )
                return adapter(adapter_input, **filtered_kwargs)

            return callback

        with HookProgramBuilder() as program:
            for direction in self._directions:
                for adapter, in_module_key, out_module_key in self._adapters.values():
                    if in_module_key == out_module_key:
                        cache_proxy = None
                    else:
                        cache_key = f"{in_module_key}_{DIRECTION_TO_TYPE[direction]}"
                        cache_proxy = CacheProxy(cache_key, cache)
                        program.register_path(
                            module,
                            HookFactory.make_caching_hook(
                                cache_key,
                                cache,
                                callback=self._cache_callback,
                                direction=direction,
                            ),
                            HookSpec(in_module_key, "capture", direction),
                            relative_path=relative_path,
                        )

                    program.register_path(
                        module,
                        HookFactory.make_setting_hook(
                            None,
                            callback=callback_factory(adapter, cache_proxy),
                            direction=direction,
                        ),
                        HookSpec(out_module_key, "replace", direction),
                        relative_path=relative_path,
                    )
            return program.build()
