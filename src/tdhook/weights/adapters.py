from typing import Callable, Optional, List, Dict, Tuple
from torch import nn
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory, HookingContextWithCache
from tdhook.modules import HookedModule
from tdhook.hooks import DIRECTION_TO_RETURN, MultiHookHandle, HookDirection


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

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        cache = module.hooking_context.cache

        def callback_factory(adapter, cache_proxy=None):
            import inspect

            def callback(**kwargs):
                nonlocal adapter, cache_proxy
                if cache_proxy is not None:
                    adapter_input = cache_proxy.resolve()
                else:
                    adapter_input = kwargs.pop(DIRECTION_TO_RETURN[kwargs["direction"]])
                # Filter kwargs to only those accepted by the adapter
                adapter_params = inspect.signature(adapter).parameters
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in adapter_params}
                return adapter(adapter_input, **filtered_kwargs)

            return callback

        handles = []
        for direction in self._directions:
            for adapter, in_module_key, out_module_key in self._adapters.values():
                if in_module_key == out_module_key:
                    cache_proxy = None
                else:
                    handle, cache_proxy = module.get(
                        cache=cache,
                        module_key=in_module_key,
                        callback=self._cache_callback,
                        direction=direction,
                        relative=self._relative,
                    )
                    handles.append(handle)

                handle = module.set(
                    module_key=out_module_key,
                    value=None,
                    callback=callback_factory(adapter, cache_proxy),
                    direction=direction,
                    relative=self._relative,
                )
                handles.append(handle)
        return MultiHookHandle(handles)
