"""
HookedModule
"""

from torch.utils.hooks import RemovableHandle
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from typing import Callable, Any, Optional, Tuple
import torch

from tdhook.hooks import register_hook_to_module, CacheProxy, HookFactory, EarlyStoppingException, HookDirection


class HookedModuleRun:
    def __init__(
        self,
        module: "HookedModule",
        data: TensorDict,
        cache: Optional[TensorDict] = None,
        run_name: str = "run",
        run_sep: str = ".",
        run_cache: Optional[TensorDict] = None,
        grad_enabled: bool = False,
        run_callback: Optional[Callable] = None,
    ):
        self._module = module
        self._data = data
        self._outer_cache = cache
        self._name = run_name
        self._sep = run_sep
        self._cache = run_cache or TensorDict()
        self._grad_enabled = grad_enabled
        self._run_callback = run_callback or (lambda module, data: module(data))

        if self._outer_cache is None:
            self._save_cache = self._cache
        else:
            self._save_cache = self._outer_cache

        self._handles = []
        self._in_context = False

    @property
    def cache(self) -> TensorDict:
        return self._cache

    @cache.setter
    def cache(self, cache: TensorDict):
        self._cache = cache

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            with torch.set_grad_enabled(self._grad_enabled):
                self._run_callback(self._module, self._data)
        except EarlyStoppingException:
            pass
        except Exception as e:
            raise e
        finally:
            for handle in self._handles:
                handle.remove()
            self._in_context = False

    def _ensure_in_context(self, method: str):
        if not self._in_context:
            raise RuntimeError(f"Not in context, method {method} must be called in context or directly on the module")

    def set(
        self,
        key: str,
        value: Any,
        *,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
    ) -> None:
        self._ensure_in_context("set")
        handle = self._module.set(key, value, callback=callback, direction=direction, prepend=prepend)
        self._handles.append(handle)

    def get(
        self, key: str, *, callback: Optional[Callable] = None, direction: HookDirection = "fwd", prepend: bool = False
    ) -> CacheProxy:
        self._ensure_in_context("get")
        handle, proxy = self._module.get(self._cache, key, callback=callback, direction=direction, prepend=prepend)
        self._handles.append(handle)
        return proxy

    def save(
        self, key: str, *, callback: Optional[Callable] = None, direction: HookDirection = "fwd", prepend: bool = False
    ) -> None:
        self._ensure_in_context("save")
        cache_key = self._name + self._sep + key
        handle, proxy = self._module.save(
            self._save_cache, key, cache_key=cache_key, callback=callback, direction=direction, prepend=prepend
        )
        self._handles.append(handle)
        return proxy

    def set_grad(self, *args, **kwargs):
        self._ensure_in_context("set_grad")
        self._grad_enabled = True
        kwargs["direction"] = "bwd"
        self.set("grad", *args, **kwargs)

    def get_grad(self, *args, **kwargs):
        self._ensure_in_context("get_grad")
        self._grad_enabled = True
        kwargs["direction"] = "bwd"
        return self.get("grad", *args, **kwargs)

    def save_grad(self, *args, **kwargs):
        self._ensure_in_context("save_grad")
        self._grad_enabled = True
        kwargs["direction"] = "bwd"
        return self.save("grad", *args, **kwargs)

    def set_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        self.set("input", *args, **kwargs)

    def get_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        return self.get("input", *args, **kwargs)

    def save_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        return self.save("input", *args, **kwargs)

    def set_grad_input(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        self.set("grad_input", *args, **kwargs)

    def get_grad_input(self, *args, **kwargs):
        kwargs["direction"] = "bwd"
        return self.get("grad_input", *args, **kwargs)

    def save_grad_input(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        return self.save("grad_input", *args, **kwargs)

    def stop(self, key: str) -> None:
        self._ensure_in_context("stop")
        handle = self._module.stop(key)
        self._handles.append(handle)


class HookedModule(TensorDictModule):
    def run(
        self,
        data: TensorDict,
        cache: Optional[TensorDict] = None,
        run_name: Optional[str] = None,
        run_cache: Optional[TensorDict] = None,
        grad_enabled: bool = False,
    ) -> HookedModuleRun:
        return HookedModuleRun(self, data, cache, run_name, run_cache, grad_enabled)

    def _resolve_submodule_path(self, key: str):
        """
        Resolve a submodule path that may contain indexing expressions.

        Supports any valid Python attribute access and indexing:
        - "layers[-1]" -> self.layers[-1]
        - "layers['attr']" -> self.layers['attr']
        - "layers.attention" -> self.layers.attention
        - "layers[1:3]" -> self.layers[1:3]
        """
        if key == "":
            return self

        # Create a safe environment with only the current module
        safe_dict = {"self": self}

        try:
            # Evaluate the expression in the safe environment
            return eval(f"self.{key}", {"__builtins__": {}}, safe_dict)
        except (AttributeError, IndexError, KeyError, SyntaxError) as e:
            raise ValueError(f"Invalid submodule path '{key}': {e}")

    def register_submodule_hook(
        self,
        key: str,
        hook: Callable,
        direction: HookDirection,
        prepend: bool = False,
    ):
        submodule = self._resolve_submodule_path(key)
        return register_hook_to_module(submodule, hook, direction, prepend)

    def set(
        self,
        moduel_key: str,
        value: Any,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
    ) -> RemovableHandle:
        handle = self.register_submodule_hook(
            key=moduel_key,
            hook=HookFactory.make_setting_hook(value, callback=callback, direction=direction),
            direction=direction,
            prepend=prepend,
        )
        return handle

    def get(
        self,
        cache: TensorDict,
        moduel_key: str,
        cache_key: Optional[str] = None,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
    ) -> Tuple[RemovableHandle, CacheProxy]:
        cache_key = cache_key or moduel_key
        proxy = CacheProxy(cache_key, cache)
        handle = self.register_submodule_hook(
            key=moduel_key,
            hook=HookFactory.make_caching_hook(cache_key, cache, callback=callback, direction=direction),
            direction=direction,
            prepend=prepend,
        )
        return handle, proxy

    def set_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        self.set("input", *args, **kwargs)

    def get_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd"
        return self.get("input", *args, **kwargs)

    def set_grad(self, *args, **kwargs):
        kwargs["direction"] = "bwd"
        self.set("grad", *args, **kwargs)

    def get_grad(self, *args, **kwargs):
        kwargs["direction"] = "bwd"
        return self.get("grad", *args, **kwargs)

    def set_grad_input(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        self.set("grad_input", *args, **kwargs)

    def get_grad_input(self, *args, **kwargs):
        kwargs["direction"] = "bwd"
        return self.get("grad_input", *args, **kwargs)

    def stop(self, key: str) -> None:
        handle = self.register_submodule_hook(
            key=key,
            hook=HookFactory.make_stopping_hook(key),
            direction="fwd",
        )
        return handle
