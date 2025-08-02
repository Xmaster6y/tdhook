"""
HookedModule
"""

from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from typing import Callable, Any, Literal, Optional
import torch

from tdhook.hooks import register_hook_to_module, CacheProxy, HookFactory, EarlyStoppingException


class HookedModuleRun:
    def __init__(
        self,
        module: "HookedModule",
        data: TensorDict,
        cache: Optional[TensorDict] = None,
        run_name: str = "run",
        run_cache: Optional[TensorDict] = None,
        grad_enabled: bool = False,
    ):
        self._module = module
        self._data = data
        self._outer_cache = cache
        self._name = run_name
        self._cache = run_cache or TensorDict()
        self._grad_enabled = grad_enabled

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
                self._module(self._data)
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
            raise RuntimeError(f"Not in context, method {method} must be called in context")

    def set(self, key: str, value: Any, sep: str = ".", callback: Optional[Callable] = None) -> None:
        self._ensure_in_context("set")
        self._module.register_submodule_hook(
            key=key,
            hook=HookFactory.make_setting_hook(value, callback=callback),
            direction="fwd",
        )

    def get(self, key: str, sep: str = ".", callback: Optional[Callable] = None) -> CacheProxy:
        self._ensure_in_context("get")
        proxy = CacheProxy(key, self._cache, sep=sep)
        self._module.register_submodule_hook(
            key=key,
            hook=HookFactory.make_caching_hook(key, self._cache, sep=sep, callback=callback),
            direction="fwd",
        )
        return proxy

    def save(self, key: str, sep: str = ".", callback: Optional[Callable] = None) -> None:
        self._ensure_in_context("save")
        cache_key = self._name + sep + key
        proxy = CacheProxy(cache_key, self._save_cache, sep=sep)
        self._module.register_submodule_hook(
            key=key,
            hook=HookFactory.make_caching_hook(cache_key, self._save_cache, sep=sep, callback=callback),
            direction="fwd",
        )
        return proxy

    def stop(self, key: str) -> None:
        self._ensure_in_context("stop")
        self._module.register_submodule_hook(
            key=key,
            hook=HookFactory.make_stopping_hook(key),
            direction="fwd",
        )

    def set_grad(self):
        self._ensure_in_context("set_grad")
        self._grad_enabled = True

    def get_grad(self):
        self._ensure_in_context("get_grad")
        return self._grad_enabled

    def set_grad_enabled(self, grad_enabled: bool):
        self._ensure_in_context("set_grad_enabled")
        self._grad_enabled = grad_enabled


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

    def register_submodule_hook(
        self,
        key: str,
        hook: Callable,
        direction: Literal["fwd", "bwd", "fwd_pre", "bwd_pre"],
        prepend: bool = False,
        with_kwargs: bool = False,
    ):
        submodule = self
        if key != "":
            for subname in key.split("."):
                submodule = getattr(submodule, subname)
        return register_hook_to_module(submodule, hook, direction, prepend, with_kwargs)
