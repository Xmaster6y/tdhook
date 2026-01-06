from torch.utils.hooks import RemovableHandle
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper, TensorDictModuleBase
from tensordict import TensorDict, NonTensorData
from typing import Callable, Any, Optional, Tuple, TYPE_CHECKING, List
import torch
import warnings
import torch.nn as nn
from contextlib import contextmanager
from textwrap import indent

from tdhook.hooks import (
    register_hook_to_module,
    CacheProxy,
    HookFactory,
    EarlyStoppingException,
    resolve_submodule_path,
    HookDirection,
    DIRECTION_TO_TYPE,
    MutableWeakRef,
    TensorDictRef,
)
from tdhook._types import UnraveledKey

if TYPE_CHECKING:
    from tdhook.contexts import HookingContext


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def flatten_select_reshape_call(
    module: TensorDictModuleBase, td: TensorDict, flatten: bool = True, select: bool = True, reshape: bool = True
) -> TensorDict:
    _td = td.flatten() if flatten else td
    _td = module(_td)
    _td = _td.select(*module.out_keys) if select else _td
    _td = _td.reshape(td.shape) if reshape else _td
    return _td


class FunctionModule(TensorDictModuleBase):
    """
    Wrapper for a function to be used as a module.
    """

    def __init__(
        self, td_fn: Callable[[TensorDict], TensorDict], in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._td_fn = td_fn

    def forward(self, td: TensorDict) -> TensorDict:
        return self._td_fn(td)

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys},\ntd_fn={self._td_fn}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


class ModuleCall(TensorDictModuleBase):
    """
    Wrapper to manage module calls.
    """

    def __init__(
        self,
        td_module: TensorDictModuleBase,
        in_key: Optional[UnraveledKey] = None,
        out_key: Optional[UnraveledKey] = None,
        flatten: bool = True,
    ):
        super().__init__()
        self.in_keys = [k if in_key is None else (in_key, k) for k in td_module.in_keys]
        self.out_keys = [k if out_key is None else (out_key, k) for k in td_module.out_keys]

        self._td_module = td_module
        self._in_key = in_key
        self._out_key = out_key
        self._flatten = flatten

    def forward(self, td: TensorDict) -> TensorDict:
        inputs = td if self._in_key is None else td[self._in_key]
        outputs = flatten_select_reshape_call(self._td_module, inputs, flatten=self._flatten)

        if self._out_key is not None:
            prev_out = td.get(self._out_key)
            if isinstance(prev_out, TensorDict):
                prev_out.update(outputs)
            else:
                td[self._out_key] = outputs
        else:
            td.update(outputs)

        return td

    def __repr__(self):
        fields = indent(
            f"td_module={self._td_module},\nin_keys={self.in_keys},\nout_keys={self.out_keys}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


class ModuleCallWithCache(TensorDictModuleBase):
    """
    Wrapper to manage module calls with cache.
    """

    def __init__(
        self,
        td_module: TensorDictModuleBase,
        stored_keys: List[UnraveledKey],
        cache_key: Optional[UnraveledKey] = None,
        in_key: Optional[UnraveledKey] = None,
        out_key: Optional[UnraveledKey] = None,
        cache_ref: Optional[MutableWeakRef | TensorDictRef] = None,
        flatten: bool = True,
        cache_as_output: bool = True,
    ):
        super().__init__()
        self.in_keys = [k if in_key is None else (in_key, k) for k in td_module.in_keys]

        if cache_as_output:
            self.out_keys = [k if out_key is None else (out_key, k) for k in td_module.out_keys] + [
                k if cache_key is None else (cache_key, k) for k in stored_keys
            ]
        else:
            self.out_keys = [k if out_key is None else (out_key, k) for k in td_module.out_keys]

        self._td_module = td_module
        self._cache_key = cache_key
        self._in_key = in_key
        self._out_key = out_key
        self._flatten = flatten
        self._cache_as_output = cache_as_output

        self._cache_ref = cache_ref or MutableWeakRef(TensorDict())

    @property
    def cache_ref(self) -> MutableWeakRef | TensorDictRef:
        return self._cache_ref

    def forward(self, td: TensorDict) -> TensorDict:
        inputs = td if self._in_key is None else td[self._in_key]
        cache = TensorDict(batch_size=inputs.batch_size, device=inputs.device).flatten()
        self._cache_ref.set(cache)

        outputs = flatten_select_reshape_call(self._td_module, inputs, flatten=self._flatten)

        if self._out_key is not None:
            td[self._out_key] = outputs
        else:
            td.update(outputs)

        if self._cache_as_output and self._cache_key is not None:
            td[self._cache_key] = cache.reshape(inputs.shape)
        elif self._cache_as_output:
            td.update(cache.reshape(inputs.shape))
        else:
            cache["_shape"] = NonTensorData(tuple(inputs.shape))

        return td

    def __repr__(self):
        fields = indent(
            f"td_module={self._td_module},\nin_keys={self.in_keys},\nout_keys={self.out_keys}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


class PGDModule(TensorDictModuleBase):
    """
    Wrapper to manage PGD module calls.
    """

    def __init__(
        self,
        td_module: TensorDictModuleBase,
        alpha: float = 0.1,
        n_steps: int = 10,
        min_value: float = -float("Inf"),
        max_value: float = float("Inf"),
        grad_key: UnraveledKey = "_grad",
        working_key: UnraveledKey = "_working",
        ascent: bool = False,
        use_sign: bool = True,
    ):
        super().__init__()
        self._td_module = td_module

        self.in_keys = td_module.in_keys
        self.out_keys = [k if working_key is None else (working_key, k) for k in td_module.out_keys]

        self._alpha = alpha
        self._n_steps = n_steps
        self._min_value = min_value
        self._max_value = max_value
        self._grad_key = grad_key
        self._working_key = working_key
        self._ascent = ascent
        self._use_sign = use_sign

    def forward(self, td: TensorDict) -> TensorDict:
        working_td = td if self._working_key is None else td[self._working_key]
        for _ in range(self._n_steps):
            working_td = self._td_module(working_td)
            working_td = self._pgd_step(working_td)
        if self._working_key is not None:
            td[self._working_key] = working_td
        else:
            td.update(working_td)
        return td

    def _pgd_step(self, td: TensorDict) -> TensorDict:
        grads: TensorDict = td[self._grad_key]
        if self._ascent:
            grads = -grads
        if self._use_sign:
            grads = torch.sign(grads)
        for key in grads.keys(True, True):
            td[key] = torch.clamp(td[key] - self._alpha * grads[key], min=self._min_value, max=self._max_value)
        return td

    def __repr__(self):
        fields = indent(f"td_module={self._td_module},\nin_keys={self.in_keys},\nout_keys={self.out_keys},\n", 4 * " ")
        return f"{type(self).__name__}(\n{fields})"


class IntermediateKeysCleaner(TensorDictModuleBase):
    """
    Wrapper to clean intermediate keys.
    """

    def __init__(self, intermediate_keys: List[UnraveledKey]):
        super().__init__()
        self.in_keys = intermediate_keys
        self.out_keys = []

        self._intermediate_keys = intermediate_keys

    def forward(self, td: TensorDict) -> TensorDict:
        return td.exclude(*self._intermediate_keys)

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


class HookedModuleRun:
    """
    Context manager to execute module runs.
    """

    def __init__(
        self,
        module: "HookedModule",
        data: TensorDict,
        cache: Optional[TensorDict] = None,
        run_name: Optional[str] = None,
        run_sep: Optional[str] = None,
        run_cache: Optional[TensorDict] = None,
        grad_enabled: bool = False,
        run_callback: Optional[Callable] = None,
    ):
        self._module = module
        self._data = data
        self._outer_cache = cache
        self._name = run_name or "run"
        self._sep = run_sep or "."
        self._cache = TensorDict() if run_cache is None else run_cache
        self._grad_enabled = grad_enabled
        self._run_callback = run_callback or (lambda module, data: module(data))

        self._save_cache = self._cache if self._outer_cache is None else self._outer_cache
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
        relative: bool = True,
    ) -> None:
        self._ensure_in_context("set")
        handle = self._module.set(
            key, value, callback=callback, direction=direction, prepend=prepend, relative=relative
        )
        self._handles.append(handle)

    def get(
        self,
        key: str,
        *,
        cache_key: Optional[str] = None,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
        relative: bool = True,
    ) -> CacheProxy:
        self._ensure_in_context("get")
        handle, proxy = self._module.get(
            self._cache, key, cache_key, callback=callback, direction=direction, prepend=prepend, relative=relative
        )
        self._handles.append(handle)
        return proxy

    def save(
        self,
        key: str,
        *,
        cache_key: Optional[str] = None,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
        relative: bool = True,
    ) -> CacheProxy:
        self._ensure_in_context("save")
        cache_key = cache_key or f"{self._name + self._sep + key}_{DIRECTION_TO_TYPE[direction]}"
        handle, proxy = self._module.get(
            self._save_cache,
            key,
            cache_key=cache_key,
            callback=callback,
            direction=direction,
            prepend=prepend,
            relative=relative,
        )
        self._handles.append(handle)
        return proxy

    # TODO: rename grad_input
    def set_grad(self, *args, **kwargs):
        self._ensure_in_context("set_grad")
        self._grad_enabled = True
        kwargs["direction"] = "bwd"
        self.set(*args, **kwargs)

    def get_grad(self, *args, **kwargs):
        self._ensure_in_context("get_grad")
        self._grad_enabled = True
        kwargs["direction"] = "bwd"
        return self.get(*args, **kwargs)

    def save_grad(self, *args, **kwargs):
        self._ensure_in_context("save_grad")
        self._grad_enabled = True
        kwargs["direction"] = "bwd"
        return self.save(*args, **kwargs)

    def set_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        self.set(*args, **kwargs)

    def get_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        return self.get(*args, **kwargs)

    def save_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        return self.save(*args, **kwargs)

    def set_grad_output(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        self.set(*args, **kwargs)

    def get_grad_output(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        return self.get(*args, **kwargs)

    def save_grad_output(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        return self.save(*args, **kwargs)

    def stop(self, key: str) -> None:
        self._ensure_in_context("stop")
        handle = self._module.stop(key)
        self._handles.append(handle)


class HookedModule(TensorDictModuleWrapper):
    """
    Wrapper to enhance a module with hooking capabilities.
    """

    def __init__(
        self,
        td_module: TensorDictModule,
        hooking_context: Optional["HookingContext"] = None,
        relative_path: str = "td_module",
    ):
        super().__init__(td_module)
        self._hooking_context = hooking_context
        self._relative_path = relative_path

    @property
    def relative_path(self) -> str:
        return self._relative_path

    def __repr__(self):
        fields = indent(
            f"td_module={self.td_module},\nin_keys={self.in_keys},\nout_keys={self.out_keys}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"

    @property
    def hooking_context(self) -> Optional["HookingContext"]:
        return self._hooking_context

    @classmethod
    def from_module(
        cls,
        module: Callable,
        in_keys: List[str],
        out_keys: List[str],
        *,
        hooking_context: Optional["HookingContext"] = None,
        **kwargs,
    ) -> "HookedModule":
        td_module = TensorDictModule(module, in_keys, out_keys, **kwargs)
        return cls(td_module, hooking_context=hooking_context)

    def run(
        self,
        data: TensorDict,
        cache: Optional[TensorDict] = None,
        run_name: Optional[str] = None,
        run_sep: Optional[str] = None,
        run_cache: Optional[TensorDict] = None,
        grad_enabled: bool = False,
        run_callback: Optional[Callable] = None,
    ) -> HookedModuleRun:
        return HookedModuleRun(self, data, cache, run_name, run_sep, run_cache, grad_enabled, run_callback)

    def register_submodule_hook(
        self,
        key: str,
        hook: Callable,
        direction: HookDirection,
        prepend: bool = False,
        relative: bool = True,
    ):
        root = resolve_submodule_path(self, self._relative_path) if relative else self

        submodule = resolve_submodule_path(root, key)
        if isinstance(submodule, nn.ModuleList):
            warnings.warn(f"You are hooking a ModuleList ({key}), which will never be executed.")
        return register_hook_to_module(submodule, hook, direction, prepend)

    def set(
        self,
        module_key: str,
        value: Any,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
        relative: bool = True,
    ) -> RemovableHandle:
        return self.register_submodule_hook(
            key=module_key,
            hook=HookFactory.make_setting_hook(value, callback=callback, direction=direction),
            direction=direction,
            prepend=prepend,
            relative=relative,
        )

    def get(
        self,
        cache: TensorDict,
        module_key: str,
        cache_key: Optional[str] = None,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
        relative: bool = True,
    ) -> Tuple[RemovableHandle, CacheProxy]:
        cache_key = cache_key or f"{module_key}_{DIRECTION_TO_TYPE[direction]}"
        proxy = CacheProxy(cache_key, cache)
        handle = self.register_submodule_hook(
            key=module_key,
            hook=HookFactory.make_caching_hook(cache_key, cache, callback=callback, direction=direction),
            direction=direction,
            prepend=prepend,
            relative=relative,
        )
        return handle, proxy

    def set_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        return self.set(*args, **kwargs)

    def get_input(self, *args, **kwargs):
        kwargs["direction"] = "fwd_pre"
        return self.get(*args, **kwargs)

    def set_grad(self, *args, **kwargs):
        kwargs["direction"] = "bwd"
        return self.set(*args, **kwargs)

    def get_grad(self, *args, **kwargs):
        kwargs["direction"] = "bwd"
        return self.get(*args, **kwargs)

    def set_grad_output(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        return self.set(*args, **kwargs)

    def get_grad_output(self, *args, **kwargs):
        kwargs["direction"] = "bwd_pre"
        return self.get(*args, **kwargs)

    def stop(self, key: str) -> None:
        return self.register_submodule_hook(
            key=key,
            hook=HookFactory.make_stopping_hook(key),
            direction="fwd",
        )

    def forward(self, *args, **kwargs):
        if self._hooking_context is not None and not self._hooking_context._in_context:
            raise RuntimeError("Contextual HookedModule must be called in context")
        return self.td_module(*args, **kwargs)

    @contextmanager
    def disable_context_hooks(self):
        if self._hooking_context is None:
            raise RuntimeError("No hooking context provided to this module")
        with self._hooking_context.disable_hooks():
            yield

    @contextmanager
    def disable_context(self):
        if self._hooking_context is None:
            raise RuntimeError("No hooking context provided to this module")
        with self._hooking_context.disable() as raw_module:
            yield raw_module

    def restore(self):
        """
        Restore the module to its original state.
        This is useful when using prepare(return_context=False) instead of the context manager.
        """
        if self._hooking_context is None:
            raise RuntimeError("No hooking context provided to this module")
        if not self._hooking_context._in_context:
            raise RuntimeError("Context is not active")
        if self._hooking_context._managed_by_context_manager:
            raise RuntimeError("Cannot call restore() when context is managed by a context manager. ")
        self._hooking_context.__exit__(None, None, None)
