"""
HookedModule
"""

from torch.utils.hooks import RemovableHandle
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper, TensorDictModuleBase
from tensordict import TensorDict
from typing import Callable, Any, Optional, Tuple, TYPE_CHECKING, List
import torch
import warnings
import torch.nn as nn
from contextlib import contextmanager

from tdhook.hooks import (
    register_hook_to_module,
    CacheProxy,
    HookFactory,
    EarlyStoppingException,
    HookDirection,
    DIRECTION_TO_TYPE,
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


def td_grad(
    outputs: TensorDict | Tuple[TensorDict, ...],
    inputs: TensorDict | Tuple[TensorDict, ...],
    grad_outputs: TensorDict | Tuple[TensorDict, ...],
    **kwargs: Any,
) -> TensorDict:
    if isinstance(outputs, tuple) and len(outputs) > 1:
        raise ValueError("torch.autograd.grad for TensorDict only supports a single output")
    elif isinstance(outputs, tuple):
        outputs = outputs[0]
    if not isinstance(outputs, TensorDict):
        raise ValueError("torch.autograd.grad for TensorDict only supports TensorDict as output")
    else:
        tup_outputs = tuple(outputs[k] for k in outputs.keys(True, True))

    if isinstance(inputs, tuple) and len(inputs) > 1:
        raise ValueError("torch.autograd.grad for TensorDict only supports a single input")
    elif isinstance(inputs, tuple):
        inputs = inputs[0]
    if not isinstance(inputs, TensorDict):
        raise ValueError("torch.autograd.grad for TensorDict only supports TensorDict as input")
    else:
        tup_inputs = tuple(inputs[k] for k in inputs.keys(True, True))

    if grad_outputs is not None and isinstance(grad_outputs, tuple) and len(grad_outputs) > 1:
        raise ValueError("torch.autograd.grad for TensorDict only supports a single grad_output")
    elif isinstance(grad_outputs, tuple):
        grad_outputs = grad_outputs[0]
    if not isinstance(grad_outputs, TensorDict):
        raise ValueError("torch.autograd.grad for TensorDict only supports TensorDict as grad_output")
    else:
        tup_grad_outputs = tuple(grad_outputs[k] for k in grad_outputs.keys(True, True))

    tup_grads = torch.autograd.grad(tup_outputs, tup_inputs, tup_grad_outputs, **kwargs)
    return TensorDict(
        dict(zip(inputs.keys(True, True), tup_grads)), batch_size=inputs.batch_size, device=inputs.device
    )


def flatten_reshape_call(module: TensorDictModuleBase, td: TensorDict) -> TensorDict:
    return module(td.flatten()).reshape(td.shape)


def flatten_select_reshape_call(module: TensorDictModuleBase, td: TensorDict) -> TensorDict:
    return module(td.flatten()).select(*module.out_keys).reshape(td.shape)


class FunctionModule(TensorDictModuleBase):
    def __init__(
        self, td_fn: Callable[[TensorDict], TensorDict], in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._td_fn = td_fn

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return self._td_fn(tensordict)

    def __repr__(self):
        return f"FunctionModule(in_keys={self.in_keys}, out_keys={self.out_keys}, td_fn={self._td_fn})"


class HookedModuleRun:
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
    ) -> None:
        self._ensure_in_context("set")
        handle = self._module.set(key, value, callback=callback, direction=direction, prepend=prepend)
        self._handles.append(handle)

    def get(
        self,
        key: str,
        *,
        cache_key: Optional[str] = None,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
        prepend: bool = False,
    ) -> CacheProxy:
        self._ensure_in_context("get")
        handle, proxy = self._module.get(
            self._cache, key, cache_key, callback=callback, direction=direction, prepend=prepend
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
    ) -> CacheProxy:
        self._ensure_in_context("save")
        cache_key = cache_key or f"{self._name + self._sep + key}_{DIRECTION_TO_TYPE[direction]}"
        handle, proxy = self._module.get(
            self._save_cache, key, cache_key=cache_key, callback=callback, direction=direction, prepend=prepend
        )
        self._handles.append(handle)
        return proxy

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
    def __init__(self, td_module: TensorDictModule, hooking_context: Optional["HookingContext"] = None):
        super().__init__(td_module)
        self._hooking_context = hooking_context

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

    def _resolve_submodule_path(self, key: str, relative: bool = True):
        """
        Resolve a submodule path that may contain indexing expressions.

        Supports any valid Python attribute access and indexing:
        - "layers[-1]" -> self.layers[-1]
        - "layers['attr']" -> self.layers['attr']
        - "layers.attention" -> self.layers.attention
        - "layers[1:3]" -> self.layers[1:3]
        """
        root = self.td_module.module if relative else self

        if not key:
            return root

        # Create a safe environment with only the current module
        safe_dict = {"root": root}

        try:
            # Evaluate the expression in the safe environment
            return eval(f"root.{key}", {"__builtins__": {}}, safe_dict)
        except (AttributeError, IndexError, KeyError, SyntaxError) as e:
            raise ValueError(f"Invalid submodule path '{key}': {e}") from e

    def register_submodule_hook(
        self,
        key: str,
        hook: Callable,
        direction: HookDirection,
        prepend: bool = False,
        relative: bool = True,
    ):
        submodule = self._resolve_submodule_path(key, relative)
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
        handle = self.register_submodule_hook(
            key=module_key,
            hook=HookFactory.make_setting_hook(value, callback=callback, direction=direction),
            direction=direction,
            prepend=prepend,
            relative=relative,
        )
        return handle

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
        handle = self.register_submodule_hook(
            key=key,
            hook=HookFactory.make_stopping_hook(key),
            direction="fwd",
        )
        return handle

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
