"""
MultiHook
"""

import weakref
from typing import Callable, Any, Optional, List, Literal
import inspect

from tensordict import TensorDict
import re
from torch.utils.hooks import RemovableHandle
from torch import nn


def _check_hook_signature(hook: Callable, direction: Literal["fwd", "bwd", "fwd_pre", "bwd_pre"], with_kwargs: bool):
    """Check the signature of the hook."""
    param_len = len(inspect.signature(hook).parameters)
    if with_kwargs:
        fwd_offset = 1
        signature = ", kwargs"
    else:
        fwd_offset = 0
        signature = ""
    if direction == "fwd" and param_len != 3 + fwd_offset:
        raise ValueError(f"Forward hooks must have the signature (module, args{signature}, output)")
    elif direction == "bwd" and param_len != 3:
        raise ValueError("Backward hooks must have the signature (module, grad_input, grad_output)")
    elif direction == "fwd_pre" and param_len != 2 + fwd_offset:
        raise ValueError(f"Forward pre-hooks must have the signature (module, args{signature})")
    elif direction == "bwd_pre" and param_len != 2:
        raise ValueError("Backward pre-hooks must have the signature (module, grad_input)")


def register_hook_to_module(
    module: nn.Module,
    hook: Callable,
    direction: Literal["fwd", "bwd", "fwd_pre", "bwd_pre"],
    prepend: bool = False,
    with_kwargs: bool = False,
) -> RemovableHandle:
    """Register the hook to the module."""
    _check_hook_signature(hook, direction, with_kwargs)
    if direction == "fwd":
        return module.register_forward_hook(hook, prepend=prepend, with_kwargs=with_kwargs)
    elif direction == "bwd":
        return module.register_full_backward_hook(hook, prepend=prepend)
    elif direction == "fwd_pre":
        return module.register_forward_pre_hook(hook, prepend=prepend, with_kwargs=with_kwargs)
    elif direction == "bwd_pre":
        return module.register_full_backward_pre_hook(hook, prepend=prepend)
    else:
        raise ValueError(f"Invalid direction: {direction}")


class MultiHookHandle:
    def __init__(self, handles: Optional[List[RemovableHandle]] = None):
        self._handles = handles or []

    def remove(self):
        for handle in self._handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()

    def __add__(self, other: Any):
        if not isinstance(other, MultiHookHandle):
            raise TypeError(f"MultiHookHandle cannot be added to {type(other)}")
        return MultiHookHandle(self._handles + other._handles)


class MultiHookManager:
    def __init__(self, pattern: Optional[str] = None):
        if pattern is None:
            pattern = r"a^"  # match nothing by default
        self._pattern = pattern
        self._reg_exp = re.compile(pattern)

    @property
    def pattern(self) -> str:
        """The pattern to match the modules."""
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: str):
        self._pattern = pattern
        self._reg_exp = re.compile(pattern)

    def register_hook(
        self,
        module: nn.Module,
        hook: Callable,
        direction: Literal["fwd", "bwd", "fwd_pre", "bwd_pre"],
        prepend: bool = False,
        with_kwargs: bool = False,
    ):
        """Register the hook to the module."""
        handles = []
        for name, module in module.named_modules():
            if self._reg_exp.match(name):
                handles.append(register_hook_to_module(module, hook, direction, prepend, with_kwargs))
        return MultiHookHandle(handles)


class CacheProxy:
    def __init__(self, key: str, cache: TensorDict, sep: str = "."):
        self._key = key
        self._cache = weakref.ref(cache)
        self._sep = sep

    def resolve(self) -> Any:
        cache = self._cache()
        if cache is None:
            raise ValueError("Dead reference to cache")
        value = cache.get(self._key)
        if value is None:
            raise ValueError(f"Key {self._key} not found in cache")
        return value


class EarlyStoppingException(Exception):
    def __init__(self, key: str):
        self._key = key
        super().__init__(f"Early stopping triggered for key {key}")


class HookFactory:
    @staticmethod
    def _check_callback_signature(callback: Callable, expected_param_names: set[str]):
        """Check callback signature matches expected parameter names."""
        if callback is None:
            return
        sig = inspect.signature(callback)
        param_names = set(sig.parameters.keys())

        has_positional_only = any(param.kind == inspect.Parameter.POSITIONAL_ONLY for param in sig.parameters.values())
        if has_positional_only:
            raise ValueError("Callback cannot have positional-only parameters since we only pass named arguments")

        has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        if has_kwargs:
            return

        missing_params = expected_param_names - param_names
        if missing_params:
            raise ValueError(f"Callback missing required parameters: {missing_params}")

    @staticmethod
    def make_caching_hook(
        key: str, cache: TensorDict, sep: str = ".", callback: Optional[Callable] = None
    ) -> Callable:
        HookFactory._check_callback_signature(callback, {"output", "module", "args"})

        def hook(module, args, output):
            nonlocal key, cache, sep, callback
            if callback is not None:
                output = callback(output=output, module=module, args=args)
            cache[key] = output

        return hook

    @staticmethod
    def make_setting_hook(value: Any, callback: Optional[Callable] = None) -> Callable:
        HookFactory._check_callback_signature(callback, {"value", "module", "args", "output"})

        def hook(module, args, output):
            nonlocal value, callback
            if isinstance(value, CacheProxy):
                value = value.resolve()
            if callback is not None:
                value = callback(value=value, module=module, args=args, output=output)
            return value

        return hook

    @staticmethod
    def make_stopping_hook(key: str) -> Callable:
        def hook(module, args, output):
            nonlocal key
            raise EarlyStoppingException(key)

        return hook
