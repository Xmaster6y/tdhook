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


HookDirection = Literal["fwd", "bwd", "fwd_pre", "bwd_pre", "fwd_kwargs", "fwd_pre_kwargs"]

DIRECTION_TO_PARAMS = {
    "fwd": ("module", "args", "output"),
    "bwd": ("module", "grad_input", "grad_output"),
    "fwd_pre": ("module", "args"),
    "bwd_pre": ("module", "grad_input"),
    "fwd_kwargs": ("module", "args", "kwargs", "output"),
    "fwd_pre_kwargs": ("module", "args", "kwargs"),
}

DIRECTION_TO_TYPE = {
    "fwd": "output",
    "bwd": "grad",
    "fwd_pre": "input",
    "bwd_pre": "grad_input",
    "fwd_kwargs": "output",
    "fwd_pre_kwargs": "input",
}


def _check_hook_signature(hook: Callable, direction: HookDirection):
    """Check the signature of the hook."""
    if direction not in DIRECTION_TO_PARAMS:
        raise ValueError(f"Invalid direction: {direction}")

    param_len = len(inspect.signature(hook).parameters)
    expected_params = DIRECTION_TO_PARAMS[direction]

    if param_len != len(expected_params):
        raise ValueError(f"Hook ({direction}) must have the signature {expected_params}")


def register_hook_to_module(
    module: nn.Module,
    hook: Callable,
    direction: HookDirection,
    prepend: bool = False,
) -> RemovableHandle:
    """Register the hook to the module."""
    _check_hook_signature(hook, direction)
    if direction in ["fwd", "fwd_kwargs"]:
        return module.register_forward_hook(hook, prepend=prepend, with_kwargs=direction == "fwd_kwargs")
    elif direction == "bwd":
        return module.register_full_backward_hook(hook, prepend=prepend)
    elif direction in ["fwd_pre", "fwd_pre_kwargs"]:
        return module.register_forward_pre_hook(hook, prepend=prepend, with_kwargs=direction == "fwd_pre_kwargs")
    else:
        return module.register_full_backward_pre_hook(hook, prepend=prepend)


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
        hook_factory: Callable[[str], Callable],
        *,
        direction: HookDirection = "fwd",
        prepend: bool = False,
    ):
        """Register the hook to the module."""
        handles = []
        for name, module in module.named_modules():
            if self._reg_exp.match(name):
                handles.append(register_hook_to_module(module, hook_factory(name), direction, prepend))
        return MultiHookHandle(handles)


class CacheProxy:
    def __init__(self, key: str, cache: TensorDict):
        self._key = key
        self._cache = weakref.ref(cache)

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
        key: str, cache: TensorDict, *, callback: Optional[Callable] = None, direction: HookDirection = "fwd"
    ) -> Callable:
        """
        Make a caching hook.
        """

        if direction not in DIRECTION_TO_PARAMS:
            raise ValueError(f"Invalid direction: {direction}")

        params = DIRECTION_TO_PARAMS[direction]
        value_index = -2 if direction == "fwd_pre_kwargs" else -1
        HookFactory._check_callback_signature(callback, set(params))

        def hook(*args):
            nonlocal key, cache, callback
            if callback is not None:
                value = callback(**dict(zip(params, args)))
            else:
                value = args[value_index]
            cache[key] = value

        return hook

    @staticmethod
    def make_setting_hook(
        value: Any, *, callback: Optional[Callable] = None, direction: HookDirection = "fwd"
    ) -> Callable:
        """
        Make a setting hook.
        """

        if direction not in DIRECTION_TO_PARAMS:
            raise ValueError(f"Invalid direction: {direction}")

        params = DIRECTION_TO_PARAMS[direction]
        HookFactory._check_callback_signature(callback, set(params))

        def hook(*args):
            nonlocal value, callback
            if isinstance(value, CacheProxy):
                value = value.resolve()
            if callback is not None:
                value = callback(**dict(zip(params, args)))
            return value

        return hook

    @staticmethod
    def make_stopping_hook(key: str) -> Callable:
        def hook(module, args, output):
            nonlocal key
            raise EarlyStoppingException(key)

        return hook
