"""
Hooks
"""

import weakref
from typing import Callable, Any, Optional, List, Literal, Protocol
import inspect

from tensordict import TensorDict
import re
from torch.utils.hooks import RemovableHandle
from torch import nn
import torch


HookDirection = Literal["fwd", "bwd", "fwd_pre", "bwd_pre", "fwd_kwargs", "fwd_pre_kwargs"]

DIRECTION_TO_PARAMS = {
    "fwd": ("module", "args", "output"),
    "bwd": ("module", "grad_input", "grad_output"),
    "fwd_pre": ("module", "args"),
    "bwd_pre": ("module", "grad_output"),
    "fwd_kwargs": ("module", "args", "kwargs", "output"),
    "fwd_pre_kwargs": ("module", "args", "kwargs"),
}

DIRECTION_TO_RETURN = {
    "fwd": "output",
    "bwd": "grad_input",
    "fwd_pre": "args",
    "bwd_pre": "grad_output",
    "fwd_kwargs": "output",
    "fwd_pre_kwargs": "args",
}

DIRECTION_TO_RETURN_INDEX = {k: v.index(DIRECTION_TO_RETURN[k]) for k, v in DIRECTION_TO_PARAMS.items()}

DIRECTION_TO_TYPE = {
    "fwd": "output",
    "bwd": "grad_input",
    "fwd_pre": "input",
    "bwd_pre": "grad_output",
    "fwd_kwargs": "output",
    "fwd_pre_kwargs": "input",
}


def _check_hook_signature(hook: Callable, direction: HookDirection):
    """Check the signature of the hook."""
    if direction not in DIRECTION_TO_PARAMS:
        raise ValueError(f"Invalid direction: {direction}")

    sig = inspect.signature(hook)
    param_len = len(sig.parameters)
    expected_params = DIRECTION_TO_PARAMS[direction]

    has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in sig.parameters.values())

    num_optional_params = sum(
        1
        for param in sig.parameters.values()
        if param.default is not inspect.Parameter.empty or param.kind == inspect.Parameter.VAR_KEYWORD
    )

    if has_varargs:
        if param_len > len(expected_params) + 1 + num_optional_params:
            raise ValueError(
                f"Hook ({direction}) must have at most {len(expected_params) + 1 + num_optional_params} positional parameters"
            )
        return

    if param_len != len(expected_params) + num_optional_params:
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


class RemovableHandleProtocol(Protocol):
    def remove(self): ...


class MultiHookHandle:
    def __init__(self, handles: Optional[List[RemovableHandleProtocol]] = None):
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
            raise TypeError(f"MultiHookHandle cannot be added to {type(other).__name__}")
        return MultiHookHandle(self._handles + other._handles)


class MultiHookManager:
    def __init__(self, pattern: Optional[str] = None, relative: bool = False):
        if pattern is None:
            pattern = r"a^"  # match nothing by default
        self._pattern = pattern
        self._reg_exp = re.compile(pattern)
        self._relative = relative

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
            if self._relative:
                if name.startswith("td_module"):
                    name = name[len("td_module") :].lstrip(".")
                if name.startswith("module"):
                    name = name[len("module") :].lstrip(".")
                if name == "":
                    continue
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
                value = callback(**dict(zip(params, args)), key=key)
            else:
                value = args[value_index]
            if not isinstance(value, torch.Tensor) and not isinstance(value, TensorDict):
                raise RuntimeError(
                    f"{type(value).__name__} values are not supported for caching, use a `callback` to return a tensor or a tensordict"
                )
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
        return_index = DIRECTION_TO_RETURN_INDEX[direction]
        HookFactory._check_callback_signature(callback, set(params))

        def hook(*args):
            nonlocal value, callback, params, return_index
            original_type = type(args[return_index])
            if isinstance(value, CacheProxy):
                value = value.resolve()
            if callback is not None:
                value = callback(**dict(zip(params, args)), value=value)
            if type(value) is not original_type:
                raise RuntimeError(
                    f"Callback returned a value of type {type(value).__name__} but the original value was of type {original_type.__name__}"
                )
            return value

        return hook

    @staticmethod
    def make_reading_hook(*, callback: Callable, direction: HookDirection = "fwd") -> Callable:
        """
        Make a reading hook.
        """

        if direction not in DIRECTION_TO_PARAMS:
            raise ValueError(f"Invalid direction: {direction}")

        params = DIRECTION_TO_PARAMS[direction]
        HookFactory._check_callback_signature(callback, set(params))

        def hook(*args):
            nonlocal callback, params
            callback(**dict(zip(params, args)))

        return hook

    @staticmethod
    def make_stopping_hook(key: str) -> Callable:
        def hook(module, args, output):
            nonlocal key
            raise EarlyStoppingException(key)

        return hook
