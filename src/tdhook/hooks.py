"""
Hooks
"""

import weakref
from typing import Callable, Any, Optional, List, Literal, Protocol, Generic, TypeVar, Type, Tuple
import inspect
from tensordict import TensorDict
import re
from torch.utils.hooks import RemovableHandle
from torch import nn
import torch

from tdhook._types import UnraveledKey


HookDirection = Literal["fwd", "bwd", "fwd_pre", "bwd_pre", "fwd_kwargs", "fwd_pre_kwargs"]
T = TypeVar("T")

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


def resolve_submodule_path(root: nn.Module, key: str):
    """
    Resolve a submodule path that may contain indexing expressions.

    Supports any valid Python attribute access and indexing:
    - "[0]" -> root[0]
    - "layers[-1]" -> root.layers[-1]
    - "layers['attr']" -> root.layers['attr']
    - "layers.attention" -> root.layers.attention
    - "layers[1:3]" -> root.layers[1:3]

    Supports custom attributes:
    - ":block0/module:" -> getattr(root, "block0/module")
    - ":block0/module:layers.attention[0]" -> getattr(root, "block0/module").layers.attention[0]
    - "m1:block0/module:layers:module:linear[0]" -> getattr(getattr(root.m1, "block0/module").layers, "module").linear[0]
    """

    if not key:
        return root

    start_key, *rest = key.split(":", maxsplit=1)

    if rest:
        start_root = resolve_submodule_path(root, start_key)
        attr, *rest = rest[0].split(":", maxsplit=1)
        if not rest:
            raise ValueError(f"Invalid submodule path '{key}', missing closing ':'")
        return resolve_submodule_path(getattr(start_root, attr), rest[0])

    # Create a safe environment with only the current module
    safe_dict = {"root": root}

    try:
        if key.startswith("["):
            return eval(f"root{key}", {"__builtins__": {}}, safe_dict)
        else:
            return eval(f"root.{key}", {"__builtins__": {}}, safe_dict)
    except (AttributeError, IndexError, KeyError, SyntaxError) as e:
        raise ValueError(f"Invalid submodule path '{key}': {e}") from e


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
    def __init__(
        self,
        pattern: Optional[str] = None,
        module_classes: Tuple[Type[nn.Module], ...] = (nn.Module,),
        relative: bool = False,
    ):
        if pattern is None:
            pattern = r"a^"  # match nothing by default
        self._pattern = pattern
        self._module_classes = module_classes
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
            if not isinstance(module, self._module_classes):
                continue
            if self._reg_exp.match(name):
                handles.append(register_hook_to_module(module, hook_factory(name), direction, prepend))
        return MultiHookHandle(handles)


class MutableWeakRef(Generic[T]):
    def __init__(self, referee: T):
        self._ref = weakref.ref(referee)

    def resolve(self) -> T:
        return self._ref()

    def set(self, referee: T):
        self._ref = weakref.ref(referee)


class TensorDictRef:
    def __init__(self, td: Optional[TensorDict]):
        self._td = td

    def resolve(self) -> TensorDict:
        return self._td

    def set(self, td: TensorDict):
        self._td = td


class CacheProxy:
    def __init__(self, key: str, cache: TensorDict | MutableWeakRef[TensorDict] | TensorDictRef):
        self._key = key
        self._cache = weakref.ref(cache)

    def resolve(self) -> Any:
        cache = self._cache()
        if isinstance(cache, (MutableWeakRef, TensorDictRef)):
            cache = cache.resolve()
        if cache is None:
            raise ValueError("Dead reference to cache")
        return cache.get(self._key)


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
        key: UnraveledKey,
        cache: TensorDict | MutableWeakRef,
        *,
        callback: Optional[Callable] = None,
        direction: HookDirection = "fwd",
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
            nonlocal key, cache, callback, direction
            if callback is not None:
                value = callback(**dict(zip(params, args)), key=key, direction=direction)
            else:
                value = args[value_index]
            if not isinstance(value, torch.Tensor) and not isinstance(value, TensorDict):
                raise RuntimeError(
                    f"{type(value).__name__} values are not supported for caching, use a `callback` to return a tensor or a tensordict"
                )
            if isinstance(cache, MutableWeakRef | TensorDictRef):
                _cache = cache.resolve()
                if _cache is None:
                    raise ValueError("Dead reference to cache")
            else:
                _cache = cache
            _cache[key] = value

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
            nonlocal value, callback, params, return_index, direction
            original_type = type(args[return_index])
            _value = value.resolve() if isinstance(value, CacheProxy) else value
            if callback is not None:
                _value = callback(**dict(zip(params, args)), value=_value, direction=direction)
            if type(_value) is not original_type:
                raise RuntimeError(
                    f"Callback returned a value of type {type(_value).__name__} but the original value was of type {original_type.__name__}"
                )
            return _value

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
            nonlocal callback, params, direction
            callback(**dict(zip(params, args)), direction=direction)

        return hook

    @staticmethod
    def make_stopping_hook(key: str) -> Callable:
        def hook(module, args, output):
            nonlocal key
            raise EarlyStoppingException(key)

        return hook
