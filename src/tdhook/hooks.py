import weakref
from typing import Callable, Any, Optional, List, Literal, Protocol, Generic, TypeVar, Type, Tuple
import inspect
from tensordict import TensorDict
from tensordict.utils import NestedKey
import re
from torch.utils.hooks import RemovableHandle
from torch import nn
import torch

from tdhook.paths import resolve_submodule_path as resolve_submodule_path
from tdhook.paths import submodule_path_to_name as submodule_path_to_name


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


def merge_paths(*paths: str) -> str:
    """Merge multiple paths into a single path."""
    return ".".join(path for path in paths if path)


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
    """
    Handle for multiple hooks.
    """

    def __init__(self, handles: Optional[List[RemovableHandleProtocol]] = None):
        self._handles = handles or []

    def remove(self):
        error = None
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()

    def __add__(self, other: Any):
        if not isinstance(other, MultiHookHandle):
            raise TypeError(f"MultiHookHandle cannot be added to {type(other).__name__}")
        return MultiHookHandle(self._handles + other._handles)


class MultiHookManager:
    """
    Manager for multiple hooks.
    """

    def __init__(
        self,
        pattern: Optional[str] = None,
        classes_to_hook: Tuple[Type[nn.Module], ...] = (nn.Module,),
        classes_to_skip: Tuple[Type[nn.Module], ...] = (),
    ):
        if pattern is None:
            pattern = r"a^"  # match nothing by default
        self._pattern = pattern
        self._classes_to_hook = classes_to_hook
        self._classes_to_skip = classes_to_skip
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
        relative_path: Optional[str] = None,
    ):
        """Register the hook to the module."""
        handles = []
        for name, submodule in self.iter_modules(module, relative_path=relative_path):
            handles.append(register_hook_to_module(submodule, hook_factory(name), direction, prepend))
        return MultiHookHandle(handles)

    def iter_modules(
        self,
        module: nn.Module,
        *,
        relative_path: Optional[str] = None,
    ):
        """Yield matching executable submodules in registration order."""
        root_module = resolve_submodule_path(module, relative_path) if relative_path else module
        for name, submodule in root_module.named_modules():
            if name == "":
                continue
            if not isinstance(submodule, self._classes_to_hook) or isinstance(submodule, self._classes_to_skip):
                continue
            if self._reg_exp.match(name):
                yield name, submodule


class MutableWeakRef(Generic[T]):
    """
    Weak reference to a mutable object.
    """

    def __init__(self, referee: T):
        self._ref = weakref.ref(referee)

    def resolve(self) -> T:
        return self._ref()

    def set(self, referee: T):
        self._ref = weakref.ref(referee)


class TensorDictRef:
    """
    Reference to a TensorDict.
    """

    def __init__(self, td: Optional[TensorDict]):
        self._td = td

    def resolve(self) -> TensorDict:
        return self._td

    def set(self, td: TensorDict):
        self._td = td


class EarlyStoppingException(BaseException):
    """
    Internal control-flow signal for early stopping.

    This intentionally bypasses ordinary ``except Exception`` handlers in
    model code so they cannot accidentally resume a stopped forward pass.
    """

    def __init__(self, key: str):
        self._key = key
        super().__init__(f"Early stopping triggered for key {key}")


class HookFactory:
    """
    Factory for creating hooks.
    """

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
        key: NestedKey,
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
            if _cache.is_locked:
                if key not in _cache.keys(include_nested=True):
                    raise RuntimeError(
                        f"Locked caches require a preallocated entry for {key!r}; "
                        "create every cache key before locking or memory-mapping the TensorDict"
                    )
                _cache.set_(key, value)
            else:
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
            _value = value
            if callback is not None:
                _value = callback(**dict(zip(params, args)), value=_value, direction=direction)
            if _value is not None and type(_value) is not original_type:
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
