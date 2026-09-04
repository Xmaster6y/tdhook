from tensordict.nn import TensorDictModuleWrapper, TensorDictModuleBase, TensorDictSequential
from tensordict import NonTensorData, TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from typing import Callable, Optional, TYPE_CHECKING, List
import torch
from textwrap import indent

from tdhook.hooks import (
    MutableWeakRef,
    TensorDictRef,
)

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

    def __init__(self, td_fn: Callable[[TensorDict], TensorDict], in_keys: List[NestedKey], out_keys: List[NestedKey]):
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
        in_key: Optional[NestedKey] = None,
        out_key: Optional[NestedKey] = None,
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
        stored_keys: List[NestedKey],
        cache_key: Optional[NestedKey] = None,
        in_key: Optional[NestedKey] = None,
        out_key: Optional[NestedKey] = None,
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


class _CacheRefSequential(TensorDictSequential):
    """Internal sequential wrapper carrying a cache reference for hook callbacks."""

    def __init__(self, *modules, cache_ref: MutableWeakRef | TensorDictRef):
        super().__init__(*modules)
        self.cache_ref = cache_ref


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
        grad_key: NestedKey = "_grad",
        working_key: NestedKey = "_working",
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

    def __init__(self, intermediate_keys: List[NestedKey]):
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


class HookedModule(TensorDictModuleWrapper):
    """Internal execution wrapper owned by :class:`HookingContext`."""

    def __init__(
        self,
        td_module: TensorDictModuleBase,
        hook_root: TensorDictModuleBase,
        hooking_context: Optional["HookingContext"] = None,
        relative_path: str = "",
    ):
        super().__init__(td_module)
        self._hook_root = hook_root
        self._hooking_context = hooking_context
        self._relative_path = relative_path

    @property
    def hook_root(self) -> TensorDictModuleBase:
        """Return the caller-owned module against which hook paths resolve."""

        return self._hook_root

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

    def finalize_tensordict(self, data: TensorDictBase) -> TensorDictBase:
        """Publish method-owned products after a shared model execution."""

        return data

    def forward(self, *args, **kwargs):
        if self._hooking_context is not None and not self._hooking_context._in_context:
            raise RuntimeError("Contextual HookedModule must be called in context")
        result = self.td_module(*args, **kwargs)
        return self.finalize_tensordict(result) if isinstance(result, TensorDictBase) else result
