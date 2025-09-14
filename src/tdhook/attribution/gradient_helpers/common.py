"""
Gradient attribution
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Tuple, List, Dict

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory
from tdhook.modules import FunctionModule, flatten_select_reshape_call, IntermediateKeysCleaner, ModuleCallWithCache
from tdhook._types import UnraveledKey
from tdhook.modules import HookedModule, td_grad
from tdhook.hooks import MultiHookHandle, MutableWeakRef, TensorDictRef


class GradientAttribution(HookingContextFactory, metaclass=ABCMeta):
    def __init__(
        self,
        use_inputs: bool = True,
        use_outputs: bool = True,
        input_modules: Optional[List[str]] = None,
        target_modules: Optional[List[str]] = None,
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_inputs: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        output_grad_callbacks: Optional[Dict[str, Callable]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_intermediate_keys: bool = True,
        cache_callback: Optional[Callable] = None,
    ):
        super().__init__()

        self._use_inputs = use_inputs
        self._use_outputs = use_outputs
        self._input_modules = input_modules or []
        self._target_modules = target_modules or []
        self._init_attr_targets = init_attr_targets
        self._init_attr_inputs = init_attr_inputs
        self._init_attr_grads = init_attr_grads
        self._output_grad_callbacks = output_grad_callbacks or {}
        self._cache_callback = cache_callback

        self._additional_init_keys = additional_init_keys or []
        self._attr_key = attribution_key
        self._clean_intermediate_keys = clean_intermediate_keys
        self._hooked_module_kwargs["relative_path"] = "td_module.module[2]._td_module.module"

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        register_in_keys = [("_register_in", in_key) for in_key in in_keys]
        mod_in_keys = [("_mod_in", in_key) for in_key in in_keys]
        cache_in_keys = [("_cache_in", in_key) for in_key in self._input_modules]
        mod_out_keys = [("_mod_out", out_key) for out_key in out_keys]
        cache_out_keys = [("_cache_out", out_key) for out_key in self._target_modules]
        attr_keys = [(self._attr_key, in_key) for in_key in in_keys]

        if set(self._additional_init_keys) & (set(in_keys) | set(out_keys)):
            raise ValueError("Additional init keys must not be in the in_keys or out_keys")

        cache_ref = TensorDictRef(TensorDict())
        modules = [
            TensorDictModule(
                lambda *tensors: tensors,
                in_keys=in_keys,
                out_keys=register_in_keys,
            ),
            FunctionModule(
                self._register_inputs_fn,
                in_keys=register_in_keys,
                out_keys=mod_in_keys,
            ),
            ModuleCallWithCache(
                module,
                in_key="_mod_in",
                out_key="_mod_out",
                stored_keys=cache_in_keys + cache_out_keys,
                cache_ref=cache_ref,
                cache_as_output=False,
            ),
            FunctionModule(
                lambda td: self._attributor_fn(td, cache_ref),
                in_keys=(mod_in_keys if self._use_inputs else [])
                + (mod_out_keys if self._use_outputs else [])
                + self._additional_init_keys,
                out_keys=attr_keys,
            ),
        ]
        if self._clean_intermediate_keys:
            modules.append(
                IntermediateKeysCleaner(
                    intermediate_keys=["_register_in", "_mod_in", "_mod_out", "_cache_in", "_cache_out"]
                )
            )
        return TensorDictSequential(*modules)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        cache_ref = module.td_module[2].cache_ref
        handles = []
        for module_key in self._input_modules:

            def callback(**kwargs):
                nonlocal module_key, self
                if self._cache_callback is not None:
                    output = self._cache_callback(**kwargs)
                else:
                    output = kwargs["output"]
                return output.requires_grad_(True)

            handle, _ = module.get(  # TODO: replace by a read
                cache=cache_ref,
                cache_key=("_cache_in", module_key),
                module_key=module_key,
                callback=callback,
            )
            handles.append(handle)
        for module_key in self._target_modules:
            handle, _ = module.get(
                cache=cache_ref,
                cache_key=("_cache_out", module_key),
                module_key=module_key,
                callback=self._cache_callback,
            )
            handles.append(handle)
        for module_key, callback in self._output_grad_callbacks.items():
            handle = module.set_grad_output(module_key, value=None, callback=callback)
            handles.append(handle)
        return MultiHookHandle(handles)

    def _register_inputs_fn(self, td: TensorDict) -> TensorDict:
        inputs = td["_register_in"]
        if self._init_attr_inputs is not None:
            needs_grad = self._init_attr_inputs(inputs, td.select(*self._additional_init_keys))
            if not isinstance(inputs, TensorDict):
                raise ValueError("init_attr_inputs function must return a TensorDict")
        elif self._use_inputs:
            needs_grad = inputs
        else:
            needs_grad = TensorDict()
        needs_grad.requires_grad_(True)
        inputs.update(needs_grad)
        td["_mod_in"] = inputs
        return td

    def _attributor_fn(self, td: TensorDict, cache_ref: MutableWeakRef | TensorDictRef) -> TensorDict:
        additional_init_tensors = td.select(*self._additional_init_keys)
        cache = cache_ref.resolve()

        inputs = td["_mod_in"] if self._use_inputs else TensorDict()
        if self._init_attr_inputs is not None and self._use_inputs:
            inputs = self._init_attr_inputs(inputs, additional_init_tensors)
            if not isinstance(inputs, TensorDict):
                raise ValueError("init_attr_inputs function must return a TensorDict")
        cache_in = cache["_cache_in"] if self._input_modules else TensorDict()

        targets = td["_mod_out"] if self._use_outputs else TensorDict()
        targets.update(cache["_cache_out"].reshape(cache["_shape"]) if self._target_modules else {})

        if self._init_attr_targets is not None:
            targets = self._init_attr_targets(targets, additional_init_tensors)
            if not isinstance(targets, TensorDict):
                raise ValueError("init_attr_targets function must return a TensorDict")

        if self._init_attr_grads is not None:
            init_grads = self._init_attr_grads(targets, additional_init_tensors)
            if not isinstance(init_grads, TensorDict):
                raise ValueError("init_attr_grads function must return a TensorDict")
        else:
            init_grads = torch.ones_like(targets)

        if init_grads.batch_size != targets.batch_size:
            raise ValueError("init_grads should have the same batch size as targets")

        if set(targets.keys(True, True)) != set(init_grads.keys(True, True)):
            raise ValueError("Targets and init_grads must have the same keys")

        for target_key, target in targets.items():
            if target.grad_fn is None:
                raise ValueError(f"Target {target_key} has no grad_fn")

        _grads = td_grad(targets, TensorDict(inputs=inputs, cache_in=cache_in), init_grads)
        if self._use_inputs:
            grads = _grads["inputs"]
            grads.batch_size = inputs.batch_size
        else:
            grads = TensorDict(batch_size=cache["_shape"])
        if self._input_modules:
            cache_in_grads = _grads["cache_in"]
            cache_in_grads.batch_size = cache_in.batch_size
            grads.update(cache_in_grads.reshape(cache["_shape"]))
            inputs.update(cache_in.reshape(cache["_shape"]))
        attrs = self._grad_attr(grads, inputs)
        td[self._attr_key] = attrs
        return td

    @abstractmethod
    def _grad_attr(
        self,
        grads: TensorDict,
        inputs: TensorDict,
    ) -> TensorDict:
        pass


class GradientAttributionWithBaseline(GradientAttribution):
    def __init__(
        self,
        *args,
        compute_convergence_delta: bool = False,
        baseline_key: UnraveledKey = "baseline",
        multiply_by_inputs: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._compute_convergence_delta = compute_convergence_delta
        self._baseline_key = baseline_key
        self._multiply_by_inputs = multiply_by_inputs

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        n_in_keys = len(in_keys)
        register_in_keys = [("_register_in", in_key) for in_key in in_keys]
        attr_keys = [(self._attr_key, in_key) for in_key in in_keys]
        baseline_keys = [(self._baseline_key, in_key) for in_key in in_keys]
        (_, register_inputs, module_call, attributor, *_) = super()._prepare_module(module, in_keys, out_keys)

        modules = [
            FunctionModule(
                lambda td: self._reduce_baselines_fn(td, in_keys),
                in_keys=in_keys + baseline_keys,
                out_keys=register_in_keys,
            ),
            register_inputs,
            module_call,
            attributor,
        ]
        if self._multiply_by_inputs:
            modules.append(
                TensorDictModule(
                    lambda *tensors: self._multiply_by_inputs_fn(
                        tensors[:n_in_keys], tensors[n_in_keys : n_in_keys * 2], tensors[n_in_keys * 2 :]
                    ),
                    in_keys=in_keys + baseline_keys + attr_keys,
                    out_keys=attr_keys,
                    inplace=True,
                )
            )
        if self._compute_convergence_delta:
            modules.append(
                FunctionModule(
                    lambda td: self._compute_convergence_delta_fn(td, in_keys, out_keys, module),
                    in_keys=in_keys + baseline_keys + attr_keys + self._additional_init_keys,
                    out_keys=["convergence_delta"],
                )
            )

        return TensorDictSequential(*modules)

    @abstractmethod
    def _reduce_baselines_fn(self, td: TensorDict, in_keys: List[UnraveledKey]) -> TensorDict:
        pass

    @torch.no_grad()
    def _multiply_by_inputs_fn(  # TODO: is it faster with TensorDict?
        self, inputs: Tuple[torch.Tensor, ...], baselines: Tuple[torch.Tensor, ...], attrs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(attr * (inp - baseline) for attr, inp, baseline in zip(attrs, inputs, baselines))

    @torch.no_grad()
    def _compute_convergence_delta_fn(
        self,
        td: TensorDict,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        module: TensorDictModuleBase,
    ) -> TensorDict:
        inputs = td.select(*in_keys)
        baselines = td[self._baseline_key]
        attrs = td[self._attr_key]
        additional_init_tensors = td.select(*self._additional_init_keys)

        if self._init_attr_targets is not None:
            start_out = self._init_attr_targets(
                flatten_select_reshape_call(module, baselines), additional_init_tensors
            )
        else:
            start_out = flatten_select_reshape_call(module, baselines)
        start_out_sum = start_out.sum(dim="feature", reduce=True)
        if self._init_attr_targets is not None:
            end_out = self._init_attr_targets(flatten_select_reshape_call(module, inputs), additional_init_tensors)
        else:
            end_out = flatten_select_reshape_call(module, inputs)
        end_out_sum = end_out.sum(dim="feature", reduce=True)

        attr_sum = attrs.sum(dim="feature", reduce=True)
        td["convergence_delta"] = attr_sum - (end_out_sum - start_out_sum)
        return td
