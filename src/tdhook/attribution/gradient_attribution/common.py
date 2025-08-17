"""
Gradient attribution
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Tuple, List

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory
from tdhook.module import FunctionModule, flatten_select_reshape_call
from tdhook._types import UnraveledKey


class GradientAttribution(HookingContextFactory, metaclass=ABCMeta):
    def __init__(
        self,
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_inputs: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        multiply_by_inputs: bool = False,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_keys: bool = True,
        keep_output_keys: bool = True,
    ):
        self._init_attr_targets = init_attr_targets
        self._init_attr_inputs = init_attr_inputs
        self._init_attr_grads = init_attr_grads

        self._multiply_by_inputs = multiply_by_inputs
        self._additional_init_keys = additional_init_keys or []
        self._attr_key = attribution_key
        self._clean_keys = clean_keys  # TODO: clean intermediate keys
        self._keep_output_keys = keep_output_keys  # TODO: move output keys before cleaning

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        n_in_keys = len(in_keys)
        register_in_keys = [("_register_in", in_key) for in_key in in_keys]
        mod_in_keys = [("_mod_in", in_key) for in_key in in_keys]
        mod_out_keys = [("_mod_out", out_key) for out_key in out_keys]
        attr_keys = [(self._attr_key, in_key) for in_key in in_keys]

        if set(self._additional_init_keys) & (set(in_keys) | set(out_keys)):
            raise ValueError("Additional init keys must not be in the in_keys or out_keys")
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
            FunctionModule(
                lambda td: self._module_call_fn(td, module),
                in_keys=mod_in_keys,
                out_keys=mod_out_keys,
            ),
            FunctionModule(
                self._attributor_fn,
                in_keys=mod_in_keys + mod_out_keys + self._additional_init_keys,
                out_keys=attr_keys,
            ),
        ]
        if self._multiply_by_inputs:
            modules.append(
                TensorDictModule(
                    lambda *tensors: self._multiply_by_inputs_fn(tensors[:n_in_keys], tensors[n_in_keys:]),
                    in_keys=mod_in_keys + attr_keys,
                    out_keys=attr_keys,
                    inplace=True,
                )
            )
        return TensorDictSequential(*modules)

    def _register_inputs_fn(self, td: TensorDict) -> TensorDict:
        inputs = td["_register_in"]
        if self._init_attr_inputs is not None:
            needs_grad = self._init_attr_inputs(inputs, td.select(*self._additional_init_keys))
            if not isinstance(inputs, TensorDict):
                raise ValueError("init_attr_inputs function must return a TensorDict")
        else:
            needs_grad = inputs
        needs_grad.requires_grad_(True)
        inputs.update(needs_grad)
        td["_mod_in"] = inputs
        return td

    def _module_call_fn(self, td: TensorDict, module: TensorDictModuleBase) -> TensorDict:
        inputs = td["_mod_in"]
        outputs = flatten_select_reshape_call(module, inputs)
        td["_mod_out"] = outputs
        return td

    def _attributor_fn(self, td: TensorDict) -> TensorDict:
        additional_init_tensors = td.select(*self._additional_init_keys)

        if self._init_attr_targets is not None:
            targets = self._init_attr_targets(td["_mod_out"], additional_init_tensors)
            if not isinstance(targets, TensorDict):
                raise ValueError("init_attr_targets function must return a TensorDict")
        else:
            targets = td["_mod_out"]
        if targets.batch_size != td["_mod_in"].batch_size:
            raise ValueError("init_attr_targets should not change the batch size")

        if self._init_attr_inputs is not None:
            inputs = self._init_attr_inputs(td["_mod_in"], additional_init_tensors)
            if not isinstance(inputs, TensorDict):
                raise ValueError("init_attr_inputs function must return a TensorDict")
        else:
            inputs = td["_mod_in"]
        if inputs.batch_size != td["_mod_in"].batch_size:
            raise ValueError("init_attr_inputs should not change the batch size")

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

        grads = self._grad_attr(targets, inputs, init_grads)
        td[self._attr_key] = grads
        return td

    @abstractmethod
    def _grad_attr(
        self,
        targets: TensorDict,
        inputs: TensorDict,
        init_grads: TensorDict,
    ) -> TensorDict:
        pass

    @torch.no_grad()
    def _multiply_by_inputs_fn(self, inputs, attrs):
        return tuple(attr * inp for attr, inp in zip(attrs, inputs))


class GradientAttributionWithBaseline(GradientAttribution):
    def __init__(
        self, *args, compute_convergence_delta: bool = False, baseline_key: UnraveledKey = "baseline", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._compute_convergence_delta = compute_convergence_delta
        self._baseline_key = baseline_key

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
