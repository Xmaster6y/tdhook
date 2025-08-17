"""
Gradient attribution
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Tuple, List

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory
from tdhook.module import HookedModule, FunctionModule
from tdhook.hooks import MultiHookHandle
from tdhook._types import UnraveledKey


class GradientAttribution(HookingContextFactory, metaclass=ABCMeta):
    def __init__(
        self,
        init_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        multiply_by_inputs: bool = False,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        attribution_key: Optional[UnraveledKey] = None,
    ):
        self._init_targets = init_targets
        self._init_grads = init_grads
        self._multiply_by_inputs = multiply_by_inputs
        self._additional_init_keys = additional_init_keys or []
        self._attr_key = attribution_key or "attr"

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        n_in_keys = len(in_keys)
        saved_keys = [("saved", in_key) for in_key in in_keys]
        attr_keys = [(self._attr_key, in_key) for in_key in in_keys]

        if set(self._additional_init_keys) & (set(in_keys) | set(out_keys)):
            raise ValueError("Additional init keys must not be in the in_keys or out_keys")
        modules = []

        modules.append(
            TensorDictModule(
                self._register_inputs_fn,
                in_keys=in_keys,
                out_keys=saved_keys,
                inplace=True,
            )
        )
        modules.append(module)
        modules.append(
            FunctionModule(
                lambda td: self._attributor_fn(td, saved_keys, out_keys),
                in_keys=saved_keys + out_keys + self._additional_init_keys,
                out_keys=attr_keys,
            )
        )
        if self._multiply_by_inputs:
            modules.append(
                TensorDictModule(
                    lambda *tensors: self._multiply_by_inputs_fn(tensors[:n_in_keys], tensors[n_in_keys:]),
                    in_keys=saved_keys + attr_keys,
                    out_keys=attr_keys,
                    inplace=True,
                )
            )
        return TensorDictSequential(*modules)

    def _register_inputs_fn(self, *inputs):  # TODO: needed with autograd?
        return tuple(inp.requires_grad_(True) for inp in inputs)

    def _attributor_fn(
        self, td: TensorDict, saved_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDict:
        outputs = td.select(*out_keys)
        additional_init_tensors = td.select(*self._additional_init_keys)

        if self._init_targets is not None:
            targets = self._init_targets(outputs, additional_init_tensors)
            if not isinstance(targets, TensorDict):
                raise ValueError("init_targets function must return a TensorDict")
        else:
            targets = outputs
        if self._init_grads is not None:
            init_grads = self._init_grads(targets, additional_init_tensors)
            if not isinstance(init_grads, TensorDict):
                raise ValueError("init_grads function must return a TensorDict")
        else:
            init_grads = torch.ones_like(targets)

        if set(targets.keys(True, True)) != set(init_grads.keys(True, True)):
            raise ValueError("Targets and init_grads must have the same keys")
        for target_key, target in targets.items():
            if target.grad_fn is None:
                raise ValueError(f"Target {target_key} has no grad_fn")

        inputs = td.select(*saved_keys)

        grads = self._grad_attr(targets, inputs, init_grads)
        td[self._attr_key] = grads["saved"]
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
    def __init__(self, *args, compute_convergence_delta: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_convergence_delta = compute_convergence_delta

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        n_in_keys = len(in_keys)
        saved_keys = [("saved", in_key) for in_key in in_keys]
        attr_keys = [(self._attr_key, in_key) for in_key in in_keys]
        baseline_keys = [("baseline", in_key) for in_key in in_keys]
        (_, _, attributor_module, *_) = super()._prepare_module(module, in_keys, out_keys)

        modules = []
        modules.append(
            TensorDictModule(
                lambda *tensors: self._reduce_baselines_fn(tensors[:n_in_keys], tensors[n_in_keys:]),
                in_keys=in_keys + baseline_keys,
                out_keys=saved_keys,
                inplace=True,
            )
        )
        modules.append(
            TensorDictModule(
                self._register_inputs_fn,
                in_keys=saved_keys,
                out_keys=saved_keys,
                inplace=True,
            )
        )
        modules.append(
            FunctionModule(
                lambda td: self._module_call_fn(td, module),
                in_keys=saved_keys,
                out_keys=out_keys,
            )
        )
        modules.append(attributor_module)
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

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []

        handles.append(
            module.register_submodule_hook(
                "td_module",
                self._assert_batched_hook,  # TODO: support unbatched inputs
                direction="fwd_pre",
                relative=False,
            )
        )

        handles.append(super()._hook_module(module))

        return MultiHookHandle(handles)

    def _assert_batched_hook(self, module, args):
        if args[0].ndim == 0:
            raise NotImplementedError("This attribution method requires batched inputs")

    @abstractmethod
    def _reduce_baselines_fn(
        self, inputs: Tuple[torch.Tensor, ...], baselines: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def _module_call_fn(self, td: TensorDict, module: TensorDictModuleBase) -> TensorDict:
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
        baseline_keys = [("baseline", in_key) for in_key in in_keys]
        attr_keys = [(self._attr_key, in_key) for in_key in in_keys]

        inputs = td.select(*in_keys)
        baselines = td.select(*baseline_keys)["baseline"]
        attrs = td.select(*attr_keys)
        additional_init_tensors = td.select(*self._additional_init_keys)

        if self._init_targets is not None:
            start_out = self._init_targets(module(baselines), additional_init_tensors)
        else:
            start_out = module(baselines).select(*out_keys)
        start_out_sum = start_out.sum(dim="feature", reduce=True)
        if self._init_targets is not None:
            end_out = self._init_targets(module(inputs), additional_init_tensors)
        else:
            end_out = module(inputs).select(*out_keys)
        end_out_sum = end_out.sum(dim="feature", reduce=True)

        attr_sum = attrs.sum(dim="feature", reduce=True)
        td["convergence_delta"] = attr_sum - (end_out_sum - start_out_sum)
        return td
