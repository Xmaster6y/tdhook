"""
Gradient attribution
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Tuple, Dict, List

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential

from tdhook.contexts import HookingContextFactory
from tdhook.module import HookedModule
from tdhook.hooks import MultiHookHandle


class GradientAttribution(HookingContextFactory, metaclass=ABCMeta):
    def __init__(
        self,
        init_targets: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        init_grads: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        multiply_by_inputs: bool = False,
        additional_init_keys: Optional[List[str]] = None,
    ):
        self._init_targets = init_targets
        self._init_grads = init_grads
        self._multiply_by_inputs = multiply_by_inputs
        self._additional_init_keys = additional_init_keys or []

    def _prepare_module(self, module: TensorDictModule) -> TensorDictModule:
        n_in_keys = len(module.in_keys)
        attr_keys = [f"{in_key}_attr" for in_key in module.in_keys]

        if set(self._additional_init_keys) & (set(module.in_keys) | set(module.out_keys)):
            raise ValueError("Additional init keys must not be in the module's in_keys or out_keys")

        modules = [module]
        modules.append(
            TensorDictModule(
                lambda **tensors: self._attributor_fn(tensors, module.in_keys, module.out_keys),
                in_keys={k: k for k in module.in_keys + module.out_keys + self._additional_init_keys},
                out_keys=attr_keys,
                inplace=True,
            )  # TODO: replace with custom module to pass TensorDict to _attributor_fn
        )
        if self._multiply_by_inputs:
            modules.append(
                TensorDictModule(
                    lambda *tensors: self._multiply_by_inputs_fn(tensors[:n_in_keys], tensors[n_in_keys:]),
                    in_keys=module.in_keys + attr_keys,
                    out_keys=attr_keys,
                    inplace=True,
                )
            )
        return TensorDictSequential(*modules)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handle = module.register_submodule_hook(
            "td_module",
            lambda *args: self._set_requires_grad_hook(*args, module.in_keys),
            direction="fwd_pre",
            relative=False,
        )
        return MultiHookHandle([handle])

    def _set_requires_grad_hook(self, module, args, in_keys):  # TODO: needed with autograd?
        for arg in args:
            arg.requires_grad_(True)

    def _attributor_fn(
        self, tensors: Dict[str, torch.Tensor], in_keys: List[str], out_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        outputs = {k: tensors[k] for k in out_keys}
        additional_init_tensors = {k: tensors[k] for k in self._additional_init_keys}

        if self._init_targets is not None:
            targets = self._init_targets(outputs, additional_init_tensors)
        else:
            targets = outputs
        if self._init_grads is not None:
            init_grads = self._init_grads(targets, additional_init_tensors)
        else:
            init_grads = {k: torch.ones_like(target) for k, target in targets.items()}

        if set(targets.keys()) != set(init_grads.keys()):
            raise ValueError("Targets and init_grads must have the same keys")

        tup_targets = tuple(targets[k] for k in targets.keys())
        tup_init_grads = tuple(init_grads[k] for k in targets.keys())
        tup_inputs = tuple(tensors[k] for k in in_keys)

        return self._grad_attr(tup_targets, tup_inputs, tup_init_grads)

    @abstractmethod
    def _grad_attr(
        self,
        targets: Tuple[torch.Tensor, ...],
        inputs: Tuple[torch.Tensor, ...],
        init_grads: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @torch.no_grad()
    def _multiply_by_inputs_fn(self, inputs, attrs):
        return tuple(attr * inp for attr, inp in zip(attrs, inputs))


class GradientAttributionWithBaseline(GradientAttribution):
    def __init__(self, *args, compute_convergence_delta: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_convergence_delta = compute_convergence_delta

    def _prepare_module(self, module: TensorDictModule) -> TensorDictModule:
        n_in_keys = len(module.in_keys)
        attr_keys = [f"{in_key}_attr" for in_key in module.in_keys]
        original_in_keys = [f"{in_key}_original" for in_key in module.in_keys]
        baseline_keys = [f"{in_key}_baseline" for in_key in module.in_keys]
        (_, attributor_module, *_) = super()._prepare_module(module)

        modules = []
        modules.append(
            TensorDictModule(
                lambda *tensors: tensors,
                in_keys=module.in_keys,
                out_keys=original_in_keys,
                inplace=True,
            )
        )
        modules.append(
            TensorDictModule(
                lambda *tensors: self._reduce_baselines_fn(tensors[:n_in_keys], tensors[n_in_keys:]),
                in_keys=original_in_keys + baseline_keys,
                out_keys=module.in_keys,
                inplace=True,
            )
        )

        modules.append(
            TensorDictModule(
                lambda *inputs: self._module_call_fn(inputs, module),
                in_keys=module.in_keys,
                out_keys=module.out_keys,
                inplace=True,
            )
        )
        modules.append(attributor_module)
        if self._multiply_by_inputs:
            modules.append(
                TensorDictModule(
                    lambda *tensors: self._multiply_by_inputs_fn(
                        tensors[:n_in_keys], tensors[n_in_keys : n_in_keys * 2], tensors[n_in_keys * 2 :]
                    ),
                    in_keys=original_in_keys + baseline_keys + attr_keys,
                    out_keys=attr_keys,
                    inplace=True,
                )
            )
        if self._compute_convergence_delta:
            modules.append(
                TensorDictModule(
                    lambda **tensors: self._compute_convergence_delta_fn(
                        tensors, module.in_keys, module.out_keys, module
                    ),
                    in_keys={k: k for k in original_in_keys + baseline_keys + attr_keys + self._additional_init_keys},
                    out_keys=["convergence_delta"],
                    inplace=True,
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
    def _module_call_fn(self, inputs: Tuple[torch.Tensor, ...], module: TensorDictModule) -> Tuple[torch.Tensor, ...]:
        pass

    @torch.no_grad()
    def _multiply_by_inputs_fn(
        self, inputs: Tuple[torch.Tensor, ...], baselines: Tuple[torch.Tensor, ...], attrs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(attr * (inp - baseline) for attr, inp, baseline in zip(attrs, inputs, baselines))

    @torch.no_grad()
    def _compute_convergence_delta_fn(
        self,
        tensors: Dict[str, torch.Tensor],
        in_keys: List[str],
        out_keys: List[str],
        module: TensorDictModule,
    ) -> torch.Tensor:
        tup_inputs = tuple(tensors[f"{k}_original"] for k in in_keys)
        tup_baselines = tuple(tensors[f"{k}_baseline"] for k in in_keys)
        tup_attrs = tuple(tensors[f"{k}_attr"] for k in in_keys)

        additional_init_tensors = {k: tensors[k] for k in self._additional_init_keys}

        batch_size = tup_inputs[0].shape[0]

        if self._init_targets is not None:
            start_out = self._init_targets(dict(zip(out_keys, module(*tup_baselines))), additional_init_tensors)
        else:
            start_out = dict(zip(out_keys, module(*tup_baselines)))
        start_out_sum = sum(start_out[k].reshape(batch_size, -1).sum(dim=1) for k in start_out.keys())
        if self._init_targets is not None:
            end_out = self._init_targets(dict(zip(out_keys, module(*tup_inputs))), additional_init_tensors)
        else:
            end_out = dict(zip(out_keys, module(*tup_inputs)))
        end_out_sum = sum(end_out[k].reshape(batch_size, -1).sum(dim=1) for k in end_out.keys())

        inputs_sums = torch.stack([attr.reshape(batch_size, -1).sum(dim=1) for attr in tup_attrs])
        attr_sum = inputs_sums.sum(dim=0)
        return attr_sum - (end_out_sum - start_out_sum)
