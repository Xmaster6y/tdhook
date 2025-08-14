"""
Gradient attribution
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, List

import torch
from tensordict import TensorDict
from torch import nn

from tdhook.contexts import HookingContextFactory, HookingContext
from tdhook.module import HookedModule
from tdhook.hooks import MultiHookHandle


class GradientAttribution(HookingContextFactory, metaclass=ABCMeta):
    def __init__(
        self,
        init_target: Optional[Callable] = None,
        init_grad: Optional[Callable] = None,
        multiply_by_inputs: bool = False,
    ):
        self._init_target = init_target
        self._init_grad = init_grad
        self._multiply_by_inputs = multiply_by_inputs

    def _spawn_hooked_module(
        self, prep_module: nn.Module, in_keys: List[str], out_keys: List[str], hooking_context: HookingContext
    ):
        out_keys = out_keys + [f"{in_key}_attr" for in_key in in_keys]

        return super()._spawn_hooked_module(prep_module, in_keys, out_keys, hooking_context)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []

        handles.append(
            module.register_submodule_hook(
                "module",
                self._input_grad_hook,
                direction="fwd_pre",
            )
        )

        handles.append(
            module.register_submodule_hook(
                "module",
                self._output_backward_hook,
                direction="fwd",
            )
        )

        if self._multiply_by_inputs:
            handles.append(
                module.register_submodule_hook(
                    "",
                    self._attr_multiply_hook,
                    direction="fwd",
                )
            )

        return MultiHookHandle(handles)

    def _input_grad_hook(self, module, args):
        for arg in args:
            arg.requires_grad_(True)

    def _output_backward_hook(self, module, args, output):
        if self._init_target is not None:
            target = self._init_target(output)  # TODO: pass kwargs to init_target (e.g. labels)
        else:
            target = output
        if self._init_grad is not None:
            init_grad = self._init_grad(output)
        else:
            init_grad = torch.ones_like(target)
        attrs = self._grad_attr(target, args, init_grad)
        if isinstance(output, tuple):
            return *output, *attrs
        else:
            return output, *attrs

    def _attr_multiply_hook(self, module, args, output):
        self._multiply_by_inputs_(output, module.in_keys)
        return output

    @abstractmethod
    def _grad_attr(self, target, args, init_grad):
        pass

    @torch.no_grad()
    def _multiply_by_inputs_(self, output, in_keys):
        for key in in_keys:
            output[f"{key}_attr"] *= output[key]


class GradientAttributionWithBaseline(GradientAttribution):
    def __init__(self, *args, compute_convergence_delta: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_convergence_delta = compute_convergence_delta

    def _spawn_hooked_module(
        self, prep_module: nn.Module, in_keys: List[str], out_keys: List[str], hooking_context: HookingContext
    ):
        hooked_module = super()._spawn_hooked_module(prep_module, in_keys, out_keys, hooking_context)
        hooked_module.td_module.in_keys = in_keys + [f"{in_key}_baseline" for in_key in in_keys]
        if self._compute_convergence_delta:
            hooked_module.td_module._out_keys = hooked_module.td_module._out_keys + ["convergence_delta"]
        return hooked_module

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []

        handles.append(
            module.register_submodule_hook(
                "",
                self._assert_batched_hook,
                direction="fwd_pre",
            )
        )

        handles.append(
            module.register_submodule_hook(
                "module",
                self._reduce_baselines_hook,
                direction="fwd_pre",
            )
        )

        handles.append(super()._hook_module(module))

        if self._compute_convergence_delta:
            handles.append(
                module.register_submodule_hook(
                    "module",
                    self._convergence_delta_placeholder_hook,
                    direction="fwd",
                )
            )

            handles.append(
                module.register_submodule_hook(
                    "",
                    self._compute_convergence_delta_hook,
                    direction="fwd",
                )
            )

        return MultiHookHandle(handles)

    def _assert_batched_hook(self, module, args):
        if args[0].ndim == 0:
            raise NotImplementedError("This attribution method requires batched inputs")

    def _reduce_baselines_hook(self, module, args):
        inputs = args[: len(args) // 2]
        baselines = args[len(args) // 2 :]
        return self._reduce_baselines(inputs, baselines)

    def _convergence_delta_placeholder_hook(self, module, args, output):
        if isinstance(output, tuple):
            return *output, torch.empty(output[0].shape[0], 1)
        else:
            return output, torch.empty(output.shape[0], 1)

    def _compute_convergence_delta_hook(self, module, args, output):
        return self._compute_convergence_delta_(module, output)

    def _multiply_by_inputs_(self, output, in_keys):
        for key in in_keys:
            if not key.endswith("_baseline"):
                output[f"{key}_attr"] *= output[key] - output[f"{key}_baseline"]

    @abstractmethod
    def _reduce_baselines(self, inputs, baselines):
        pass

    def _compute_convergence_delta_(
        self,
        hooked_module: HookedModule,
        output: TensorDict,
    ):
        module_in_keys = [key for key in hooked_module.in_keys if not key.endswith("_baseline")]

        with torch.no_grad():
            with hooked_module.disable_context() as raw_module:
                if self._init_target is not None:
                    start_out = self._init_target(raw_module(*(output[f"{key}_baseline"] for key in module_in_keys)))
                else:
                    start_out = raw_module(*(output[f"{key}_baseline"] for key in module_in_keys))
                start_out_sum = start_out.reshape(output.shape[0], -1).sum(dim=1)
                if self._init_target is not None:
                    end_out = self._init_target(raw_module(*(output[key] for key in module_in_keys)))
                else:
                    end_out = raw_module(*(output[key] for key in module_in_keys))
                end_out_sum = end_out.reshape(output.shape[0], -1).sum(dim=1)

                row_sums = torch.stack(
                    [output[f"{key}_attr"].reshape(output.shape[0], -1).sum(dim=1) for key in module_in_keys]
                )
                attr_sum = row_sums.sum(dim=0)
                output["convergence_delta"] = attr_sum - (end_out_sum - start_out_sum)
                return output
