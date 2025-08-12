"""
Integrated gradients
"""

import torch

from .helpers import approximation_parameters

from tdhook.hooks import MultiHookHandle
from tdhook.module import HookedModule
from tdhook.attribution.gradient_attribution import GradientAttributionWithBaseline


class IntegratedGradients(GradientAttributionWithBaseline):
    def __init__(self, method: str = "gausslegendre", n_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._method = method
        self._n_steps = n_steps
        self._step_sizes = None

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []

        def reshape_hook(module, args, output):
            if isinstance(output, tuple):
                full_bs, *rest = output[0].shape
                return tuple(out.reshape(full_bs // self._n_steps, self._n_steps, *rest) for out in output)
            else:
                full_bs, *rest = output.shape
                return output.reshape(full_bs // self._n_steps, self._n_steps, *rest)

        handles.append(
            module.register_submodule_hook(
                "module",
                reshape_hook,
                direction="fwd",
            )
        )
        handles.append(super()._hook_module(module))
        return MultiHookHandle(handles)

    def _reduce_baselines(self, inputs, baselines):
        step_sizes_func, alphas_func = approximation_parameters(self._method)
        step_sizes, alphas = step_sizes_func(self._n_steps), alphas_func(self._n_steps)
        self._step_sizes = step_sizes

        bs, *rest = inputs[0].shape

        return tuple(
            torch.stack([baseline + alpha * (input - baseline) for alpha in alphas], dim=1)
            .reshape(bs * self._n_steps, *rest)
            .requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

    def _grad_attr(self, target, args, init_grad):
        grads = torch.autograd.grad(target, args, init_grad)
        in_dims = args[0].ndim - 1
        scaled_grads = tuple(
            grad * torch.tensor(self._step_sizes).float().view(1, self._n_steps, *(1,) * in_dims).to(grad.device)
            for grad in grads
        )
        total_grads = tuple(torch.sum(scaled_grad, dim=1) for scaled_grad in scaled_grads)
        return total_grads
