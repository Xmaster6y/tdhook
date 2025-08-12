"""
Integrated gradients
"""

import torch

from .helpers import approximation_parameters

from tdhook.attribution.gradient_attribution import GradientAttributionWithBaseline


class IntegratedGradients(GradientAttributionWithBaseline):
    def __init__(self, method: str = "gausslegendre", n_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._method = method
        self._n_steps = n_steps
        self._step_sizes = None

    def _output_backward_hook(self, module, args, output):
        if isinstance(output, tuple):
            full_bs, *rest = output[0].shape
            new_shape = (full_bs // self._n_steps, self._n_steps, *rest)
            perm = (0,) + tuple(range(2, len(new_shape))) + (1,)
            output = tuple(out.reshape(new_shape).permute(perm) for out in output)
        else:
            full_bs, *rest = output.shape
            new_shape = (full_bs // self._n_steps, self._n_steps, *rest)
            perm = (0,) + tuple(range(2, len(new_shape))) + (1,)
            output = output.reshape(new_shape).permute(perm)

        if self._init_target is not None:
            target = self._init_target(output)
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
        full_bs, *rest = grads[0].shape
        new_shape = (full_bs // self._n_steps, self._n_steps, *rest)
        perm = (0,) + tuple(range(2, len(new_shape))) + (1,)
        scaled_grads = tuple(
            grad.reshape(new_shape).permute(perm) * torch.tensor(self._step_sizes).float().to(grad.device)
            for grad in grads
        )
        total_grads = tuple(torch.sum(scaled_grad, dim=-1) for scaled_grad in scaled_grads)
        return total_grads
