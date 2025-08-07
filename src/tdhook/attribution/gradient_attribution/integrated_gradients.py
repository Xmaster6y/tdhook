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

    def _reduce_baselines(self, inputs, baselines):
        step_sizes_func, alphas_func = approximation_parameters(self._method)
        step_sizes, alphas = step_sizes_func(self._n_steps), alphas_func(self._n_steps)
        self._step_sizes = step_sizes

        return tuple(
            torch.cat([baseline + alpha * (input - baseline) for alpha in alphas], dim=0).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

    def _grad_attr(self, args, output):
        scaled_grads = tuple(
            arg.grad.contiguous().view(self._n_steps, -1)
            * torch.tensor(self._step_sizes).float().view(self._n_steps, 1).to(arg.grad.device)
            for arg in args
        )
        total_grads = tuple(
            torch.sum(
                scaled_grad.reshape((self._n_steps, arg.grad.shape[0] // self._n_steps) + arg.grad.shape[1:]), dim=0
            )
            for (scaled_grad, arg) in zip(scaled_grads, args)
        )
        return total_grads
