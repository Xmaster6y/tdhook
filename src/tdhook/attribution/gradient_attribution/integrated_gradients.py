"""
Integrated gradients
"""

import torch
from typing import Tuple
from tensordict.nn import TensorDictModule

from .helpers import approximation_parameters

from tdhook.attribution.gradient_attribution import GradientAttributionWithBaseline


class IntegratedGradients(GradientAttributionWithBaseline):
    def __init__(self, method: str = "gausslegendre", n_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._method = method
        self._n_steps = n_steps
        self._step_sizes = None

    def _reduce_baselines_fn(self, inputs, baselines):
        step_sizes_func, alphas_func = approximation_parameters(self._method)
        step_sizes, alphas = step_sizes_func(self._n_steps), alphas_func(self._n_steps)
        self._step_sizes = step_sizes

        return tuple(
            torch.stack([baseline + alpha * (input - baseline) for alpha in alphas], dim=-1).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

    def _module_call_fn(self, inputs: Tuple[torch.Tensor, ...], module: TensorDictModule) -> Tuple[torch.Tensor, ...]:
        n_in_dim = len(inputs[0].shape)
        bs, *rest, n_steps = inputs[0].shape
        perm = (0, n_in_dim - 1) + tuple(range(1, n_in_dim - 1))

        outputs = module(*(input.permute(perm).reshape(bs * n_steps, *rest) for input in inputs))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        n_out_dim = len(outputs[0].shape) + 1
        _, *rest = outputs[0].shape
        inv_perm = (0,) + tuple(range(2, n_out_dim)) + (1,)
        return tuple(output.reshape(bs, n_steps, *rest).permute(inv_perm) for output in outputs)

    def _grad_attr(
        self,
        targets: Tuple[torch.Tensor, ...],
        inputs: Tuple[torch.Tensor, ...],
        init_grads: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        grads = torch.autograd.grad(targets, inputs, init_grads)
        scaled_grads = tuple(grad * torch.tensor(self._step_sizes).float().to(grad.device) for grad in grads)
        return tuple(torch.sum(scaled_grad, dim=-1) for scaled_grad in scaled_grads)
