"""
Integrated gradients
"""

import torch
from typing import Tuple
from tensordict.nn import TensorDictModuleBase
from tensordict import TensorDict, merge_tensordicts

from .helpers import approximation_parameters

from tdhook.attribution.gradient_attribution import GradientAttributionWithBaseline
from tdhook.module import td_grad


class IntegratedGradients(GradientAttributionWithBaseline):
    def __init__(self, method: str = "gausslegendre", n_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._method = method
        self._n_steps = n_steps
        self._step_sizes = None

    def _reduce_baselines_fn(self, inputs: Tuple[torch.Tensor, ...], baselines: Tuple[torch.Tensor, ...]):
        step_sizes_func, alphas_func = approximation_parameters(self._method)
        step_sizes, alphas = step_sizes_func(self._n_steps), alphas_func(self._n_steps)
        self._step_sizes = step_sizes

        return tuple(
            torch.stack([baseline + alpha * (input - baseline) for alpha in alphas], dim=1).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

    def _module_call_fn(self, td: TensorDict, module: TensorDictModuleBase) -> TensorDict:
        inputs = td["saved"]
        inputs.batch_size = (*inputs.batch_size, self._n_steps)

        outputs = module(inputs.flatten()).select(*module.out_keys).reshape(inputs.shape)

        inputs.batch_size = inputs.batch_size[:-1]
        outputs.batch_size = outputs.batch_size[:-1]

        return merge_tensordicts(td, outputs.apply(self._permute_fn))  # TODO: update td inplace?

    def _grad_attr(
        self,
        targets: TensorDict,
        inputs: TensorDict,
        init_grads: TensorDict,
    ) -> TensorDict:
        grads = td_grad(targets, inputs, init_grads).apply(self._permute_fn)

        steps = torch.tensor(self._step_sizes).float().to(grads.device)

        def multiply_sum_fn(grad: torch.Tensor) -> torch.Tensor:
            return torch.sum(grad.mul_(steps), dim=-1)

        return grads.apply(multiply_sum_fn)

    @staticmethod
    def _permute_fn(out: torch.Tensor) -> torch.Tensor:
        n_out_dim = len(out.shape)
        perm = (0,) + tuple(range(2, n_out_dim)) + (1,)
        return out.permute(perm)
