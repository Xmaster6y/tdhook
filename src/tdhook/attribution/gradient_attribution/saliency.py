"""
Saliency attribution
"""

import torch
from typing import Tuple

from tdhook.attribution.gradient_attribution import GradientAttribution


class Saliency(GradientAttribution):
    def __init__(self, *args, absolute: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._absolute = absolute

    def _grad_attr(
        self,
        targets: Tuple[torch.Tensor, ...],
        inputs: Tuple[torch.Tensor, ...],
        init_grads: Tuple[torch.Tensor, ...],
    ):
        grads = torch.autograd.grad(targets, inputs, init_grads)
        return tuple(grad.abs() if self._absolute else grad for grad in grads)
