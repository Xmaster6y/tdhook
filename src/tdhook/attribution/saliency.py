"""
Saliency attribution
"""

from tensordict import TensorDict
import torch

from tdhook.attribution.gradient_helpers import GradientAttribution
from tdhook.modules import td_grad


class Saliency(GradientAttribution):
    def __init__(self, *args, absolute: bool = False, multiply_by_inputs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._absolute = absolute
        self._multiply_by_inputs = multiply_by_inputs

    @torch.no_grad()
    def _grad_attr(
        self,
        targets: TensorDict,
        inputs: TensorDict,
        init_grads: TensorDict,
    ):
        grads = td_grad(targets, inputs, init_grads)
        if self._absolute:
            grads.abs_()
        if self._multiply_by_inputs:
            grads.mul_(inputs)
        return grads
