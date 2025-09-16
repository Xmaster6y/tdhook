"""
Saliency attribution
"""

from tensordict import TensorDict
import torch

from tdhook.attribution.gradient_helpers import GradientAttribution


class Saliency(GradientAttribution):
    def __init__(self, *args, absolute: bool = False, multiply_by_inputs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._absolute = absolute
        self._multiply_by_inputs = multiply_by_inputs

    @torch.no_grad()
    def _grad_attr(
        self,
        grads: TensorDict,
        inputs: TensorDict,
    ):
        if self._absolute:
            grads.abs_()
        if self._multiply_by_inputs:
            grads.mul_(inputs)
        return grads
