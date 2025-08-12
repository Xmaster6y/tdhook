"""
Saliency attribution
"""

import torch

from tdhook.attribution.gradient_attribution import GradientAttribution


class Saliency(GradientAttribution):
    def __init__(self, *args, absolute: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._absolute = absolute

    def _grad_attr(self, target, args, init_grad):
        grads = torch.autograd.grad(target, args, init_grad)
        return tuple(grad.abs() if self._absolute else grad for grad in grads)
