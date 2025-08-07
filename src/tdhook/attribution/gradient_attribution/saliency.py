"""
Saliency attribution
"""

from tdhook.attribution.gradient_attribution import GradientAttribution


class Saliency(GradientAttribution):
    def _grad_attr(self, args, output):
        return (arg.grad for arg in args)
