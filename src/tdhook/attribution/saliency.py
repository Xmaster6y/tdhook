"""
Saliency attribution
"""

from tdhook.attribution.gradient_attribution import GradientAttribution


class Saliency(GradientAttribution):
    @staticmethod
    def _grad_attr(args, output):
        return (arg.grad for arg in args)
