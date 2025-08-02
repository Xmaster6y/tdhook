"""
Input times gradient
"""

from tdhook.attribution.gradient_attribution import GradientAttribution


class InputTimesGradient(GradientAttribution):
    @staticmethod
    def _grad_attr(args, output):
        return (arg * arg.grad for arg in args)
