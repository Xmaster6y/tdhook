"""
Guided Backpropagation
"""

from typing import Callable
from torch import nn
from tensordict import TensorDict

from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle, MultiHookManager, HookFactory, DIRECTION_TO_RETURN
from tdhook.modules import td_grad
from tdhook.attribution.gradient_helpers import GradientAttribution


class GuidedBackpropagation(GradientAttribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_manager = MultiHookManager(pattern=r".*", module_classes=(nn.ReLU,))

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        def hook_factory(name: str) -> Callable:
            def callback(**kwargs):
                return tuple(nn.functional.relu(out) for out in kwargs[DIRECTION_TO_RETURN["bwd"]])

            return HookFactory.make_setting_hook(None, callback=callback, direction="bwd")

        return self._hook_manager.register_hook(module, hook_factory, direction="bwd")

    def _grad_attr(
        self,
        targets: TensorDict,
        inputs: TensorDict,
        init_grads: TensorDict,
    ):
        return td_grad(targets, inputs, init_grads)
