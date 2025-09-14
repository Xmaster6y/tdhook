"""
Guided Backpropagation
"""

from typing import Callable, Tuple, Type
import torch
from torch import nn
from tensordict import TensorDict

from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle, MultiHookManager, HookFactory, DIRECTION_TO_RETURN
from tdhook.attribution.gradient_helpers import GradientAttribution


class GuidedBackpropagation(GradientAttribution):
    def __init__(
        self, *args, multiply_by_inputs: bool = False, classes_to_skip: Tuple[Type[nn.Module], ...] = (), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._hook_manager = MultiHookManager(pattern=r".+", classes_to_skip=classes_to_skip)
        self._multiply_by_inputs = multiply_by_inputs

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        def hook_factory(name: str) -> Callable:
            def callback(**kwargs):
                return tuple(
                    None if out is None else nn.functional.relu(out) for out in kwargs[DIRECTION_TO_RETURN["bwd"]]
                )

            return HookFactory.make_setting_hook(None, callback=callback, direction="bwd")

        guided_handle = self._hook_manager.register_hook(
            module, hook_factory, direction="bwd", relative_path=module.relative_path
        )
        return MultiHookHandle([guided_handle, super()._hook_module(module)])

    @torch.no_grad()
    def _grad_attr(
        self,
        grads: TensorDict,
        inputs: TensorDict,
    ):
        if self._multiply_by_inputs:
            grads.mul_(inputs)
        return grads
