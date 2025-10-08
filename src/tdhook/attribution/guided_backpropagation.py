"""
Guided Backpropagation
"""

from typing import Callable, Tuple, Type, Optional, List, Dict
import torch
from torch import nn
from tensordict import TensorDict

from tdhook._types import UnraveledKey
from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle, MultiHookManager, HookFactory, DIRECTION_TO_RETURN
from tdhook.attribution.gradient_helpers import GradientAttribution


class GuidedBackpropagation(GradientAttribution):
    def __init__(
        self,
        use_inputs: bool = True,
        use_outputs: bool = True,
        input_modules: Optional[List[str]] = None,
        target_modules: Optional[List[str]] = None,
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_inputs: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_cache_in: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        output_grad_callbacks: Optional[Dict[str, Callable]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_intermediate_keys: bool = True,
        cache_callback: Optional[Callable] = None,
        multiply_by_inputs: bool = False,
        classes_to_skip: Tuple[Type[nn.Module], ...] = (),
    ):
        super().__init__(
            use_inputs=use_inputs,
            use_outputs=use_outputs,
            input_modules=input_modules,
            target_modules=target_modules,
            init_attr_targets=init_attr_targets,
            init_attr_inputs=init_attr_inputs,
            init_attr_cache_in=init_attr_cache_in,
            init_attr_grads=init_attr_grads,
            additional_init_keys=additional_init_keys,
            output_grad_callbacks=output_grad_callbacks,
            attribution_key=attribution_key,
            clean_intermediate_keys=clean_intermediate_keys,
            cache_callback=cache_callback,
        )
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
