"""
Saliency attribution
"""

from typing import List, Optional, Callable, Dict
from tensordict import TensorDict
import torch

from tdhook._types import UnraveledKey
from tdhook.attribution.gradient_helpers import GradientAttribution


class Saliency(GradientAttribution):
    def __init__(
        self,
        use_inputs: bool = True,
        use_outputs: bool = True,
        input_modules: Optional[List[str]] = None,
        target_modules: Optional[List[str]] = None,
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_inputs: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        output_grad_callbacks: Optional[Dict[str, Callable]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_intermediate_keys: bool = True,
        cache_callback: Optional[Callable] = None,
        absolute: bool = False,
        multiply_by_inputs: bool = False,
    ):
        super().__init__(
            use_inputs=use_inputs,
            use_outputs=use_outputs,
            input_modules=input_modules,
            target_modules=target_modules,
            init_attr_targets=init_attr_targets,
            init_attr_inputs=init_attr_inputs,
            init_attr_grads=init_attr_grads,
            additional_init_keys=additional_init_keys,
            output_grad_callbacks=output_grad_callbacks,
            attribution_key=attribution_key,
            clean_intermediate_keys=clean_intermediate_keys,
            cache_callback=cache_callback,
        )
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
