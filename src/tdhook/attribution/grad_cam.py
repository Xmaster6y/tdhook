"""
Grad-CAM attribution
"""

from typing import Callable, Optional, List, Tuple, Dict
from dataclasses import dataclass

from tensordict import TensorDict
import torch

from tdhook._types import UnraveledKey
from tdhook.attribution.gradient_helpers import GradientAttribution


@dataclass
class DimsConfig:
    weight_pooling_dims: Optional[Tuple[int, ...]] = None
    feature_sum_dims: Optional[Tuple[int, ...]] = None


class GradCAM(GradientAttribution):
    def __init__(
        self,
        modules_to_attribute: Optional[Dict[str, DimsConfig]],
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
        absolute: bool = False,
    ):
        super().__init__(
            use_inputs=False,
            use_outputs=True,
            input_modules=modules_to_attribute.keys() if modules_to_attribute is not None else None,
            target_modules=None,
            init_attr_targets=init_attr_targets,
            init_attr_inputs=None,
            init_attr_cache_in=init_attr_cache_in,
            init_attr_grads=init_attr_grads,
            additional_init_keys=additional_init_keys,
            attribution_key=attribution_key,
            clean_intermediate_keys=clean_intermediate_keys,
            output_grad_callbacks=output_grad_callbacks,
        )
        self._absolute = absolute
        self._modules_to_attribute = modules_to_attribute

    @torch.no_grad()
    def _grad_attr(
        self,
        grads: TensorDict,
        inputs: TensorDict,
    ) -> TensorDict:
        if self._absolute:
            grads.abs_()
        attrs = TensorDict()
        for key in grads.keys(True, True):
            dims_config = self._modules_to_attribute[key]
            if dims_config.weight_pooling_dims is not None:
                weights = grads[key].mean(dim=dims_config.weight_pooling_dims, keepdim=True)
            else:
                weights = grads[key]
            if dims_config.feature_sum_dims is not None:
                attrs[key] = (weights * inputs[key]).sum(dim=dims_config.feature_sum_dims)
            else:
                attrs[key] = weights * inputs[key]
        return attrs
