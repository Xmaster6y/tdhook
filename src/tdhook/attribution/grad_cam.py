"""
Grad-CAM attribution
"""

from typing import Callable, Optional, List, Tuple, Dict
from dataclasses import dataclass

from tensordict import TensorDict
import torch

from tdhook._types import UnraveledKey
from tdhook.attribution.gradient_helpers import GradientAttribution
from tdhook.modules import td_grad
from tdhook.contexts import HookingContextWithCache


@dataclass
class DimsConfig:
    feature_dims: Optional[Tuple[int, ...]] = None
    pooling_dims: Optional[Tuple[int, ...]] = None


class GradCAM(GradientAttribution):
    _hooking_context_class = HookingContextWithCache

    def __init__(
        self,
        modules_to_attribute: Dict[str, DimsConfig],
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_intermediate_keys: bool = True,
        absolute: bool = False,
    ):
        super().__init__(
            use_inputs=False,
            use_outputs=True,
            input_modules=modules_to_attribute.keys(),
            target_modules=None,
            init_attr_targets=init_attr_targets,
            init_attr_inputs=None,
            init_attr_grads=init_attr_grads,
            additional_init_keys=additional_init_keys,
            attribution_key=attribution_key,
            clean_intermediate_keys=clean_intermediate_keys,
        )
        self._absolute = absolute
        self._modules_to_attribute = modules_to_attribute

    @torch.no_grad()
    def _grad_attr(
        self,
        targets: TensorDict,
        inputs: TensorDict,
        init_grads: TensorDict,
    ) -> TensorDict:
        grads = td_grad(targets, inputs, init_grads)
        if self._absolute:
            grads.abs_()
        attrs = TensorDict()
        for key in grads.keys(True, True):
            dims_config = self._modules_to_attribute[key]
            if dims_config.feature_dims is not None:
                weights = grads[key].mean(dim=dims_config.feature_dims, keepdim=True)
            else:
                weights = grads[key]
            if dims_config.pooling_dims is not None:
                attrs[key] = (weights * inputs[key]).sum(dim=dims_config.pooling_dims)
            else:
                attrs[key] = weights * inputs[key]
        return attrs
