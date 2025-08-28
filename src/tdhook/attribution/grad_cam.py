"""
Grad-CAM attribution
"""

from typing import Callable, Optional, List, Tuple, Dict
from dataclasses import dataclass

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModuleBase

from tdhook._types import UnraveledKey
from tdhook.attribution.gradient_attribution import GradientAttribution
from tdhook.modules import td_grad, HookedModule, ModuleWithCache, FunctionModule
from tdhook.hooks import MultiHookHandle
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
        clean_keys: bool = True,
        keep_output_keys: bool = True,
        absolute: bool = False,
    ):
        super().__init__(
            init_attr_targets=init_attr_targets,
            init_attr_inputs=None,
            init_attr_grads=init_attr_grads,
            multiply_by_inputs=False,
            additional_init_keys=additional_init_keys,
            attribution_key=attribution_key,
            clean_keys=clean_keys,
            keep_output_keys=keep_output_keys,
        )
        self._absolute = absolute
        self._modules_to_attribute = modules_to_attribute
        self._hooked_module_kwargs["relative_path"] = "td_module.module[0]._td_module"

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        stored_keys = [f"{m}_output" for m in self._modules_to_attribute]
        mod_in_keys = [("_mod_in", key) for key in stored_keys]
        mod_out_keys = [("_mod_out", out_key) for out_key in out_keys]
        attr_keys = [(self._attr_key, key) for key in stored_keys]

        if set(self._additional_init_keys) & (set(in_keys) | set(out_keys)):
            raise ValueError("Additional init keys must not be in the in_keys or out_keys")
        modules = [
            ModuleWithCache(
                module,
                cache_key="_mod_in",
                module_out_key="_mod_out",
                stored_keys=stored_keys,
            ),
            FunctionModule(
                self._attributor_fn,
                in_keys=mod_in_keys + mod_out_keys + self._additional_init_keys,
                out_keys=attr_keys,
            ),
        ]
        return TensorDictSequential(*modules)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        cache_ref = module.td_module[0].cache_ref
        for module_key in self._modules_to_attribute:
            module.get(
                cache=cache_ref,
                cache_key=module_key,
                module_key=module_key,
                callback=lambda **kwargs: kwargs["output"].requires_grad_(True),
            )
        return MultiHookHandle()

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
