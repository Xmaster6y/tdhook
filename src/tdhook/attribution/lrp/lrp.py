"""
LRP
"""

from typing import Callable, Optional, Dict, List

from torch import nn
import torch
from warnings import warn
from tensordict.nn import TensorDictModule

from tdhook.attribution.gradient_attribution import GradientAttribution

from .rules import Rule


class LRP(GradientAttribution):
    def __init__(
        self,
        rule_mapper: Callable[[str, nn.Module], Rule | None],
        init_targets: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        init_grads: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        additional_init_keys: Optional[List[str]] = None,
        warn_on_missing_rule: bool = True,
        skip_modules: Optional[Callable[[str, nn.Module], bool]] = None,
    ):
        super().__init__(
            init_targets=init_targets,
            init_grads=init_grads,
            multiply_by_inputs=False,
            additional_init_keys=additional_init_keys,
        )
        self._rule_mapper = rule_mapper
        self._warn_on_missing_rule = warn_on_missing_rule
        self._skip_modules = skip_modules

    def _prepare_module(
        self,
        module: TensorDictModule,
    ) -> TensorDictModule:
        rule_map = {}
        for name, child in module.named_modules():
            if self._skip_modules and self._skip_modules(name, child):
                continue
            rule = self._rule_mapper(name, child)
            if rule is not None:
                rule.register(child)
                rule_map[name] = rule
            elif self._warn_on_missing_rule:
                warn(f"No rule found for module `{name}` ({type(child).__name__})")
        module._rule_map = rule_map
        return super()._prepare_module(module)

    def _restore_module(self, module: TensorDictModule) -> TensorDictModule:
        module = super()._restore_module(module)
        for name, child in module.named_modules():
            rule = self._rule_mapper(name, child)
            if rule is not None:
                rule.unregister(child)
        del module._rule_map
        return module

    def _grad_attr(self, targets, inputs, init_grads):
        return torch.autograd.grad(targets, inputs, init_grads)

    @staticmethod
    def default_skip(name: str, module: nn.Module) -> bool:
        names_to_skip = ("", "td_module")
        classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModule)
        return name in names_to_skip or isinstance(module, classes_to_skip)
