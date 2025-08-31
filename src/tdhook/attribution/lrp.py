"""
LRP
"""

from typing import Callable, Optional, List

from torch import nn
from warnings import warn
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict import TensorDict

from tdhook.attribution.gradient_helpers import GradientAttribution
from tdhook.modules import td_grad
from tdhook._types import UnraveledKey
from tdhook.attribution.lrp_helpers.rules import Rule


class LRP(GradientAttribution):
    def __init__(
        self,
        rule_mapper: Callable[[str, nn.Module], Rule | None],
        warn_on_missing_rule: bool = True,
        skip_modules: Optional[Callable[[str, nn.Module], bool]] = None,
        **kwargs,
    ):
        kwargs["multiply_by_inputs"] = False
        super().__init__(
            **kwargs,
        )
        self._rule_mapper = rule_mapper
        self._warn_on_missing_rule = warn_on_missing_rule
        self._skip_modules = skip_modules

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
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
        return super()._prepare_module(module, in_keys, out_keys)

    def _restore_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        module = super()._restore_module(module, in_keys, out_keys)
        for name, child in module.named_modules():
            rule = self._rule_mapper(name, child)
            if rule is not None:
                rule.unregister(child)
        del module._rule_map
        return module

    def _grad_attr(
        self,
        targets: TensorDict,
        inputs: TensorDict,
        init_grads: TensorDict,
    ) -> TensorDict:
        return td_grad(targets, inputs, init_grads)

    @staticmethod
    def default_skip(name: str, module: nn.Module) -> bool:
        names_to_skip = ("", "td_module")
        classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModule)
        return name in names_to_skip or isinstance(module, classes_to_skip)
