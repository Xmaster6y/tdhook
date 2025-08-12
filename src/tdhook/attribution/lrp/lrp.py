"""
LRP
"""

from typing import Callable, Optional

from torch import nn
from warnings import warn

from tdhook.attribution.gradient_attribution import GradientAttribution

from .rules import Rule


class LRP(GradientAttribution):
    def __init__(
        self,
        rule_mapper: Callable[[str, nn.Module], Rule | None],
        init_target: Optional[Callable] = None,
        init_grad: Optional[Callable] = None,
        warn_on_missing_rule: bool = True,
        skip_modules: Optional[Callable[[str, nn.Module], bool]] = None,
    ):
        super().__init__(init_target, init_grad)
        self._rule_mapper = rule_mapper
        self._warn_on_missing_rule = warn_on_missing_rule
        self._skip_modules = skip_modules

    def _prepare_module(
        self,
        module: nn.Module,
    ) -> nn.Module:
        rule_map = {}
        for name, child in module.named_modules():
            if self._skip_modules and self._skip_modules(name, child):
                continue
            rule = self._rule_mapper(name, child)
            if rule is not None:
                rule.register(child)
                rule_map[name] = rule
            elif self._warn_on_missing_rule:
                warn(f"No rule found for module `{name}`")
        module._rule_map = rule_map
        return module

    def _restore_module(self, module: nn.Module) -> nn.Module:
        for name, child in module.named_modules():
            rule = self._rule_mapper(name, child)
            if rule is not None:
                rule.unregister(child)
        del module._rule_map
        return module

    def _grad_attr(self, args, output):
        return (arg.grad for arg in args)

    @staticmethod
    def skip_root_and_modulelist(name: str, module: nn.Module) -> bool:
        return name == "" or isinstance(module, nn.ModuleList)
