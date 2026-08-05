from typing import Callable, Optional, List, Dict

from torch import nn
from warnings import warn
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict import TensorDict, TensorDictBase

from tdhook.attribution.gradient_helpers import GradientAttribution
from tdhook._types import UnraveledKey, join_keys
from tdhook.attribution.lrp_helpers.rules import Rule
from tdhook.modules import HookedModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec


class _LRPModule(TensorDictModuleBase):
    """Expose LRP's public TensorDict contract without its temporary keys."""

    def __init__(self, module: TensorDictModuleBase, in_keys, out_keys):
        super().__init__()
        self.module = module
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)

    def forward(self, data: TensorDictBase) -> TensorDictBase:
        return self.module(data)

    def __getitem__(self, index):
        return self.module[index]


class LRP(GradientAttribution):
    """
    Different LRP rules such as LRP-0, LRP-epsilon z-plus :cite:`Bach2015OnPE`, flat :cite:`Lapuschkin2019UnmaskingCH`, gamma :cite:`Montavon2019LayerWiseRP,Andol2021LearningDI`, w-square :cite:`Montavon2015ExplainingNC` and its conditional variant :cite:`Achtibat2022FromAM`.
    """

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
        rule_mapper: Callable[[str, nn.Module], Rule | None] | None = None,
        warn_on_missing_rule: bool = True,
        skip_modules: Optional[Callable[[str, nn.Module], bool]] = None,
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
        self._rule_mapper = rule_mapper or (lambda name, module: None)
        self._warn_on_missing_rule = warn_on_missing_rule
        self._skip_modules = skip_modules
        self._hooked_module_kwargs["relative_path"] = "td_module.module.module[2]._td_module"

    def _prepare_module(self, module, in_keys, out_keys, extra_relative_path):
        prepared = super()._prepare_module(module, in_keys, out_keys, extra_relative_path)
        public_in_keys = [*in_keys, *self._additional_init_keys]
        attribution_sources = [*(in_keys if self._use_inputs else ()), *self._input_modules]
        public_out_keys = [*out_keys, *(join_keys(self._attr_key, key) for key in attribution_sources)]
        return _LRPModule(prepared, public_in_keys, public_out_keys)

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        with HookProgramBuilder() as program:
            root = program.resolve_path(module, "", relative_path=module.relative_path)
            for name, child in root.named_modules():
                if self._skip_modules and self._skip_modules(name, child):
                    continue
                rule = self._rule_mapper(name, child)
                if rule is not None:
                    handle = rule.register(child)
                    program.record(HookSpec(name, "apply_rule", None), handle.remove)
                elif self._warn_on_missing_rule:
                    warn(f"No rule found for module `{name}` ({type(child).__name__})")
            self._register_hook_program(module, program)
            return program.build()

    def _grad_attr(
        self,
        grads: TensorDict,
        inputs: TensorDict,
    ) -> TensorDict:
        return grads

    @staticmethod
    def default_skip(name: str, module: nn.Module) -> bool:
        names_to_skip = ("", "td_module", "module")
        classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModule)
        return name in names_to_skip or isinstance(module, classes_to_skip)
