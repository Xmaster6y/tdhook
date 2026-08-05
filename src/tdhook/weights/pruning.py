from typing import Callable, Optional, Dict, Tuple

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.nn.utils import prune

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import merge_paths
from tdhook.modules import HookedModule
from tdhook.paths import resolve_submodule_path
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec


class Pruning(HookingContextFactory):
    """
    Relevance-based pruning :cite:`Yeom2019PruningBE` and circuit pruning :cite:`Pochinkov2024DissectingLM`.
    """

    def __init__(
        self,
        importance_callback: Callable,
        amount_to_prune: Optional[float | int] = None,
        modules_to_prune: Optional[Dict[str, Tuple[int, Optional[float]]]] = None,
        skip_modules: Optional[Callable[[str, nn.Module], bool]] = None,
        relative_path: Optional[str] = None,
    ):
        if amount_to_prune is None and modules_to_prune is None:
            raise ValueError("`amount_to_prune` is required for global pruning")

        super().__init__()
        self._importance_callback = importance_callback
        self._amount_to_prune = amount_to_prune
        self._modules_to_prune = modules_to_prune
        self._skip_modules = skip_modules
        self._relative_path = relative_path or ""

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        root_path = merge_paths(module.relative_path, self._relative_path)
        root_module = resolve_submodule_path(module, root_path)
        if self._pruning_reparameterizations(root_module):
            raise ValueError("Pruning does not accept parameters that are already reparameterized")
        old_weights = TensorDict.from_module(module.td_module).clone()
        existing_reparameterizations = self._pruning_reparameterizations(module.td_module)

        with HookProgramBuilder() as program:
            program.record(
                HookSpec(self._relative_path, "prune_parameters", None),
                lambda: self._restore_parameters(
                    module.td_module,
                    old_weights,
                    existing_reparameterizations,
                ),
            )
            self._apply_pruning(root_module)
            return program.build()

    @staticmethod
    def _pruning_reparameterizations(module: nn.Module) -> set[tuple[nn.Module, str]]:
        return {
            (submodule, buffer_name.removesuffix("_mask"))
            for submodule in module.modules()
            for buffer_name, _ in submodule.named_buffers(recurse=False)
            if buffer_name.endswith("_mask") and hasattr(submodule, f"{buffer_name.removesuffix('_mask')}_orig")
        }

    @classmethod
    def _restore_parameters(
        cls,
        module: nn.Module,
        old_weights: TensorDict,
        existing_reparameterizations: set[tuple[nn.Module, str]],
    ) -> None:
        error = None
        introduced = cls._pruning_reparameterizations(module) - existing_reparameterizations
        for submodule, parameter_name in introduced:
            try:
                prune.remove(submodule, parameter_name)
            except BaseException as exc:
                error = error or exc
        try:
            old_weights.to_module(module, inplace=True)
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error

    def _apply_pruning(self, root_module: nn.Module) -> None:
        if self._modules_to_prune is None:
            parameters_to_prune = []
            importance_scores = {}
            for name, submodule in root_module.named_modules():
                if self._skip_modules and self._skip_modules(name, submodule):
                    continue
                for param_name, param in submodule.named_parameters(recurse=False):
                    importance_score = self._importance_callback(
                        module_key=name, parameter_name=param_name, parameter=param
                    )
                    if importance_score is not None:
                        importance_scores[(submodule, param_name)] = importance_score
                        parameters_to_prune.append((submodule, param_name))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                importance_scores=importance_scores,
                amount=self._amount_to_prune,
            )
            for submodule, param_name in parameters_to_prune:
                prune.remove(submodule, param_name)
        else:
            to_prune = []
            for module_key, (dim, amount) in self._modules_to_prune.items():
                amount = amount or self._amount_to_prune
                submodule = resolve_submodule_path(root_module, module_key)
                for param_name, param in submodule.named_parameters(recurse=False):
                    importance_scores = self._importance_callback(
                        module_key=module_key, parameter_name=param_name, parameter=param
                    )
                    if importance_scores is not None:
                        to_prune.append((submodule, param_name, amount, dim, importance_scores))

            for submodule, param_name, amount, dim, importance_scores in to_prune:
                prune.ln_structured(
                    submodule,
                    param_name,
                    amount=amount,
                    dim=dim,
                    importance_scores=importance_scores,
                    n=1,
                )
                prune.remove(submodule, param_name)

    @staticmethod
    def default_skip(name: str, module: nn.Module) -> bool:
        names_to_skip = ("", "td_module", "module")
        classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModule)
        return name in names_to_skip or isinstance(module, classes_to_skip)
