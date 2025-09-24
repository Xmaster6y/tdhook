"""
Pruning
"""

from typing import Callable, Optional, List, Dict, Tuple
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict import TensorDict
from torch.nn.utils import prune

from tdhook.contexts import HookingContextFactory, HookingContext
from tdhook.modules import resolve_submodule_path

from tdhook._types import UnraveledKey


class PruningContext(HookingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._old_weights = None

    def __enter__(self):
        self._old_weights = TensorDict.from_module(self._module).clone()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._old_weights.to_module(self._module, inplace=True)
        self._old_weights = None


class Pruning(HookingContextFactory):
    _hooking_context_class = PruningContext

    def __init__(
        self,
        importance_callback: Callable,
        amount_to_prune: Optional[float | int] = None,
        modules_to_prune: Optional[Dict[str, Tuple[int, Optional[float]]]] = None,
        skip_modules: Optional[Callable[[str, nn.Module], bool]] = None,
        relative: bool = True,
        relative_path: Optional[str] = None,
    ):
        if amount_to_prune is None and modules_to_prune is None:
            raise ValueError("`amount_to_prune` is required for global pruning")

        super().__init__()
        self._importance_callback = importance_callback
        self._amount_to_prune = amount_to_prune
        self._modules_to_prune = modules_to_prune
        self._skip_modules = skip_modules
        self._relative = relative
        self._relative_path = relative_path or "module"

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        if self._relative:
            root_module = resolve_submodule_path(module, self._relative_path)
        else:
            root_module = module

        if self._modules_to_prune is None:
            parameters_to_prune = []
            importance_scores = {}
            for name, submodule in root_module.named_modules():
                if self._skip_modules and self._skip_modules(name, submodule):
                    continue
                for param_name, param in submodule.named_parameters():
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
                for param_name, param in submodule.named_parameters():
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
        return module

    @staticmethod
    def default_skip(name: str, module: nn.Module) -> bool:
        names_to_skip = ("", "td_module", "module")
        classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModule)
        return name in names_to_skip or isinstance(module, classes_to_skip)
