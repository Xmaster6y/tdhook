"""
Activation Maximisation
"""

from typing import List, Optional, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from tdhook.attribution import Saliency
from tdhook.contexts import HookingContextFactory
from tdhook.modules import PGDModule, IntermediateKeysCleaner
from tdhook._types import UnraveledKey


class ActivationMaximisation(HookingContextFactory):
    def __init__(
        self,
        modules_to_maximise: List[str],
        alpha: float = 0.1,
        n_steps: int = 10,
        min_value: float = -float("Inf"),
        max_value: float = float("Inf"),
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_intermediate_keys: bool = True,
    ):
        super().__init__()
        self._hooking_context_kwargs["pre_factories"] = [
            Saliency(
                use_inputs=True,
                use_outputs=False,
                input_modules=None,
                target_modules=modules_to_maximise,
                init_attr_targets=init_attr_targets,
                init_attr_grads=init_attr_grads,
                additional_init_keys=additional_init_keys,
                attribution_key="_grad",
                clean_intermediate_keys=True,
                absolute=False,
                multiply_by_inputs=False,
            )
        ]

        self._attribution_key = attribution_key
        self._modules_to_maximise = modules_to_maximise
        self._alpha = alpha
        self._n_steps = n_steps
        self._min_value = min_value
        self._max_value = max_value
        self._clean_intermediate_keys = clean_intermediate_keys

        self._hooked_module_kwargs["relative_path"] = "td_module.module[1]._td_module"

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        working_in_keys = [("_working", in_key) for in_key in in_keys]
        attr_keys = [(self._attribution_key, in_key) for in_key in in_keys]

        modules = [
            TensorDictModule(
                lambda *tensors: tensors,
                in_keys=in_keys,
                out_keys=working_in_keys,
            ),
            PGDModule(
                module,
                self._alpha,
                self._n_steps,
                self._min_value,
                self._max_value,
                grad_key="_grad",
                working_key="_working",
                use_sign=False,
                ascent=True,
            ),
            TensorDictModule(
                lambda *tensors: tensors,
                in_keys=working_in_keys,
                out_keys=attr_keys,
            ),
        ]

        if self._clean_intermediate_keys:
            modules.append(
                IntermediateKeysCleaner(
                    intermediate_keys=["_working"],
                )
            )
        return TensorDictSequential(*modules)
