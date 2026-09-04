from typing import Callable, List, Optional

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey

from tdhook.attribution.saliency import Saliency
from tdhook.methods import Method
from tdhook.execution import ExecutionSpec, GradientMode
from tdhook.modules import IntermediateKeysCleaner, PGDModule


class ActivationMaximisation(Method):
    """
    Activation maximisation :cite:`Mahendran2015VisualizingDC`.
    """

    def __init__(
        self,
        modules_to_maximise: List[str],
        alpha: float = 0.1,
        n_steps: int = 10,
        min_value: float = -float("Inf"),
        max_value: float = float("Inf"),
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[NestedKey]] = None,
        attribution_key: NestedKey = "attr",
        clean_intermediate_keys: bool = True,
    ):
        super().__init__()
        self._binding_kwargs["pre_methods"] = [
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

    @property
    def execution_spec(self) -> ExecutionSpec:
        """Activation maximisation performs one gradient pass per optimisation step."""

        return ExecutionSpec(model_passes=self._n_steps, gradient_mode=GradientMode.REQUIRED)

    def _bind_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
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
