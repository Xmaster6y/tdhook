from typing import Callable, Optional, List

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey

from tdhook.contexts import HookingContextFactory
from tdhook.execution import ExecutionSpec
from tdhook.hooks import HookFactory, MutableWeakRef
from tdhook.latent._targets import activation_target
from tdhook.modules import HookedModule, IntermediateKeysCleaner, ModuleCallWithCache, FunctionModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook.targets import Target


def _selected_output(value: object, target: Target | None):
    if target is None:
        return value
    return target.select_output(value)


def _replace_selected_output(value: object, replacement: object, target: Target | None):
    if target is None:
        return replacement
    return target.replace_output(value, replacement)


class SteeringVectors(HookingContextFactory):
    """
    Steering vectors :cite:`rimsky2023steering`.
    """

    def __init__(
        self,
        modules_to_steer: List[str | Target],
        steer_fn: Callable,
    ):
        super().__init__()

        self._targets_to_steer = [activation_target(value, argument="modules_to_steer") for value in modules_to_steer]
        self._modules_to_steer = [path for path, _target in self._targets_to_steer]
        self._steer_fn = steer_fn

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        with HookProgramBuilder() as program:
            for module_key, target in self._targets_to_steer:

                def callback(*, _module_key=module_key, _target=target, **kwargs):
                    output = kwargs["output"]
                    replacement = self._steer_fn(module_key=_module_key, output=_selected_output(output, _target))
                    return _replace_selected_output(output, replacement, _target)

                hook = HookFactory.make_setting_hook(None, callback=callback)
                spec = HookSpec(module_key, "replace", "fwd", target=target)
                register = program.register_path if target is None else program.register_target
                register(module, hook, spec, relative_path=module.relative_path)
            return program.build()


class ActivationAddition(HookingContextFactory):
    def __init__(
        self,
        modules_to_steer: List[str | Target],
        positive_key: NestedKey = "positive",
        negative_key: NestedKey = "negative",
        steer_key: NestedKey = "steer",
        clean_intermediate_keys: bool = True,
        cache_callback: Optional[Callable] = None,
    ):
        super().__init__()

        self._targets_to_steer = [activation_target(value, argument="modules_to_steer") for value in modules_to_steer]
        self._modules_to_steer = [path for path, _target in self._targets_to_steer]
        self._positive_key = positive_key
        self._negative_key = negative_key
        self._steer_key = steer_key
        self._clean_intermediate_keys = clean_intermediate_keys
        self._cache_callback = cache_callback

        self._hooked_module_kwargs["relative_path"] = "td_module.module[0]._td_module"

    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(model_passes=2)

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        stored_keys = list(self._modules_to_steer)
        positive_keys = [(self._positive_key, key) for key in stored_keys]
        negative_keys = [(self._negative_key, key) for key in stored_keys]
        steer_keys = [(self._steer_key, key) for key in stored_keys]

        cache_ref = MutableWeakRef(TensorDict())
        modules = [
            ModuleCallWithCache(
                module,
                cache_key="_positive_cache",
                in_key=self._positive_key,
                out_key="_positive_out",
                cache_ref=cache_ref,
                stored_keys=stored_keys,
            ),
            ModuleCallWithCache(
                module,
                cache_key="_negative_cache",
                in_key=self._negative_key,
                out_key="_negative_out",
                cache_ref=cache_ref,
                stored_keys=stored_keys,
            ),
            FunctionModule(
                self._compute_steering_vectors,
                in_keys=positive_keys + negative_keys,
                out_keys=steer_keys,
            ),
        ]
        if self._clean_intermediate_keys:
            modules.append(
                IntermediateKeysCleaner(
                    intermediate_keys=["_positive_cache", "_positive_out", "_negative_cache", "_negative_out"]
                )
            )
        return TensorDictSequential(*modules)

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        cache_ref = module.td_module[0].cache_ref
        with HookProgramBuilder() as program:
            for module_key, target in self._targets_to_steer:

                def capture_callback(*, _target=target, **kwargs):
                    output = kwargs["output"]
                    if self._cache_callback is not None:
                        output = self._cache_callback(**kwargs)
                    return _selected_output(output, _target)

                hook = HookFactory.make_caching_hook(
                    module_key,
                    cache_ref,
                    callback=capture_callback if target is not None else self._cache_callback,
                )
                spec = HookSpec(module_key, "capture", "fwd", target=target)
                register = program.register_path if target is None else program.register_target
                register(module, hook, spec, relative_path=module.relative_path)
            return program.build()

    def _compute_steering_vectors(self, td: TensorDict) -> TensorDict:
        positive_outputs = td["_positive_cache"]
        negative_outputs = td["_negative_cache"]
        steering_vectors = (positive_outputs - negative_outputs).mean(dim=tuple(range(td.dim())))
        return TensorDict({self._steer_key: steering_vectors}, device=td.device)
