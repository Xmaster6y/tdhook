"""
Steering Vectors
"""

from typing import Callable, Optional, List

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookHandle
from tdhook.modules import HookedModule, IntermediateKeysCleaner, ModuleCallWithCache, FunctionModule
from tdhook._types import UnraveledKey
from tdhook.hooks import MutableWeakRef


class SteeringVectors(HookingContextFactory):
    def __init__(
        self,
        modules_to_steer: List[str],
        steer_fn: Callable,
    ):
        super().__init__()

        self._modules_to_steer = modules_to_steer
        self._steer_fn = steer_fn

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []
        for module_key in self._modules_to_steer:

            def callback(**kwargs):
                nonlocal module_key, self
                output = kwargs["output"]
                return self._steer_fn(module_key=module_key, output=output)

            handle = module.set(
                module_key=module_key,
                value=None,
                callback=callback,
                direction="fwd",
            )
            handles.append(handle)
        return MultiHookHandle(handles)


class ActivationAddition(HookingContextFactory):
    def __init__(
        self,
        modules_to_steer: List[str],
        positive_key: UnraveledKey = "positive",
        negative_key: UnraveledKey = "negative",
        steer_key: UnraveledKey = "steer",
        clean_intermediate_keys: bool = True,
        cache_callback: Optional[Callable] = None,
    ):
        super().__init__()

        self._modules_to_steer = modules_to_steer
        self._positive_key = positive_key
        self._negative_key = negative_key
        self._steer_key = steer_key
        self._clean_intermediate_keys = clean_intermediate_keys
        self._cache_callback = cache_callback

        self._hooked_module_kwargs["relative_path"] = "td_module.module[0]._td_module"

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        stored_keys = [f"{m}_output" for m in self._modules_to_steer]
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

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        cache_ref = module.td_module[0].cache_ref
        handles = []
        for module_key in self._modules_to_steer:
            handle, _ = module.get(
                cache=cache_ref,
                cache_key=module_key,
                module_key=module_key,
                callback=self._cache_callback,
            )
            handles.append(handle)

        return MultiHookHandle(handles)

    def _compute_steering_vectors(self, td: TensorDict) -> TensorDict:
        positive_outputs = td["_positive_cache"]
        negative_outputs = td["_negative_cache"]
        steering_vectors = (positive_outputs - negative_outputs).mean(dim=tuple(range(td.dim())))
        return TensorDict({self._steer_key: steering_vectors}, device=td.device)
