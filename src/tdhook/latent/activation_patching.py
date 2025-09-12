"""
Activation Patching
"""

from typing import Callable, Optional, List

from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookHandle
from tdhook.modules import HookedModule, ModuleCallWithCache, IntermediateKeysCleaner, ModuleCall
from tdhook._types import UnraveledKey


class ActivationPatching(HookingContextFactory):
    def __init__(
        self,
        modules_to_patch: List[str],
        patch_key: UnraveledKey = "patch",
        clean_intermediate_keys: bool = True,
        patch_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self._modules_to_patch = modules_to_patch
        self._patch_key = patch_key
        self._clean_intermediate_keys = clean_intermediate_keys
        self._patch_fn = patch_fn

        self._hooked_module_kwargs["relative_path"] = "td_module.module[0]._td_module"

    def _prepare_module(
        self, module: TensorDictModuleBase, in_keys: List[UnraveledKey], out_keys: List[UnraveledKey]
    ) -> TensorDictModuleBase:
        stored_keys = [f"{m}_output" for m in self._modules_to_patch]

        modules = [
            ModuleCallWithCache(
                module,
                cache_key="_cache",
                out_key=None,
                stored_keys=stored_keys,
            ),
            ModuleCall(
                module,
                in_key=self._patch_key,
                out_key=self._patch_key,
            ),
        ]
        if self._clean_intermediate_keys:
            modules.append(IntermediateKeysCleaner(intermediate_keys=["_cache"]))
        return TensorDictSequential(*modules)

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        cache_ref = module.td_module[0].cache_ref
        handles = []
        for module_key in self._modules_to_patch:
            handle, proxy = module.get(
                cache=cache_ref,
                cache_key=module_key,
                module_key=module_key,
            )
            handles.append(handle)

            def callback(**kwargs):
                nonlocal module_key, self
                value = kwargs["value"]
                output = kwargs["output"]
                if value is None:  # clean run
                    return output
                elif self._patch_fn is not None:
                    patched_output = self._patch_fn(module_key=module_key, output=output, output_to_patch=value)
                    return value if patched_output is None else patched_output
                else:
                    return value

            handle = module.set(
                module_key=module_key,
                value=proxy,
                callback=callback,
                direction="fwd",
                prepend=True,
            )
            handles.append(handle)
        return MultiHookHandle(handles)
