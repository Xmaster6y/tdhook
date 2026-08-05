from typing import Callable, Optional, List

from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import CacheProxy, HookFactory
from tdhook.modules import HookedModule, ModuleCallWithCache, IntermediateKeysCleaner, ModuleCall
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook._types import UnraveledKey


class ActivationPatching(HookingContextFactory):
    """
    Causal mediation analysis :cite:`Vig2020InvestigatingGB` and latent editing :cite:`belrose2023leace,Dreyer2023FromHT`.
    """

    def __init__(
        self,
        modules_to_patch: List[str],
        patch_key: UnraveledKey = "patched",
        clean_intermediate_keys: bool = True,
        patch_fn: Optional[Callable] = None,
        cache_callback: Optional[Callable] = None,
    ):
        super().__init__()

        self._modules_to_patch = modules_to_patch
        self._patch_key = patch_key
        self._clean_intermediate_keys = clean_intermediate_keys
        self._patch_fn = patch_fn
        self._cache_callback = cache_callback

        self._hooked_module_kwargs["relative_path"] = "td_module.module[0]._td_module"

    def _prepare_module(
        self,
        module: TensorDictModuleBase,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
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

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        cache_ref = module.td_module[0].cache_ref
        with HookProgramBuilder() as program:
            for module_key in self._modules_to_patch:
                proxy = CacheProxy(module_key, cache_ref)
                program.register_path(
                    module,
                    HookFactory.make_caching_hook(
                        module_key,
                        cache_ref,
                        callback=self._cache_callback,
                    ),
                    HookSpec(module_key, "capture", "fwd"),
                    relative_path=module.relative_path,
                )

                def callback(*, _module_key=module_key, **kwargs):
                    value = kwargs["value"]
                    output = kwargs["output"]
                    if value is None:  # clean run
                        return output
                    if self._patch_fn is not None:
                        patched_output = self._patch_fn(
                            module_key=_module_key,
                            output=output,
                            output_to_patch=value,
                        )
                        return value if patched_output is None else patched_output
                    return value

                program.register_path(
                    module,
                    HookFactory.make_setting_hook(proxy, callback=callback),
                    HookSpec(module_key, "replace", "fwd", prepend=True),
                    relative_path=module.relative_path,
                )
            return program.build()
