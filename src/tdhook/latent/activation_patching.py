from typing import Callable, Optional, List

from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from tdhook.contexts import HookingContextFactory
from tdhook.execution import ExecutionSpec
from tdhook.hooks import CacheProxy, HookFactory
from tdhook.modules import HookedModule, ModuleCallWithCache, IntermediateKeysCleaner, ModuleCall
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook.session import HookSession
from tdhook.targets import Target
from tdhook._types import UnraveledKey


def _activation_target(value: str | Target) -> tuple[str, Target | None]:
    if isinstance(value, str):
        return value, None
    if not isinstance(value, Target):
        raise TypeError("modules_to_patch entries must be module paths or Targets")
    if value.kind != "activation":
        raise ValueError("prepared activation patching requires activation Targets")
    return value.module_path, value


def _selected_output(value: object, target: Target | None):
    if target is None:
        return value
    return target._select(HookSession._hook_tensor(value, target.output_path))


def _replace_selected_output(value: object, replacement: object, target: Target | None):
    if target is None:
        return replacement
    selected = HookSession._hook_tensor(value, target.output_path).clone()
    target._assign(selected, replacement)
    return HookSession._replace_hook_tensor(value, target.output_path, selected)


class ActivationPatching(HookingContextFactory):
    """
    Causal mediation analysis :cite:`Vig2020InvestigatingGB` and latent editing :cite:`belrose2023leace,Dreyer2023FromHT`.
    """

    def __init__(
        self,
        modules_to_patch: List[str | Target],
        patch_key: UnraveledKey = "patched",
        clean_intermediate_keys: bool = True,
        patch_fn: Optional[Callable] = None,
        cache_callback: Optional[Callable] = None,
    ):
        super().__init__()

        self._targets_to_patch = [_activation_target(value) for value in modules_to_patch]
        self._modules_to_patch = [path for path, _target in self._targets_to_patch]
        self._patch_key = patch_key
        self._clean_intermediate_keys = clean_intermediate_keys
        self._patch_fn = patch_fn
        self._cache_callback = cache_callback

        self._hooked_module_kwargs["relative_path"] = "td_module.module[0]._td_module"

    @property
    def execution_spec(self) -> ExecutionSpec:
        return ExecutionSpec(model_passes=2)

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
            for module_key, target in self._targets_to_patch:
                proxy = CacheProxy(module_key, cache_ref)

                def capture_callback(*, _target=target, **kwargs):
                    output = kwargs["output"]
                    if self._cache_callback is not None:
                        output = self._cache_callback(**kwargs)
                    return _selected_output(output, _target)

                program.register_path(
                    module,
                    HookFactory.make_caching_hook(
                        module_key,
                        cache_ref,
                        callback=capture_callback if target is not None else self._cache_callback,
                    ),
                    HookSpec(module_key, "capture", "fwd", target=target),
                    relative_path=module.relative_path,
                )

                def callback(*, _module_key=module_key, _target=target, **kwargs):
                    value = kwargs["value"]
                    output = kwargs["output"]
                    if value is None:  # clean run
                        return output
                    selected_output = _selected_output(output, _target)
                    if self._patch_fn is not None:
                        patched_output = self._patch_fn(
                            module_key=_module_key,
                            output=selected_output,
                            output_to_patch=value,
                        )
                        replacement = value if patched_output is None else patched_output
                    else:
                        replacement = value
                    return _replace_selected_output(output, replacement, _target)

                program.register_path(
                    module,
                    HookFactory.make_setting_hook(proxy, callback=callback),
                    HookSpec(module_key, "replace", "fwd", prepend=True, target=target),
                    relative_path=module.relative_path,
                )
            return program.build()
