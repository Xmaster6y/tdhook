from typing import Callable, Optional, List

from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey

from tdhook.contexts import HookingContextFactory
from tdhook.execution import ExecutionSpec
from tdhook.hooks import HookFactory, register_hook_to_module
from tdhook.latent._targets import activation_target
from tdhook.modules import HookedModule, ModuleCallWithCache, IntermediateKeysCleaner, ModuleCall
from tdhook.runtime import BoundHookProgram, CaptureSource, HookProgramBuilder, HookSpec
from tdhook.targets import Target


def _selected_output(value: object, target: Target | None):
    if target is None:
        return value
    return target.select_output(value)


def _replace_selected_output(value: object, replacement: object, target: Target | None):
    if target is None:
        return replacement
    return target.replace_output(value, replacement)


class ActivationPatching(HookingContextFactory):
    """
    Causal mediation analysis :cite:`Vig2020InvestigatingGB` and latent editing :cite:`belrose2023leace,Dreyer2023FromHT`.
    """

    def __init__(
        self,
        modules_to_patch: List[str | Target],
        patch_key: NestedKey = "patched",
        clean_intermediate_keys: bool = True,
        patch_fn: Optional[Callable] = None,
        cache_callback: Optional[Callable] = None,
    ):
        super().__init__()

        self._targets_to_patch = [activation_target(value, argument="modules_to_patch") for value in modules_to_patch]
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
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModuleBase:
        stored_keys = list(self._modules_to_patch)

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
        captured_outputs = []
        with HookProgramBuilder() as program:

            def reset_captures(_module, _args):
                for captured_output in captured_outputs:
                    captured_output["value"] = None

            reset_handle = register_hook_to_module(module, reset_captures, "fwd_pre", prepend=True)
            program.add_cleanup(reset_handle.remove)

            for module_key, target in self._targets_to_patch:
                captured_output = {"value": None}
                captured_outputs.append(captured_output)

                def capture_callback(*, _target=target, _captured_output=captured_output, **kwargs):
                    output = kwargs["output"]
                    if self._cache_callback is not None:
                        output = self._cache_callback(**kwargs)
                    _captured_output["value"] = _selected_output(output, _target)
                    return _captured_output["value"]

                capture_hook = HookFactory.make_caching_hook(
                    module_key,
                    cache_ref,
                    callback=capture_callback,
                )
                capture_spec = HookSpec(module_key, "capture", "fwd", target=target)
                capture_index = len(program.program.hooks)
                register = program.register_path if target is None else program.register_target
                register(module, capture_hook, capture_spec, relative_path=module.relative_path)

                def callback(
                    *,
                    _module_key=module_key,
                    _target=target,
                    _captured_output=captured_output,
                    **kwargs,
                ):
                    output = kwargs["output"]
                    value = _captured_output["value"]
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

                replace_hook = HookFactory.make_setting_hook(None, callback=callback)
                replace_spec = HookSpec(
                    module_key,
                    "replace",
                    "fwd",
                    prepend=True,
                    target=target,
                    source=CaptureSource(capture_index, detach=False),
                )
                register(module, replace_hook, replace_spec, relative_path=module.relative_path)
            return program.build()
