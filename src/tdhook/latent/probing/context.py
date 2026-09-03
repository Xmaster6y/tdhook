from typing import Callable, Optional, List, Any, Type, Protocol

import torch.nn as nn
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import (
    MultiHookManager,
    HookFactory,
    HookDirection,
    DIRECTION_TO_RETURN,
    register_hook_to_module,
)
from tdhook.modules import HookedModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook._types import is_nested_key


class Probe(Protocol):
    def step(self, data: Any, **kwargs) -> Any: ...


class _ProbingHookedModule(HookedModule):
    """Expose probe metadata as native inputs without mutating the model contract."""

    def __init__(self, *args, additional_keys: list[NestedKey], **kwargs):
        super().__init__(*args, **kwargs)
        self._probing_in_keys = list(dict.fromkeys([*self.td_module.in_keys, *additional_keys]))

    @property
    def in_keys(self):
        return self._probing_in_keys


class Probing(HookingContextFactory):
    """
    Linear probing :cite:`alain2018understanding` and concept activation vectors :cite:`kim2018interpretability`.
    """

    default_classes_to_hook = (nn.Module,)
    default_classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModuleBase)
    _hooked_module_class = _ProbingHookedModule

    def __init__(
        self,
        key_pattern: str,
        probe_factory: Callable[[str, str], Probe],
        relative: bool = True,
        directions: Optional[List[HookDirection]] = None,
        additional_keys: Optional[List[NestedKey]] = None,
        classes_to_hook: Optional[List[Type[nn.Module]]] = None,
        classes_to_skip: Optional[List[Type[nn.Module]]] = None,
    ):
        super().__init__()
        self._key_pattern = key_pattern
        classes_to_hook = tuple(classes_to_hook or self.default_classes_to_hook)
        classes_to_skip = tuple(classes_to_skip or self.default_classes_to_skip)
        self._hook_manager = MultiHookManager(key_pattern, classes_to_hook, classes_to_skip)
        self._relative = relative
        self._probe_factory = probe_factory
        self._directions = directions or ["fwd"]
        self._additional_keys = list(additional_keys or [])
        if not all(is_nested_key(key) for key in self._additional_keys):
            raise TypeError("additional_keys must contain TensorDict nested keys")
        self._hooked_module_kwargs["additional_keys"] = self._additional_keys

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        additional_items = None
        forward_active = False
        backward_active = False

        def begin_forward(_module, _args):
            nonlocal additional_items, forward_active
            additional_items = None
            forward_active = True

        def end_forward(_module, _args, _output):
            nonlocal forward_active
            forward_active = False

        def begin_backward(_module, _grad_output):
            nonlocal backward_active
            backward_active = True

        def end_backward(_module, _grad_input, _grad_output):
            nonlocal additional_items, backward_active
            additional_items = None
            backward_active = False

        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self, additional_items
            if module.hooking_context is not None and module.hooking_context.for_inspection:
                probe = None
            else:
                probe = self._probe_factory(name, direction)

            def callback(**kwargs):
                if probe is not None:
                    direction_active = backward_active if direction.startswith("bwd") else forward_active
                    if self._additional_keys and (additional_items is None or not direction_active):
                        raise RuntimeError("probe reached its target before additional inputs were captured")
                    metadata = additional_items if additional_items is not None else {}
                    return probe.step(kwargs[DIRECTION_TO_RETURN[direction]], **metadata)
                return None

            return HookFactory.make_reading_hook(callback=callback, direction=direction)

        with HookProgramBuilder() as program:
            if self._additional_keys:
                begin_forward_handle = register_hook_to_module(module, begin_forward, "fwd_pre", prepend=True)
                program.add_cleanup(begin_forward_handle.remove)
                end_forward_handle = register_hook_to_module(module, end_forward, "fwd")
                program.add_cleanup(end_forward_handle.remove)

                if any(direction.startswith("bwd") for direction in self._directions):
                    model_root = program.resolve_path(module, "", relative_path=module.relative_path)
                    begin_backward_handle = register_hook_to_module(
                        model_root,
                        begin_backward,
                        "bwd_pre",
                        prepend=True,
                    )
                    program.add_cleanup(begin_backward_handle.remove)
                    end_backward_handle = register_hook_to_module(model_root, end_backward, "bwd")
                    program.add_cleanup(end_backward_handle.remove)

                def capture_additional_items(**kwargs):
                    nonlocal additional_items
                    additional_items = kwargs["args"][0].select(*self._additional_keys)

                program.register(
                    module.td_module,
                    HookFactory.make_reading_hook(
                        callback=capture_additional_items,
                        direction="fwd_pre",
                    ),
                    HookSpec("", "capture_inputs", "fwd_pre"),
                )
            for direction in self._directions:
                for name, submodule in self._hook_manager.iter_modules(
                    module,
                    relative_path=module.relative_path if self._relative else None,
                ):
                    program.register(
                        submodule,
                        hook_factory(name, direction),
                        HookSpec(name, "probe", direction),
                    )

            return program.build()
