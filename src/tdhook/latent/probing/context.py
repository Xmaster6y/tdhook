from typing import Callable, Optional, List, Any, Type, Protocol

from tensordict import TensorDict
import torch.nn as nn
from tensordict.nn import TensorDictModuleBase

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import (
    CacheProxy,
    MultiHookManager,
    HookFactory,
    HookDirection,
    DIRECTION_TO_RETURN,
)
from tdhook.modules import HookedModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec


class Probe(Protocol):
    def step(self, data: Any, **kwargs) -> Any: ...


class Probing(HookingContextFactory):
    """
    Linear probing :cite:`alain2018understanding` and concept activation vectors :cite:`kim2018interpretability`.
    """

    default_classes_to_hook = (nn.Module,)
    default_classes_to_skip = (nn.ModuleList, nn.Sequential, TensorDictModuleBase)

    def __init__(
        self,
        key_pattern: str,
        probe_factory: Callable[[str, str], Probe],
        relative: bool = True,
        directions: Optional[List[HookDirection]] = None,
        additional_keys: Optional[List[str]] = None,
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
        self._additional_keys = additional_keys

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        if self._additional_keys is not None:
            tmp_cache = TensorDict()
            additional_items = CacheProxy("_additional_keys", tmp_cache)
        else:
            additional_items = None

        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self, additional_items
            probe = self._probe_factory(name, direction)

            def callback(**kwargs):
                nonlocal additional_items
                if additional_items is not None:
                    _additional_items = additional_items.resolve()
                else:
                    _additional_items = {}
                return probe.step(kwargs[DIRECTION_TO_RETURN[direction]], **_additional_items)

            return HookFactory.make_reading_hook(callback=callback, direction=direction)

        with HookProgramBuilder() as program:
            if self._additional_keys is not None:
                program.register(
                    module.td_module,
                    HookFactory.make_caching_hook(
                        "_additional_keys",
                        tmp_cache,
                        callback=lambda **kwargs: kwargs["args"][0].select(*self._additional_keys),
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
