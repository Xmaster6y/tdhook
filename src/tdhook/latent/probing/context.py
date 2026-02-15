from typing import Callable, Optional, List, Any, Type, Protocol

from tensordict import TensorDict
import torch.nn as nn
from tensordict.nn import TensorDictModuleBase

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import (
    MultiHookManager,
    HookFactory,
    HookDirection,
    MultiHookHandle,
    DIRECTION_TO_RETURN,
)
from tdhook.modules import HookedModule


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

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []
        if self._additional_keys is not None:
            tmp_cache = TensorDict()
            handle, additional_items = module.get(
                cache=tmp_cache,
                module_key="td_module",
                cache_key="_additional_keys",
                callback=lambda **kwargs: kwargs["args"][0].select(*self._additional_keys),
                direction="fwd_pre",
                relative=False,
            )
            handles.append(handle)
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

        for direction in self._directions:
            handles.append(
                self._hook_manager.register_hook(
                    module,
                    (lambda name: hook_factory(name, direction)),
                    direction=direction,
                    relative_path=module.relative_path if self._relative else None,
                )
            )

        return MultiHookHandle(handles)
