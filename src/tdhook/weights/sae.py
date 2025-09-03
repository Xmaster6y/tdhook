"""
SAE Insertion
"""

from typing import Callable, Optional, List
from torch import nn

from tdhook.contexts import HookingContextFactory
from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle, HookDirection


class SAEInsertion(HookingContextFactory):
    def __init__(
        self,
        key_pattern: str,
        sae_factory: Callable[[str, str], nn.Module],
        relative: bool = True,
        directions: Optional[List[HookDirection]] = None,
        additional_keys: Optional[List[str]] = None,
        submodules_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self._key_pattern = key_pattern
        self._sae_factory = sae_factory
        self._relative = relative
        self._directions = directions or ["fwd"]
        self._additional_keys = additional_keys
        self._submodules_paths = submodules_paths

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return MultiHookHandle([])
