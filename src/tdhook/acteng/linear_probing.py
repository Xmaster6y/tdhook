"""
Linear probing
"""

import torch.nn as nn
from typing import Callable

from tdhook.contexts import BaseContext
from tdhook.hooks import MultiHookManager


class ProbingContext(BaseContext):
    def __init__(self, key_pattern: str, get_probe: Callable[[str], nn.Module]):
        self._get_probe = get_probe
        self._hook_manager = MultiHookManager(key_pattern)

    def train(self):
        pass
