"""
Gradient attribution
"""

from contextlib import contextmanager
from typing import Callable, Optional, Generator, List

import torch
from torch import nn

from tdhook.contexts import BaseContext
from tdhook.module import HookedModule


class GradientAttribution(BaseContext):
    def __init__(
        self,
        init_target: Optional[Callable] = None,
        init_grad: Optional[Callable] = None,
        write_attr_to_input: bool = True,
    ):
        self._init_target = init_target
        self._init_grad = init_grad
        self._write_attr_to_input = write_attr_to_input

    def prepare(
        self,
        module: nn.Module,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ) -> Generator[HookedModule, None, None]:
        """
        Prepare the module for execution.
        """
        if in_keys is None:
            in_keys = ["input"]
        if self._write_attr_to_input:
            out_keys = (out_keys or ["output"]) + [f"{in_key}_attr" for in_key in in_keys]
        return super().prepare(module, in_keys, out_keys)

    @contextmanager
    def _hook_module(self, module: HookedModule) -> Generator[None, None, None]:
        handles = []

        def input_grad_hook(module, args):
            for arg in args:
                arg.requires_grad = True

        handles.append(
            module.register_submodule_hook(
                "module",
                input_grad_hook,
                direction="fwd_pre",
            )
        )

        def output_backward_hook(module, args, output):
            if self._init_target is not None:
                target = self._init_target(output)
            else:
                target = output
            if self._init_grad is not None:
                init_grad = self._init_grad(output)
            else:
                init_grad = torch.ones_like(target)
            target.backward(init_grad)
            if self._write_attr_to_input:
                return output, *self._grad_attr(args, output)

        handles.append(
            module.register_submodule_hook(
                "module",
                output_backward_hook,
                direction="fwd",
            )
        )
        try:
            with torch.set_grad_enabled(True):
                yield
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _grad_attr(args, output):
        raise NotImplementedError("Gradient attribution is not implemented")
