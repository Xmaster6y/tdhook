"""
Context
"""

from contextlib import contextmanager, ExitStack
from typing import List, Optional, Generator
from torch import nn
from tensordict.nn import TensorDictModule

from tdhook.module import HookedModule


class BaseContext:
    @contextmanager
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
        if out_keys is None:
            out_keys = ["output"]

        if isinstance(module, TensorDictModule):
            method = self._prepare_td_module
        else:
            method = self._prepare_module
        with method(module) as prep_module:
            if isinstance(module, TensorDictModule):
                hooked_module = HookedModule(
                    prep_module.module,
                    prep_module.in_keys,
                    prep_module.out_keys,
                    inplace=prep_module.inplace,
                    method=prep_module.method,
                    method_kwargs=prep_module.method_kwargs,
                    strict=prep_module.strict,
                    get_kwargs=prep_module._get_kwargs,
                )
            else:
                hooked_module = HookedModule(prep_module, in_keys, out_keys)
            with self._hook_module(hooked_module):
                yield hooked_module

    @contextmanager
    def _prepare_module(
        self,
        module: nn.Module,
    ) -> Generator[nn.Module, None, None]:
        yield module

    @contextmanager
    def _prepare_td_module(
        self,
        module: TensorDictModule,
    ) -> Generator[TensorDictModule, None, None]:
        yield module

    @contextmanager
    def _hook_module(self, module: HookedModule) -> Generator[None, None, None]:
        yield


class CompositeContext(BaseContext):
    def __init__(self, *contexts: BaseContext):
        self._contexts = contexts

    @contextmanager
    def _prepare_module(
        self,
        module: nn.Module,
    ) -> Generator[nn.Module, None, None]:
        with ExitStack() as stack:
            for context in self._contexts:
                module = stack.enter_context(context._prepare_module(module))
            yield module

    @contextmanager
    def _prepare_td_module(
        self,
        module: TensorDictModule,
    ) -> Generator[TensorDictModule, None, None]:
        with ExitStack() as stack:
            for context in self._contexts:
                module = stack.enter_context(context._prepare_td_module(module))
            yield module

    @contextmanager
    def _hook_module(self, module: HookedModule) -> Generator[None, None, None]:
        with ExitStack() as stack:
            for context in self._contexts:
                stack.enter_context(context._hook_module(module))
            yield
