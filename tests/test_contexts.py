"""
Tests for the contexts functionality.
"""

import torch
from tensordict import TensorDict
from contextlib import contextmanager
from typing import Generator

from tdhook.contexts import BaseContext, CompositeContext
from tdhook.module import HookedModule


class Context1(BaseContext):
    @contextmanager
    def _hook_module(self, module: HookedModule) -> Generator[None, None, None]:
        handle = module.register_submodule_hook(
            key="module",
            hook=lambda module, args, output: output + 1,
            direction="fwd",
        )
        try:
            yield
        finally:
            handle.remove()


class Context2(BaseContext):
    @contextmanager
    def _hook_module(self, module: HookedModule) -> Generator[None, None, None]:
        handle = module.register_submodule_hook(
            key="module",
            hook=lambda module, args, output: output * 2,
            direction="fwd",
        )
        try:
            yield
        finally:
            handle.remove()


class TestBaseContext:
    """Test the BaseContext class."""

    def test_context1(self, default_test_model):
        """Test BaseContext."""
        input = torch.randn(2, 3, 10)
        original_output = default_test_model(input)
        with Context1().prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": input}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert torch.allclose(data["output"], original_output + 1)

    def test_context2(self, default_test_model):
        """Test BaseContext."""
        input = torch.randn(2, 3, 10)
        original_output = default_test_model(input)
        with Context2().prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": input}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert torch.allclose(data["output"], original_output * 2)


class TestCompositeContext:
    """Test the CompositeContext class."""

    def test_composite_context(self, default_test_model):
        """Test preparing a regular module with CompositeContext."""
        input = torch.randn(2, 3, 10)
        original_output = default_test_model(input)
        context = CompositeContext(Context1(), Context2())
        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": input}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert torch.allclose(data["output"], (original_output + 1) * 2)
