"""
Tests for the hooks functionality.
"""

import torch
from tensordict import TensorDict
import pytest

from tdhook.hooks import (
    register_hook_to_module,
    MultiHookManager,
    MultiHookHandle,
    HookFactory,
    EarlyStoppingException,
)


class TestHookRegistration:
    """Test hook registration functionality."""

    def test_register_forward_hook(self, default_test_model):
        """Test registering a forward hook."""

        def forward_hook(module, args, output):
            return output + 1

        input = torch.randn(2, 10)
        original_output = default_test_model(input)
        handle = register_hook_to_module(default_test_model.linear1, forward_hook, direction="fwd")
        assert handle is not None
        model_output = default_test_model(input)
        assert not torch.allclose(model_output, original_output)
        handle.remove()
        output = default_test_model(input)
        assert torch.allclose(output, original_output)

    def test_multi_hook_manager(self, default_test_model):
        """Test MultiHookManager."""

        def hook_factory(name: str):
            def hook(module, args, output):
                return output + 1

            return hook

        input = torch.randn(2, 10)
        original_output = default_test_model(input)

        manager = MultiHookManager(pattern=r"linear\d+")
        handle = manager.register_hook(default_test_model, hook_factory, direction="fwd")
        assert isinstance(handle, MultiHookHandle)
        mod_output = default_test_model(input)
        assert not torch.allclose(mod_output, original_output)
        handle.remove()
        output = default_test_model(input)
        assert torch.allclose(output, original_output)


class TestHookFactory:
    """Test hook factory functionality."""

    def test_make_caching_hook(self, default_test_model):
        """Test making a caching hook."""
        cache = TensorDict()
        hook = HookFactory.make_caching_hook("key", cache)
        assert hook is not None
        hook(default_test_model, None, 1)
        assert cache["key"] == 1

    def test_make_setting_hook(self, default_test_model):
        """Test making a setting hook."""

        def callback(value, module, args, output):
            return value + 1

        hook = HookFactory.make_setting_hook(1, callback=callback)
        assert hook is not None
        output = hook(default_test_model, None, 1)
        assert output == 2

    def test_make_stopping_hook(self, default_test_model):
        """Test making a stopping hook."""

        hook = HookFactory.make_stopping_hook("key")
        assert hook is not None
        with pytest.raises(EarlyStoppingException):
            hook(default_test_model, None, 1)
