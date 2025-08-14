"""
Tests for the activation caching functionality.
"""

import torch

from tdhook.acteng.activation_caching import ActivationCaching


class TestActivationCaching:
    """Test the ActivationCaching class."""

    def test_activation_caching_context_creation(self, default_test_model):
        """Test creating a ActivationCaching."""

        context = ActivationCaching("td_module\.module\.linear2")
        assert isinstance(context, ActivationCaching)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "td_module.module.linear2" in context.cache

    def test_activation_caching_context_creation_relative(self, default_test_model):
        """Test creating a ActivationCaching with relative naming."""

        context = ActivationCaching("linear2", relative=True)
        assert isinstance(context, ActivationCaching)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "linear2" in context.cache
