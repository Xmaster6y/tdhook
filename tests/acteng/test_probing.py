"""
Tests for the probing functionality.
"""

import torch.nn as nn
import torch

from tdhook.acteng.linear_probing import LinearProbing


class TestLinearProbing:
    """Test the LinearProbing class."""

    def test_probing_context_creation(self, default_test_model):
        """Test creating a LinearProbing."""

        def probe_factory(key, direction):
            return nn.Linear(20, 1)

        context = LinearProbing("td_module\.module\.linear2", probe_factory, relative=False)
        assert isinstance(context, LinearProbing)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "td_module.module.linear2" in context.cache

    def test_probing_context_creation_relative(self, default_test_model):
        """Test creating a LinearProbing with relative naming."""

        def probe_factory(key, direction):
            return nn.Linear(20, 1)

        context = LinearProbing("linear2", probe_factory, relative=True)
        assert isinstance(context, LinearProbing)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "linear2" in context.cache
