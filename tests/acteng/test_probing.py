"""
Tests for the latents functionality.
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

        context = LinearProbing("module\.linear2", probe_factory)
        assert isinstance(context, LinearProbing)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "module.linear2" in context.cache
