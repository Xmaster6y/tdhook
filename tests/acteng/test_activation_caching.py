"""
Tests for the activation caching functionality.
"""

import torch
from tensordict import TensorDict, MemoryMappedTensor

from tdhook.latent.activation_caching import ActivationCaching
from tdhook.module import get_best_device


class TestActivationCaching:
    """Test the ActivationCaching class."""

    def test_activation_caching_context_creation(self, default_test_model):
        """Test creating a ActivationCaching."""

        context = ActivationCaching("td_module\.module\.linear2", relative=False)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "td_module.module.linear2" in context.cache

    def test_activation_caching_context_creation_relative(self, default_test_model):
        """Test creating a ActivationCaching with relative naming."""

        context = ActivationCaching("linear2", relative=True)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "linear2" in context.cache

    def test_different_device_cache(self, default_test_model):
        """Test creating a ActivationCaching with cache on a different device."""

        device = get_best_device()
        cache = TensorDict(device=device)
        context = ActivationCaching("linear2", relative=True, cache=cache)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            output = hooked_module(inputs)
        assert output.device.type == "cpu"
        assert context.cache["linear2"].device.type == device.type

    def test_memmap_cache(self, default_test_model):
        """Test creating a ActivationCaching with memmap cache."""

        cache = TensorDict()
        context = ActivationCaching("linear2", relative=True, cache=cache)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        memmap_cache = cache.memmap("results/tests/test_memmap_cache.pt", True)
        assert isinstance(memmap_cache["linear2"], MemoryMappedTensor)
