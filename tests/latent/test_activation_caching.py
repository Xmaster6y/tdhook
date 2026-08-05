"""
Tests for the activation caching functionality.
"""

import torch
import shutil
from pathlib import Path
import pytest
from tensordict import TensorDict, MemoryMappedTensor
from tensordict.nn import TensorDictModule

from tdhook.latent.activation_caching import ActivationCaching, ActivationCachingModule
from tdhook.modules import get_best_device
from tdhook.runtime import HookProgram, HookSpec


class TestActivationCaching:
    """Test the ActivationCaching class."""

    def test_activation_caching_context_creation(self, default_test_model):
        """Test creating a ActivationCaching."""

        context = ActivationCaching(r"td_module\.module\.linear2", relative=False)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "td_module.module.linear2" in hooked_module.hooking_context.cache

    def test_activation_caching_context_creation_relative(self, default_test_model):
        """Test creating a ActivationCaching with relative naming."""

        context = ActivationCaching("linear2", relative=True)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "linear2" in hooked_module.hooking_context.cache
        assert hooked_module.hooking_context.program == HookProgram((HookSpec("linear2", "capture", "fwd"),))

    def test_tensordict_execution_publishes_a_native_cache_output(self, default_test_model):
        context = ActivationCaching("linear2", cache_key=("activations", "cache"))
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with context.prepare(default_test_model) as hooked_module:
            result = hooked_module(data)

        assert hooked_module.in_keys == ["input"]
        assert hooked_module.out_keys == ["output", ("activations", "cache")]
        assert result["activations", "cache"]["linear2"].shape == (2, 20)

    def test_published_cache_is_not_cleared_by_a_later_execution(self, default_test_model):
        context = ActivationCaching("linear2")
        first = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
        second = TensorDict({"input": torch.zeros(2, 10)}, batch_size=[2])

        with context.prepare(default_test_model) as hooked_module:
            hooked_module(first)
        first_values = first["cache"]["linear2"].clone()
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(second)

        torch.testing.assert_close(first["cache"]["linear2"], first_values)

    def test_cache_output_can_be_disabled_for_context_only_use(self, default_test_model):
        context = ActivationCaching("linear2", cache_key=None)
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with context.prepare(default_test_model) as hooked_module:
            hooked_module(data)

        assert hooked_module.out_keys == ["output"]
        assert "cache" not in data

    def test_cache_output_key_is_validated_and_cannot_replace_a_model_output(self):
        with pytest.raises(TypeError, match="cache_key"):
            ActivationCaching("", cache_key=object())

        model = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=["cache"])
        with pytest.raises(ValueError, match="collides"):
            with ActivationCaching("", cache_key="cache").prepare(model):
                pass

        nested_output = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=[("result", "value")])
        for cache_key in ("result", ("result", "value", "cache")):
            with pytest.raises(ValueError, match="collides"):
                with ActivationCaching("", cache_key=cache_key).prepare(nested_output):
                    pass

    def test_cache_publication_requires_context_and_key_pattern_is_mutable(self):
        td_module = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=["output"])
        prepared = ActivationCachingModule(td_module, cache_key="cache")
        with pytest.raises(RuntimeError, match="prepared hooking context"):
            prepared.finalize_tensordict(TensorDict({"output": torch.ones(2, 3)}, batch_size=[2]))

        method = ActivationCaching("linear1")
        assert method.key_pattern == "linear1"
        method.key_pattern = "linear2"
        assert method.key_pattern == "linear2"

    def test_different_device_cache(self, default_test_model):
        """Test creating a ActivationCaching with cache on a different device."""

        device = get_best_device()
        cache = TensorDict(device=device)
        context = ActivationCaching("linear2", relative=True, cache=cache)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            output = hooked_module(inputs)
        assert output.device.type == "cpu"
        assert hooked_module.hooking_context.cache["linear2"].device.type == device.type

    def test_memmap_cache(self, default_test_model):
        """Test creating a ActivationCaching with memmap cache."""

        cache = TensorDict()
        context = ActivationCaching("linear2", relative=True, cache=cache)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        path = "results/tests/test_memmap_cache.pt"
        memmap_cache = cache.memmap(path, True)
        assert isinstance(memmap_cache["linear2"], MemoryMappedTensor)

        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                path_obj.unlink()
