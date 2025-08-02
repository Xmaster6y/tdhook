"""
Tests for the module functionality.
"""

import torch
from tensordict import TensorDict
import pytest

from tdhook.module import HookedModule


class TestHookedModule:
    """Test the HookedModule class."""

    def test_hooked_module_creation(self, default_test_model):
        """Test creating a HookedModule from a regular model."""
        hooked_module = HookedModule(default_test_model, in_keys=["input"], out_keys=["output"])
        td_output = hooked_module(TensorDict({"input": torch.randn(2, 3, 10)}, batch_size=[2, 3]))
        assert td_output["output"].shape == (2, 3, 5)
        assert torch.allclose(td_output["output"], default_test_model(td_output["input"]))

    def test_hooked_module_run(self, default_test_model):
        """Test creating a HookedModuleRun."""
        hooked_module = HookedModule(default_test_model, in_keys=["input"], out_keys=["output"])
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with hooked_module.run(data):
            pass

        assert data["output"].shape == (2, 5)

    def test_cache_proxy(self, default_test_model):
        """Test the CacheProxy class."""
        hooked_module = HookedModule(default_test_model, in_keys=["input"], out_keys=["output"])
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with hooked_module.run(data) as run:
            proxy = run.get("module")
            with pytest.raises(ValueError):
                proxy.resolve()

        assert torch.allclose(proxy.resolve(), data["output"])
