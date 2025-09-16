"""
Tests for the activation patching functionality.
"""

import pytest
import torch

from tensordict import TensorDict

from tdhook.latent.activation_patching import ActivationPatching


class TestActivationPatching:
    """Test the ActivationPatching class."""

    @pytest.mark.parametrize(
        "modules_to_patch",
        (
            ("linear2",),
            ("linear2", "linear3"),
        ),
    )
    def test_simple_activation_patching(self, default_test_model, modules_to_patch):
        """Test creating a ActivationPatching."""

        def patch_fn(output, output_to_patch, **_):
            output[:, 0] = output_to_patch[:, 0]
            return output

        context = ActivationPatching(modules_to_patch, patch_fn=patch_fn)

        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10), ("patched", "input"): torch.randn(2, 10)}, batch_size=2)
            data = hooked_module(data)
            assert data.get(("patched", "output")).shape == (2, 5)
            assert not torch.allclose(data.get("output"), data.get(("patched", "output")))
