"""
Tests for the activation patching functionality.
"""

import pytest
import torch

from tensordict import TensorDict

from tdhook.latent.activation_patching import ActivationPatching
from tdhook.runtime import HookProgram, HookSpec
from tdhook.targets import Target


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

        patched_modules = []

        def patch_fn(module_key, output, output_to_patch):
            patched_modules.append(module_key)
            output[:, 0] = output_to_patch[:, 0]
            return output

        context = ActivationPatching(modules_to_patch, patch_fn=patch_fn)

        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10), ("patched", "input"): torch.randn(2, 10)}, batch_size=2)
            data = hooked_module(data)
            assert data.get(("patched", "output")).shape == (2, 5)
            assert not torch.allclose(data.get("output"), data.get(("patched", "output")))

        assert patched_modules == list(modules_to_patch)
        assert hooked_module.hooking_context.program == HookProgram(
            tuple(
                spec
                for module_key in modules_to_patch
                for spec in (
                    HookSpec(module_key, "capture", "fwd"),
                    HookSpec(module_key, "replace", "fwd", prepend=True),
                )
            )
        )
        assert all(not submodule._forward_hooks for submodule in default_test_model.modules())

    def test_target_selection_is_executed_and_reported(self, default_test_model):
        target = Target("linear2", "activation", -1, (0,))
        context = ActivationPatching([target])
        data = TensorDict(
            {"input": torch.randn(2, 10), ("patched", "input"): torch.randn(2, 10)},
            batch_size=2,
        )

        with context.prepare(default_test_model) as hooked_module:
            result = hooked_module(data)

        assert hooked_module.hooking_context.program == HookProgram(
            (
                HookSpec("linear2", "capture", "fwd", target=target),
                HookSpec("linear2", "replace", "fwd", prepend=True, target=target),
            )
        )
        assert result["output"].shape == result["patched", "output"].shape
        assert context.execution_spec.model_passes == 2

    def test_target_patching_changes_only_selected_units(self):
        model = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
        target = Target("", "activation", -1, (0,))
        data = TensorDict(
            {
                "input": torch.tensor([[1.0, 2.0, 3.0]]),
                ("patched", "input"): torch.tensor([[10.0, 20.0, 30.0]]),
            },
            batch_size=[1],
        )

        with ActivationPatching([target]).prepare(model) as prepared:
            result = prepared(data)

        torch.testing.assert_close(result["output"], torch.tensor([[1.0, 2.0, 3.0]]))
        torch.testing.assert_close(result["patched", "output"], torch.tensor([[1.0, 20.0, 30.0]]))

    @pytest.mark.parametrize(
        "value,exception",
        [
            (object(), TypeError),
            (Target("", "gradient", -1, (0,)), ValueError),
        ],
    )
    def test_prepared_activation_targets_are_validated(self, value, exception):
        with pytest.raises(exception, match="activation|module paths"):
            ActivationPatching([value])
