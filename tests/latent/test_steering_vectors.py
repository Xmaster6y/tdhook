"""
Tests for the SteeringVectors class.
"""

import pytest
import torch

from tensordict import TensorDict

from tdhook.latent.steering_vectors import SteeringVectors, ActivationAddition
from tdhook.runtime import HookProgram, HookSpec
from tdhook.targets import Target


class TestSteeringVectors:
    """Test the SteeringVectors class."""

    @pytest.mark.parametrize(
        "modules_to_steer",
        (
            ("linear2",),
            ("linear2", "linear3"),
        ),
    )
    def test_simple_steering_vectors(self, default_test_model, modules_to_steer):
        """Test creating a ActivationPatching."""

        steered_modules = []

        def steer_fn(module_key, output):
            steered_modules.append(module_key)
            output[:, 0] = 0
            return output

        context = SteeringVectors(modules_to_steer, steer_fn=steer_fn)

        with context.bind(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
            data = hooked_module(data)
            assert data.get("output").shape == (2, 5)

        assert steered_modules == list(modules_to_steer)
        assert hooked_module.binding.program == HookProgram(
            tuple(HookSpec(module_key, "replace", "fwd") for module_key in modules_to_steer)
        )
        assert all(not submodule._forward_hooks for submodule in default_test_model.modules())

    def test_target_selection_is_steered_and_reported(self, default_test_model):
        target = Target("linear2", "activation", -1, (0,))
        context = SteeringVectors([target], steer_fn=lambda module_key, output: torch.zeros_like(output))

        with context.bind(default_test_model) as hooked_module:
            hooked_module(TensorDict({"input": torch.randn(2, 10)}, batch_size=2))

        assert hooked_module.binding.program == HookProgram((HookSpec("linear2", "replace", "fwd", target=target),))

    def test_target_steering_changes_only_selected_units(self):
        model = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
        target = Target("", "activation", -1, (1,))

        with SteeringVectors(
            [target],
            steer_fn=lambda module_key, output: torch.zeros_like(output),
        ).bind(model) as prepared:
            result = prepared(TensorDict({"input": torch.tensor([[1.0, 2.0, 3.0]])}, batch_size=[1]))

        torch.testing.assert_close(result["output"], torch.tensor([[1.0, 0.0, 3.0]]))

    def test_target_occurrence_steers_one_repeated_module_call(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Identity()

            def forward(self, value):
                return self.shared(value + 1) + self.shared(value + 2)

        target = Target("shared", "activation", -1, (0,), occurrences=(0,))
        data = TensorDict({"input": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

        with SteeringVectors([target], steer_fn=lambda module_key, output: torch.zeros_like(output)).bind(
            Model()
        ) as prepared:
            result = prepared(data)

        torch.testing.assert_close(result["output"], torch.tensor([[3.0, 7.0]]))


class TestActivationAddition:
    """Test the ActivationAddition class."""

    @pytest.mark.parametrize(
        "modules_to_steer",
        (
            ("linear1",),
            ("linear1", "linear2"),
        ),
    )
    def test_simple_activation_addition(self, default_test_model, modules_to_steer):
        """Test creating a ActivationAddition."""

        context = ActivationAddition(modules_to_steer)

        with context.bind(default_test_model) as hooked_module:
            data = TensorDict(
                {("positive", "input"): torch.randn(2, 10), ("negative", "input"): torch.randn(2, 10)}, batch_size=2
            )
            data = hooked_module(data)
            for module_key in modules_to_steer:
                assert data.get(("steer", module_key)).shape == (20,)

        assert hooked_module.binding.program == HookProgram(
            tuple(HookSpec(module_key, "capture", "fwd") for module_key in modules_to_steer)
        )
        assert all(not submodule._forward_hooks for submodule in default_test_model.modules())
        assert context.execution_spec.model_passes == 2

    def test_target_addition_extracts_only_selected_units(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3, bias=False)

            def forward(self, value):
                return self.linear(value)

        model = Model()
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(3))
        target = Target("linear", "activation", -1, (2,))
        data = TensorDict(
            {
                ("positive", "input"): torch.tensor([[1.0, 2.0, 5.0]]),
                ("negative", "input"): torch.tensor([[1.0, 2.0, 1.0]]),
            },
            batch_size=[1],
        )

        with ActivationAddition([target]).bind(model) as prepared:
            result = prepared(data)

        torch.testing.assert_close(result["steer", "linear"], torch.tensor([4.0]))
        assert ("steer", "linear") in prepared.out_keys

    def test_target_occurrence_extracts_one_repeated_module_call(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Identity()

            def forward(self, value):
                return self.shared(value * 2) + self.shared(value * 3)

        target = Target("shared", "activation", -1, (0,), occurrences=(1,))
        data = TensorDict(
            {
                ("positive", "input"): torch.tensor([[2.0, 0.0]]),
                ("negative", "input"): torch.tensor([[1.0, 0.0]]),
            },
            batch_size=[1],
        )

        with ActivationAddition([target]).bind(Model()) as prepared:
            result = prepared(data)

        torch.testing.assert_close(result["steer", "shared"], torch.tensor([3.0]))
