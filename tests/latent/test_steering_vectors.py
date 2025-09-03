"""
Tests for the SteeringVectors class.
"""

import pytest
import torch

from tensordict import TensorDict

from tdhook.latent.steering_vectors import SteeringVectors, ActivationAddition


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

        def steer_fn(output, **_):
            output[:, 0] = 0
            return output

        context = SteeringVectors(modules_to_steer, steer_fn=steer_fn)

        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
            data = hooked_module(data)
            assert data.get("output").shape == (2, 5)


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

        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict(
                {("positive", "input"): torch.randn(2, 10), ("negative", "input"): torch.randn(2, 10)}, batch_size=2
            )
            data = hooked_module(data)
            for module_key in modules_to_steer:
                assert data.get(("steer", module_key)).shape == (20,)
