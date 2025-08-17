"""
Test gradient attribution.
"""

from typing import Tuple

import torch
import torch.nn as nn
import pytest
from tensordict import TensorDict

from captum.attr import (
    IntegratedGradients as CaptumIntegratedGradients,
    Saliency as CaptumSaliency,
    InputXGradient as CaptumInputXGradient,
)
from captum.attr._utils import approximation_methods

from tdhook.attribution.gradient_attribution import helpers
from tdhook.attribution.gradient_attribution import Saliency, IntegratedGradients


def get_sequential_linear_module(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
    )


def get_sequential_conv_module(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv1d(10, 10, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(10, 10, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(-2, -1),
        nn.Linear(50, 1),
    )


def get_multitarget_module(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
    )


class WrapSum(nn.Module):
    def __init__(self, module: nn.Module, targets: Tuple[int, ...]):
        super().__init__()
        self.module = module
        self.targets = targets

    def forward(self, x):
        return self.module(x)[:, self.targets].sum(dim=-1)


class TestGradientAttribution:
    @pytest.mark.parametrize(
        "factory",
        (
            (get_sequential_linear_module, (10,)),
            (get_sequential_conv_module, (10, 5)),
        ),
    )
    @pytest.mark.parametrize(
        "absolute",
        (True, False),
    )
    @pytest.mark.parametrize(
        "batch_size",
        (
            tuple(),
            (3,),
            (2, 3),
        ),
    )
    def test_saliency(self, factory, absolute, batch_size):
        get_module, input_shape = factory
        module = get_module(seed=0)
        input_data = torch.randn(*batch_size, *input_shape, requires_grad=True)

        captum_attributor = CaptumSaliency(module)
        if len(batch_size) > 0:
            attributions = captum_attributor.attribute(
                input_data.flatten(end_dim=len(batch_size) - 1), abs=absolute
            ).reshape(input_data.shape)
        else:
            attributions = captum_attributor.attribute(input_data, abs=absolute)

        tdhook_context_factory = Saliency(absolute=absolute)
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(TensorDict({"input": input_data}, batch_size=batch_size))

        torch.testing.assert_close(output.get(("attr", "input")), attributions)

    @pytest.mark.parametrize(
        "factory",
        (
            (get_sequential_linear_module, (10,)),
            (get_sequential_conv_module, (10, 5)),
        ),
    )
    def test_input_x_gradient(self, factory):
        get_module, input_shape = factory
        module = get_module(seed=0)
        input_data = torch.randn(input_shape, requires_grad=True)

        captum_attributor = CaptumInputXGradient(module)
        attributions = captum_attributor.attribute(input_data)

        tdhook_context_factory = Saliency(multiply_by_inputs=True)
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(TensorDict({"input": input_data}))

        torch.testing.assert_close(output.get(("attr", "input")), attributions)

    @pytest.mark.parametrize(
        "factory",
        (
            (get_sequential_linear_module, (10,)),
            (get_sequential_conv_module, (10, 5)),
        ),
    )
    @pytest.mark.parametrize(
        "multiply_by_inputs",
        (True, False),
    )
    @pytest.mark.parametrize(
        "batch_size",
        (
            tuple(),
            (3,),
            (2, 3),
        ),
    )
    def test_integrated_gradients(self, factory, multiply_by_inputs, batch_size):
        get_module, input_shape = factory
        module = get_module(seed=0)
        input_data = torch.randn(*batch_size, *input_shape)
        baseline = torch.zeros_like(input_data)

        captum_attributor = CaptumIntegratedGradients(module, multiply_by_inputs=multiply_by_inputs)
        if len(batch_size) > 0:
            attributions, convergence_delta = captum_attributor.attribute(
                input_data.flatten(end_dim=len(batch_size) - 1),
                baseline.flatten(end_dim=len(batch_size) - 1),
                return_convergence_delta=True,
            )
            attributions = attributions.reshape(input_data.shape)
            convergence_delta = convergence_delta.reshape(batch_size)
        else:
            attributions, convergence_delta = captum_attributor.attribute(
                input_data.unsqueeze(0), baseline.unsqueeze(0), return_convergence_delta=True
            )
            attributions = attributions.squeeze(0)
            convergence_delta = convergence_delta.squeeze(0)

        tdhook_context_factory = IntegratedGradients(
            compute_convergence_delta=True, multiply_by_inputs=multiply_by_inputs
        )
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(
                TensorDict({"input": input_data, ("baseline", "input"): baseline}, batch_size=batch_size)
            )

        torch.testing.assert_close(output.get(("attr", "input")), attributions)
        torch.testing.assert_close(output.get("convergence_delta"), convergence_delta)

    @pytest.mark.parametrize(
        "multiply_by_inputs",
        (True, False),
    )
    def test_multitarget_integrated_gradients(self, multiply_by_inputs):
        module = get_multitarget_module(seed=0)
        captum_module = WrapSum(module, (4, 7))
        input_data = torch.randn(3, 10)
        baseline = torch.zeros_like(input_data)

        captum_attributor = CaptumIntegratedGradients(captum_module, multiply_by_inputs=multiply_by_inputs)
        attributions, convergence_delta = captum_attributor.attribute(
            input_data, baseline, return_convergence_delta=True
        )

        sum_tdhook_context_factory = IntegratedGradients(
            compute_convergence_delta=True, multiply_by_inputs=multiply_by_inputs
        )
        with sum_tdhook_context_factory.prepare(captum_module) as hooked_module:
            sum_output = hooked_module(
                TensorDict({"input": input_data, ("baseline", "input"): baseline}, batch_size=3)
            )

        torch.testing.assert_close(sum_output.get(("attr", "input")), attributions)
        torch.testing.assert_close(sum_output.get("convergence_delta"), convergence_delta)

        tdhook_context_factory = IntegratedGradients(
            compute_convergence_delta=True,
            init_attr_targets=lambda td, _: td.apply(lambda t: t[..., (4, 7)]),
            multiply_by_inputs=multiply_by_inputs,
        )
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(TensorDict({"input": input_data, ("baseline", "input"): baseline}, batch_size=3))

        torch.testing.assert_close(output.get(("attr", "input")), attributions)
        torch.testing.assert_close(output.get("convergence_delta"), convergence_delta)


class TestGradientAttributionHelpers:
    @pytest.mark.parametrize(
        "method",
        helpers.SUPPORTED_METHODS,
    )
    @pytest.mark.parametrize(
        "n_steps",
        (10, 50, 100),
    )
    def test_approximation_parameters(self, method, n_steps):
        step_sizes_func, alphas_func = helpers.approximation_parameters(method)
        captum_step_sizes_func, captum_alphas_func = approximation_methods.approximation_parameters(method)

        tdhook_tensors = [step_sizes_func(n_steps), alphas_func(n_steps)]
        captum_tensors = [captum_step_sizes_func(n_steps), captum_alphas_func(n_steps)]

        for tdhook_tensor, captum_tensor in zip(tdhook_tensors, captum_tensors):
            assert tdhook_tensor == captum_tensor


class TestInitAttrTargetsWithLabels:
    """Test the init_attr_targets_with_labels static method."""

    @pytest.mark.parametrize("batch_size", [(), (2,), (3, 2)])
    @pytest.mark.parametrize("step_size", [25, 50])
    def test_init_attr_targets_with_labels_shapes(self, batch_size, step_size):
        """Test that init_attr_targets_with_labels returns correct shapes."""
        full_batch_size = batch_size + (step_size,)
        outputs = TensorDict({"output": torch.randn(*full_batch_size, 15)}, batch_size=full_batch_size)

        additional_init_tensors = TensorDict(
            {("label", "output"): torch.randint(0, 15, batch_size)}, batch_size=batch_size
        )

        result = IntegratedGradients.init_attr_targets_with_labels(outputs, additional_init_tensors, ["output"])

        assert isinstance(result, TensorDict)
        assert result.batch_size == full_batch_size
        assert "output" in result
        assert result["output"].shape == full_batch_size
