"""
Test gradient attribution.
"""

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
    def test_saliency(self, factory, absolute):
        get_module, input_shape = factory
        module = get_module(seed=0)
        input_data = torch.randn(input_shape)

        captum_attributor = CaptumSaliency(module)
        attributions = captum_attributor.attribute(input_data, abs=absolute)

        tdhook_context_factory = Saliency(absolute=absolute)
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(TensorDict({"input": input_data}))

        torch.testing.assert_close(output.get("input_attr"), attributions)

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
        input_data = torch.randn(input_shape)

        captum_attributor = CaptumInputXGradient(module)
        attributions = captum_attributor.attribute(input_data)

        tdhook_context_factory = Saliency(multiply_by_inputs=True)
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(TensorDict({"input": input_data}))

        torch.testing.assert_close(output.get("input_attr"), attributions)

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
    def test_integrated_gradients(self, factory, multiply_by_inputs):
        get_module, input_shape = factory
        module = get_module(seed=0)
        input_data = torch.randn(input_shape).unsqueeze(0)
        baseline = torch.zeros_like(input_data)

        captum_attributor = CaptumIntegratedGradients(module, multiply_by_inputs=multiply_by_inputs)
        attributions, convergence_delta = captum_attributor.attribute(
            input_data, baseline, return_convergence_delta=True
        )

        tdhook_context_factory = IntegratedGradients(
            compute_convergence_delta=True, multiply_by_inputs=multiply_by_inputs
        )
        with tdhook_context_factory.prepare(module) as hooked_module:
            output = hooked_module(TensorDict({"input": input_data, "input_baseline": baseline}, batch_size=1))

        torch.testing.assert_close(output.get("input_attr"), attributions)
        torch.testing.assert_close(output.get("convergence_delta"), convergence_delta)

    def test_requires_batched_hook(self):
        module = nn.Linear(10, 1)
        input_data = torch.randn(10)
        baseline = torch.zeros_like(input_data)

        with pytest.raises(NotImplementedError):
            tdhook_context_factory = IntegratedGradients(compute_convergence_delta=True)
            with tdhook_context_factory.prepare(module) as hooked_module:
                hooked_module(TensorDict({"input": input_data, "input_baseline": baseline}))


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
