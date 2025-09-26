from __future__ import annotations

import torch
import pytest
from torch import nn

from tdhook.weights.pruning import Pruning


def _importance_cb(parameter, **_):
    return parameter


def _importance_cb_skip_weight(parameter, parameter_name, **_):
    return None if parameter_name == "weight" else parameter


def _importance_cb_skip_bias(parameter, parameter_name, **_):
    return None if parameter_name == "bias" else parameter


class TestPruning:
    def test_none_amount_global_pruning(self):
        with pytest.raises(ValueError):
            Pruning(importance_callback=_importance_cb)

    @pytest.mark.parametrize("amount", (0.2, 0.5))
    def test_global_pruning_applies_and_restores(self, amount):
        model = nn.Linear(10, 10)
        inp = torch.randn(10)
        original_state = (model.weight.clone(), model.bias.clone())
        original_output = model(inp)

        pruning = Pruning(importance_callback=_importance_cb, amount_to_prune=amount)
        ctx = pruning.prepare(model)
        with ctx as hooked:
            assert torch.allclose(ctx._old_weights[("module", "weight")], original_state[0])
            assert torch.allclose(ctx._old_weights[("module", "bias")], original_state[1])

            output = hooked(inp)
            assert not torch.allclose(output, original_output)
            nonzeros = torch.count_nonzero(model.weight) + torch.count_nonzero(model.bias)
            total = model.weight.numel() + model.bias.numel()
            actual_amount = (total - nonzeros) / total
            assert torch.isclose(actual_amount, torch.tensor(amount))

        assert ctx._old_weights is None
        new_output = model(inp)
        assert torch.allclose(new_output, original_output)

    def test_custom_skip(self, default_test_model):
        original_linear1_weight = default_test_model.linear1.weight.clone()
        original_linear2_weight = default_test_model.linear2.weight.clone()
        original_linear3_weight = default_test_model.linear3.weight.clone()

        def skip_linear1(name, module):
            return Pruning.default_skip(name, module) or name == "linear1"

        pruning = Pruning(importance_callback=_importance_cb_skip_bias, amount_to_prune=0.5, skip_modules=skip_linear1)
        ctx = pruning.prepare(default_test_model)
        with ctx:
            assert torch.allclose(default_test_model.linear1.weight, original_linear1_weight)
            assert not torch.allclose(default_test_model.linear2.weight, original_linear2_weight)
            assert not torch.allclose(default_test_model.linear3.weight, original_linear3_weight)

    def test_custom_relative_path(self, default_test_model):
        original_linear1_weight = default_test_model.linear1.weight.clone()
        original_linear2_weight = default_test_model.linear2.weight.clone()
        original_linear3_weight = default_test_model.linear3.weight.clone()

        pruning = Pruning(importance_callback=_importance_cb_skip_bias, amount_to_prune=0.5, relative_path="linear1")
        ctx = pruning.prepare(default_test_model)
        with ctx:
            assert not torch.allclose(default_test_model.linear1.weight, original_linear1_weight)
            assert torch.allclose(default_test_model.linear2.weight, original_linear2_weight)
            assert torch.allclose(default_test_model.linear3.weight, original_linear3_weight)

    def test_pruning_modules(self, default_test_model):
        original_linear1_weight = default_test_model.linear1.weight.clone()
        original_linear2_weight = default_test_model.linear2.weight.clone()
        original_linear3_weight = default_test_model.linear3.weight.clone()

        pruning = Pruning(
            importance_callback=_importance_cb_skip_bias,
            amount_to_prune=0.5,
            modules_to_prune={"linear1": (1, 0.5), "linear3": (1, 0.1)},
        )
        ctx = pruning.prepare(default_test_model)
        with ctx:
            assert torch.allclose(default_test_model.linear2.weight, original_linear2_weight)
            assert not torch.allclose(default_test_model.linear1.weight, original_linear1_weight)
            assert torch.isclose(
                torch.count_nonzero(default_test_model.linear1.weight) / default_test_model.linear1.weight.numel(),
                torch.tensor(0.5),
            )
            assert not torch.allclose(default_test_model.linear3.weight, original_linear3_weight)
            assert torch.isclose(
                torch.count_nonzero(default_test_model.linear3.weight) / default_test_model.linear3.weight.numel(),
                torch.tensor(0.9),
            )

    def test_no_pruning(self):
        model = nn.Linear(10, 10)
        inp = torch.randn(10)
        original_output = model(inp)

        pruning = Pruning(importance_callback=_importance_cb, amount_to_prune=0)
        ctx = pruning.prepare(model)
        with ctx as hooked:
            output = hooked(inp)
            assert torch.allclose(output, original_output)

    def test_skip_importance_cb(self):
        model = nn.Linear(10, 10)
        original_state = (model.weight.clone(), model.bias.clone())

        pruning = Pruning(importance_callback=_importance_cb_skip_weight, amount_to_prune=0.5)
        ctx = pruning.prepare(model)
        with ctx:
            assert torch.allclose(model.weight, original_state[0])
            assert not torch.allclose(model.bias, original_state[1])
