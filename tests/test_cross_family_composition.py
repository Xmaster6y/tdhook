import pytest
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from tdhook.contexts import HookingContextFactory
from tdhook.latent import ActivationCaching, Probing
from tdhook.weights import Adapters
from tdhook.workflow import Workflow


def _hook_count(model):
    hook_attributes = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    return sum(len(getattr(module, attribute)) for module in model.modules() for attribute in hook_attributes)


def test_activation_capture_and_probing_split_conservatively(default_test_model):
    class RecordingProbe:
        def __init__(self):
            self.values = []

        def step(self, data, **kwargs):
            self.values.append(data.detach().clone())

    probe = RecordingProbe()
    workflow = Workflow(
        ActivationCaching("linear1", cache_key=("activations", "cache")),
        Probing("linear2", lambda *_: probe),
    )
    inputs = torch.ones(2, 10)
    data = TensorDict({"input": inputs}, batch_size=[2])
    hooks_before = _hook_count(default_test_model)

    result = workflow(default_test_model, data)

    assert result["activations", "cache"]["linear1"].shape == (2, 20)
    assert result["activations", "cache"]["linear1"].device == inputs.device
    assert len(probe.values) == 1 and probe.values[0].shape == (2, 20)
    assert _hook_count(default_test_model) == hooks_before


def test_intervention_and_activation_read_split_without_state_leaks(default_test_model):
    state_before = {key: value.detach().clone() for key, value in default_test_model.state_dict().items()}
    hooks_before = _hook_count(default_test_model)
    workflow = Workflow(
        Adapters({"identity": (nn.Identity(), "linear1", "linear1")}),
        ActivationCaching("linear2", cache_key=("activations", "cache")),
    )
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    result = workflow(default_test_model, data)

    assert result["activations", "cache"]["linear2"].shape == (2, 20)
    for key, value in default_test_model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])
    assert _hook_count(default_test_model) == hooks_before
    assert not any("adapter" in name for name, _ in default_test_model.named_modules())


def test_workflow_supports_nested_multiple_input_output_keys():
    class Pair(nn.Module):
        def forward(self, left, right):
            return left + right, left - right

    in_keys = [("inputs", "left"), ("inputs", "right")]
    out_keys = [("outputs", "sum"), ("outputs", "difference")]
    model = TensorDictModule(Pair(), in_keys=in_keys, out_keys=out_keys)
    workflow = Workflow(HookingContextFactory())
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    right = torch.full((2, 3), 2.0)
    data = TensorDict({"inputs": {"left": left, "right": right}}, batch_size=[2])

    result = workflow(model, data)

    torch.testing.assert_close(result["outputs", "sum"], left + right)
    torch.testing.assert_close(result["outputs", "difference"], left - right)
    assert _hook_count(model) == 0


def test_cross_family_failure_removes_hooks_and_restores_model(default_test_model):
    class ExplodingProbe:
        def step(self, data, **kwargs):
            raise ValueError("probe failed")

    state_before = {key: value.detach().clone() for key, value in default_test_model.state_dict().items()}
    hooks_before = _hook_count(default_test_model)
    workflow = Workflow(
        Adapters({"identity": (nn.Identity(), "linear1", "linear1")}),
        Probing("linear2", lambda *_: ExplodingProbe()),
    )
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="probe failed"):
        workflow(default_test_model, data)

    for key, value in default_test_model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])
    assert _hook_count(default_test_model) == hooks_before
    assert not any("adapter" in name for name, _ in default_test_model.named_modules())
