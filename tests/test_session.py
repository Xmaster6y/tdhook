import gc

import pytest
import torch
from torch import nn

from tdhook.session import HookProgram, HookSession, HookSpec
from tdhook.targets import Target


def test_session_captures_and_replaces_an_activation(default_test_model):
    target = Target("linear1", "activation", -1, (2,))
    x = torch.randn(3, 10)
    baseline = default_test_model(x)

    with HookSession(default_test_model) as session:
        captured = session.capture(target)
        session.replace(target, 100)
        modified = default_test_model(x)
        program = session.program

    assert captured.value is not None
    assert captured.value.shape == (3, 1)
    assert not torch.allclose(modified, baseline)
    assert torch.allclose(default_test_model(x), baseline)
    assert program == HookProgram(
        (
            HookSpec(target.module_path, "capture", "fwd", target=target),
            HookSpec(target.module_path, "replace", "fwd", target=target),
        )
    )


def test_session_preserves_single_tensor_tuple_output():
    class TupleModule(nn.Module):
        def forward(self, x):
            return (x + 1,)

    model = TupleModule()
    x = torch.ones(2, 3)
    with HookSession(model) as session:
        session.replace(Target("", "activation", -1, (1,)), 0)
        output = model(x)

    assert isinstance(output, tuple)
    assert torch.equal(output[0][:, 1], torch.zeros(2))


def test_session_selects_and_replaces_one_leaf_of_a_structured_output():
    class StructuredModule(nn.Module):
        def forward(self, x):
            return {"predictions": (x + 1, x + 2), "metadata": "kept"}

    model = StructuredModule()
    x = torch.ones(2, 3)
    target = Target("", "activation", -1, (1,), output_path=("predictions", 1))

    with HookSession(model) as session:
        captured = session.capture(target)
        session.replace(target, 0)
        output = model(x)

    assert captured.value is not None
    assert torch.equal(captured.value, torch.full((2, 1), 3.0))
    assert torch.equal(output["predictions"][0], x + 1)
    assert torch.equal(output["predictions"][1][:, 1], torch.zeros(2))
    assert output["metadata"] == "kept"


def test_session_selects_one_gradient_from_multiple_module_outputs():
    class MultiOutputModule(nn.Module):
        def forward(self, x):
            return x * 2, x * 3

    model = MultiOutputModule()
    x = torch.ones(2, 3, requires_grad=True)
    target = Target("", "gradient", -1, (0,), output_path=(1,))

    with HookSession(model) as session:
        captured = session.capture(target)
        first, second = model(x)
        (first.sum() + second.sum()).backward()

    assert captured.value is not None
    assert torch.equal(captured.value, torch.ones(2, 1))


def test_session_structured_output_path_errors_are_specific():
    value = {"items": [torch.ones(1)]}
    replacement = torch.zeros(1)

    assert HookSession._hook_tensor(value, ("items", 0)).shape == (1,)
    assert HookSession._replace_hook_tensor(value["items"], (0,), replacement) == [replacement]
    with pytest.raises(ValueError, match="out of range"):
        HookSession._hook_tensor(value, ("items", 1))
    with pytest.raises(ValueError, match="is missing"):
        HookSession._hook_tensor(value, ("missing",))
    with pytest.raises(ValueError, match="does not match"):
        HookSession._hook_tensor(value, (0,))
    with pytest.raises(ValueError, match="does not match"):
        HookSession._replace_hook_tensor(value, (0,), replacement)
    with pytest.raises(ValueError, match="tensor leaf"):
        HookSession._hook_tensor({"metadata": "text"}, ("metadata",))


def test_session_removes_activation_hooks_after_failure(default_test_model):
    target = Target("linear1", "activation", -1, (0,))
    x = torch.randn(2, 10)
    baseline = default_test_model(x)

    with pytest.raises(RuntimeError, match="boom"):
        with HookSession(default_test_model) as session:
            session.replace(target, 100)
            assert not torch.allclose(default_test_model(x), baseline)
            raise RuntimeError("boom")

    assert torch.allclose(default_test_model(x), baseline)


def test_session_captures_channels_and_replaces_gradients():
    model = nn.Conv2d(2, 3, kernel_size=1, bias=False)
    activation = Target("", "activation", 1, (1,))
    gradient = Target("", "gradient", 1, (1,))
    x = torch.randn(2, 2, 4, 4, requires_grad=True)

    with HookSession(model) as session:
        captured = session.capture(activation)
        captured_gradient = session.capture(gradient)
        session.replace(gradient, 0)
        model(x).sum().backward()

    assert captured.value is not None
    assert captured.value.shape == (2, 1, 4, 4)
    assert captured_gradient.value is not None
    assert captured_gradient.value.shape == (2, 1, 4, 4)
    assert torch.allclose(model.weight.grad[1], torch.zeros_like(model.weight.grad[1]))


def test_session_captures_parameter_values():
    model = nn.Linear(3, 2, bias=False)
    target = Target("", "parameter", 0, (1,), parameter="weight")

    with HookSession(model) as session:
        captured = session.capture(target)

    assert captured.value is not None
    assert torch.equal(captured.value, model.weight[1:2])
    assert session.program == HookProgram((HookSpec(target.module_path, "capture", None, target=target),))


@pytest.mark.parametrize("axis,indices", [(0, (0,)), (1, (1,))])
def test_session_restores_parameters_after_success_and_failure(axis, indices):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    target = Target("0", "parameter", axis, indices, parameter="weight")
    original = model[0].weight.detach().clone()

    with pytest.raises(RuntimeError, match="boom"):
        with HookSession(model) as session:
            session.replace(target, -3)
            assert not torch.equal(model[0].weight, original)
            raise RuntimeError("boom")
    assert torch.equal(model[0].weight, original)

    with HookSession(model) as session:
        session.replace(target, -3)
        assert not torch.equal(model[0].weight, original)
    assert torch.equal(model[0].weight, original)


def test_session_restores_parameter_when_replacement_assignment_fails():
    model = nn.Linear(3, 2, bias=False)
    target = Target("", "parameter", 0, (0,), parameter="weight")
    original = model.weight.detach().clone()

    with HookSession(model) as session:
        with pytest.raises(RuntimeError):
            session.replace(target, torch.ones(2, 2))
        assert torch.equal(model.weight, original)
        assert session.program == HookProgram()


def test_session_restores_the_live_parameter_after_reference_reassignment():
    model = nn.Linear(3, 2, bias=False)
    target = Target("", "parameter", 0, (0,), parameter="weight")
    original_row = model.weight[0].detach().clone()

    with HookSession(model) as session:
        session.replace(target, -3)
        replacement = nn.Parameter(model.weight.detach().clone())
        replacement.data[0].fill_(9)
        model.weight = replacement

    torch.testing.assert_close(model.weight[0], original_row)


def test_session_requires_an_active_live_model(default_test_model):
    session = HookSession(default_test_model)
    target = Target("linear1", "activation", -1, (0,))

    with pytest.raises(RuntimeError, match="active context"):
        session.capture(target)
    with session:
        with pytest.raises(RuntimeError, match="twice"):
            session.__enter__()

    model = nn.Identity()
    dead_session = HookSession(model)
    del model
    gc.collect()
    with pytest.raises(RuntimeError, match="no longer exists"):
        dead_session.__enter__()


def test_session_validates_model_type_and_exit_state():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        HookSession(object())

    session = HookSession(nn.Identity())
    with pytest.raises(RuntimeError, match="not active"):
        session.__exit__(None, None, None)

    with pytest.raises(ValueError, match="output_path"):
        HookSession._hook_tensor((torch.ones(1), torch.ones(1)))
