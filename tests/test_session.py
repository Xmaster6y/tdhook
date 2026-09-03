import gc
from collections import UserDict

from dataclasses import asdict
import json

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from tdhook.session import CapturedTarget, CaptureSource, EarlyStopResult, HookProgram, HookSession, HookSpec
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

    assert len(captured.values) == 1
    assert captured.values[-1].shape == (3, 1)
    assert not torch.allclose(modified, baseline)
    assert torch.allclose(default_test_model(x), baseline)
    assert program == HookProgram(
        (
            HookSpec(target.module_path, "capture", "fwd", target=target),
            HookSpec(target.module_path, "replace", "fwd", target=target),
        )
    )


def test_session_preserves_repeated_captures_in_call_order(default_test_model):
    target = Target("linear1", "activation", -1, (0,))
    inputs = (torch.zeros(1, 10), torch.ones(1, 10))

    with HookSession(default_test_model) as session:
        captured = session.capture(target)
        for value in inputs:
            default_test_model(value)

    assert len(captured.values) == 2
    assert not torch.equal(captured.values[0], captured.values[1])


def test_session_routes_each_live_capture_to_a_later_target():
    model = nn.Sequential(nn.Identity(), nn.Identity())
    source = Target("0", "activation", -1, (0,))
    destination = Target("1", "activation", -1, (2,))
    inputs = (torch.tensor([[2.0, 3.0, 4.0]]), torch.tensor([[5.0, 6.0, 7.0]]))

    with HookSession(model) as session:
        captured = session.capture(source)
        session.replace(destination, captured, direction="fwd_pre", transform=lambda value: value * 10)
        outputs = tuple(model(value) for value in inputs)

    assert torch.equal(outputs[0], torch.tensor([[2.0, 3.0, 20.0]]))
    assert torch.equal(outputs[1], torch.tensor([[5.0, 6.0, 50.0]]))
    assert session.program == HookProgram(
        (
            HookSpec("0", "capture", "fwd", target=source),
            HookSpec(
                "1",
                "replace",
                "fwd_pre",
                target=destination,
                source=CaptureSource(hook_index=0, detach=True),
            ),
        )
    )
    assert not model[0]._forward_hooks
    assert not model[1]._forward_pre_hooks


@pytest.mark.parametrize(
    "detach,expected",
    [(True, torch.tensor([[0.0, 0.0, 0.0]])), (False, torch.tensor([[1.0, 0.0, 0.0]]))],
)
def test_session_live_replacement_makes_graph_retention_explicit(detach, expected):
    model = nn.Sequential(nn.Identity(), nn.Identity())
    source = Target("0", "activation", -1, (0,))
    destination = Target("1", "activation", -1, (2,))
    value = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)

    with HookSession(model) as session:
        captured = session.capture(source, detach=detach)
        session.replace(destination, captured, direction="fwd_pre")
        model(value)[:, 2].sum().backward()

    assert value.grad is not None
    assert torch.equal(value.grad, expected)
    assert session.program.hooks[1].source == CaptureSource(hook_index=0, detach=detach)


def test_session_routes_a_live_gradient_to_a_later_backward_target():
    model = nn.Sequential(nn.Identity(), nn.Identity())
    source = Target("1", "gradient", -1, (0,))
    destination = Target("0", "gradient", -1, (2,))
    value = torch.ones(1, 3, requires_grad=True)

    with HookSession(model) as session:
        captured = session.capture(source, direction="bwd_pre")
        session.replace(destination, captured, direction="bwd_pre", transform=lambda gradient: gradient * 4)
        model(value).sum().backward()

    assert captured.values
    assert torch.equal(value.grad, torch.tensor([[1.0, 1.0, 4.0]]))


def test_session_rejects_stale_or_incompatible_live_captures():
    class ReorderableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = nn.Identity()
            self.destination = nn.Identity()
            self.reverse = False

        def forward(self, value):
            if self.reverse:
                return self.source(self.destination(value))
            return self.destination(self.source(value))

    model = ReorderableModel()
    source = Target("source", "activation", -1, (0,))
    destination = Target("destination", "activation", -1, (1,))

    with pytest.raises(RuntimeError, match="fresh source capture"):
        with HookSession(model) as session:
            captured = session.capture(source)
            session.replace(destination, captured)
            model(torch.ones(1, 2))
            model.reverse = True
            model(torch.ones(1, 2))

    assert not model.source._forward_hooks
    assert not model.destination._forward_hooks

    gradient = Target("source", "gradient", -1, (0,))
    with HookSession(model) as session:
        captured = session.capture(source)
        with pytest.raises(ValueError, match="same kind"):
            session.replace(gradient, captured)
        with pytest.raises(ValueError, match="only valid for a live"):
            session.replace(destination, 0, transform=lambda value: value)

    with HookSession(model) as session:
        with pytest.raises(ValueError, match="same active HookSession"):
            session.replace(destination, captured)


def test_session_rejects_an_unconsumed_capture_from_an_earlier_execution():
    class ConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = nn.Identity()
            self.destination = nn.Identity()
            self.use_source = True

        def forward(self, value):
            if self.use_source:
                return self.source(value)
            return self.destination(value)

    model = ConditionalModel()
    source = Target("source", "activation", -1, (0,))
    destination = Target("destination", "activation", -1, (1,))

    with pytest.raises(RuntimeError, match="fresh source capture"):
        with HookSession(model) as session:
            captured = session.capture(source)
            session.replace(destination, captured)
            model(torch.ones(1, 2))
            model.use_source = False
            model(torch.ones(1, 2))

    assert not model._forward_pre_hooks
    assert not model.source._forward_hooks
    assert not model.destination._forward_hooks


def test_session_validates_live_capture_options():
    model = nn.Identity()
    activation = Target("", "activation", -1, (0,))
    parameter_model = nn.Linear(2, 2)
    parameter = Target("", "parameter", 0, (0,), parameter="weight")

    with HookSession(model) as session:
        with pytest.raises(TypeError, match="detach must be a bool"):
            session.capture(activation, detach=1)
        forged = CapturedTarget(_session_token=session._session_token)
        with pytest.raises(ValueError, match="created by HookSession.capture"):
            session.replace(activation, forged)
        captured = session.capture(activation)
        with pytest.raises(TypeError, match="transform must be callable"):
            session.replace(activation, captured, transform=1)
        session.replace(activation, captured)
        session.replace(activation, captured)
        assert torch.equal(model(torch.ones(1)), torch.ones(1))

    assert not model._forward_pre_hooks
    assert not model._forward_hooks

    with HookSession(parameter_model) as session:
        captured = session.capture(parameter)
        with pytest.raises(ValueError, match="only activation or gradient"):
            session.replace(parameter, captured)


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

    assert torch.equal(captured.values[-1], torch.full((2, 1), 3.0))
    assert torch.equal(output["predictions"][0], x + 1)
    assert torch.equal(output["predictions"][1][:, 1], torch.zeros(2))
    assert output["metadata"] == "kept"


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda value: UserDict({"predictions": value, "metadata": "kept"}),
        lambda value: TensorDict({"predictions": value, "metadata": "kept"}, batch_size=[]),
    ],
)
def test_session_preserves_mapping_output_types(container_factory):
    class StructuredModule(nn.Module):
        def forward(self, x):
            return container_factory(x + 1)

    model = StructuredModule()
    x = torch.ones(2, 3)
    target = Target("", "activation", -1, (1,), output_path=("predictions",))

    with HookSession(model) as session:
        captured = session.capture(target)
        session.replace(target, 0)
        output = model(x)

    assert isinstance(output, type(container_factory(x)))
    assert torch.equal(captured.values[-1], torch.full((2, 1), 2.0))
    assert torch.equal(output["predictions"][:, 1], torch.zeros(2))
    assert output["metadata"] == "kept"


def test_session_captures_and_replaces_keyword_inputs():
    class KeywordModule(nn.Module):
        def forward(self, x, *, scale):
            return x * scale

    model = KeywordModule()
    x = torch.ones(2, 3)
    target = Target("", "activation", -1, (1,), output_path=(1, "scale"))

    with HookSession(model) as session:
        captured = session.capture(target, direction="fwd_pre_kwargs")
        session.replace(target, 4, direction="fwd_pre_kwargs")
        output = model(x, scale=torch.ones(2, 3))

    assert torch.equal(captured.values[-1], torch.ones(2, 1))
    assert torch.equal(output[:, 1], torch.full((2,), 4.0))
    assert torch.equal(output[:, (0, 2)], torch.ones(2, 2))
    assert session.program == HookProgram(
        (
            HookSpec("", "capture", "fwd_pre_kwargs", target=target),
            HookSpec("", "replace", "fwd_pre_kwargs", target=target),
        )
    )
    assert not model._forward_pre_hooks


@pytest.mark.parametrize(
    "container_factory,output_path",
    [
        (lambda value: (value, "kept"), (0, 0)),
        (lambda value: [value, "kept"], (0, 0)),
        (lambda value: UserDict({"predictions": value, "metadata": "kept"}), (0, "predictions")),
        (
            lambda value: TensorDict({"predictions": value, "metadata": "kept"}, batch_size=[]),
            (0, "predictions"),
        ),
    ],
)
def test_session_preserves_structured_forward_inputs(container_factory, output_path):
    class InputModule(nn.Module):
        def forward(self, value):
            return value

    model = InputModule()
    original = container_factory(torch.ones(2, 3))
    target = Target("", "activation", -1, (2,), output_path=output_path)

    with HookSession(model) as session:
        captured = session.capture(target, direction="fwd_pre")
        session.replace(target, 0, direction="fwd_pre")
        output = model(original)

    assert isinstance(output, type(original))
    assert torch.equal(captured.values[-1], torch.ones(2, 1))
    selected = Target._output_tensor((output,), output_path)
    assert torch.equal(selected[:, 2], torch.zeros(2))


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

    assert torch.equal(captured.values[-1], torch.ones(2, 1))


def test_session_captures_and_replaces_gradient_inputs_and_outputs():
    model = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    grad_output_target = Target("", "gradient", -1, (1,), output_path=(0,))
    first_input = torch.ones(1, 3, requires_grad=True)
    with HookSession(model) as output_session:
        captured_output = output_session.capture(grad_output_target, direction="bwd_pre")
        output_session.replace(grad_output_target, 0, direction="bwd_pre")
        model(first_input).sum().backward()

    assert torch.equal(captured_output.values[-1], torch.ones(1, 1))
    assert torch.equal(first_input.grad, model.weight[0:1])
    assert output_session.program == HookProgram(
        (
            HookSpec("", "capture", "bwd_pre", target=grad_output_target),
            HookSpec("", "replace", "bwd_pre", target=grad_output_target),
        )
    )

    grad_input_target = Target("", "gradient", -1, (1,), output_path=(0,))
    second_input = torch.ones(1, 3, requires_grad=True)
    with HookSession(model) as input_session:
        captured_input = input_session.capture(grad_input_target, direction="bwd")
        input_session.replace(grad_input_target, 0, direction="bwd")
        model(second_input).sum().backward()

    assert torch.equal(captured_input.values[-1], torch.tensor([[7.0]]))
    assert second_input.grad is not None
    assert torch.equal(second_input.grad, torch.tensor([[5.0, 0.0, 9.0]]))
    assert input_session.program == HookProgram(
        (
            HookSpec("", "capture", "bwd", target=grad_input_target),
            HookSpec("", "replace", "bwd", target=grad_input_target),
        )
    )
    assert not model._backward_hooks
    assert not model._backward_pre_hooks


def test_session_rejects_directions_that_do_not_match_the_target():
    model = nn.Linear(3, 2)
    activation = Target("", "activation", -1, (0,))
    gradient = Target("", "gradient", -1, (0,))
    parameter = Target("", "parameter", 0, (0,), parameter="weight")

    with HookSession(model) as session:
        with pytest.raises(ValueError, match="activation target"):
            session.capture(activation, direction="bwd")
        with pytest.raises(ValueError, match="gradient target"):
            session.replace(gradient, 0, direction="fwd")
        with pytest.raises(ValueError, match="parameter target"):
            session.capture(parameter, direction="fwd")
        assert session.program == HookProgram()


def test_session_removes_input_and_gradient_hooks_after_hook_failures():
    input_model = nn.Identity()
    bad_input = Target("", "activation", -1, (0,), output_path=(2,))

    with pytest.raises(ValueError, match="out of range"):
        with HookSession(input_model) as session:
            session.capture(bad_input, direction="fwd_pre_kwargs")
            input_model(torch.ones(1))

    assert not input_model._forward_pre_hooks

    gradient_model = nn.Linear(2, 1)
    bad_gradient = Target("", "gradient", -1, (0,), output_path=(1,))
    value = torch.ones(1, 2, requires_grad=True)

    with pytest.raises(ValueError, match="out of range"):
        with HookSession(gradient_model) as session:
            session.capture(bad_gradient, direction="bwd")
            gradient_model(value).sum().backward()

    assert not gradient_model._backward_hooks


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


def test_session_stops_forward_execution_and_exposes_partial_results():
    events = []

    class RecordingLinear(nn.Linear):
        def forward(self, x):
            events.append(self.out_features)
            return super().forward(x)

    model = nn.Sequential(RecordingLinear(3, 4), nn.ReLU(), RecordingLinear(4, 2))
    target = Target("0", "activation", -1, (0,))
    x = torch.randn(2, 3)
    expected = torch.relu(model[0](x))
    events.clear()
    session = HookSession(model)

    with session:
        captured = session.capture(target)
        stopped = session.stop("1")
        model(x)
        pytest.fail("execution after a managed early stop must be skipped")

    assert isinstance(stopped, EarlyStopResult)
    assert stopped.reached
    assert stopped.output is not None
    assert events == [4]
    assert torch.equal(stopped.output, expected)
    assert captured.values
    assert session.program == HookProgram(
        (
            HookSpec("0", "capture", "fwd", target=target),
            HookSpec("1", "stop", "fwd"),
        ),
        stopped_at="1",
    )
    assert len(model[0]._forward_hooks) == 0
    assert len(model[1]._forward_hooks) == 0


def test_session_leaves_unreached_stop_explicit_and_propagates_failures():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    session = HookSession(model)

    with pytest.raises(RuntimeError, match="boom"):
        with session:
            stopped = session.stop("1")
            raise RuntimeError("boom")

    assert not stopped.reached
    assert stopped.output is None
    assert session.program == HookProgram((HookSpec("1", "stop", "fwd"),))
    assert len(model[1]._forward_hooks) == 0


def test_session_validates_early_stop_location():
    model = nn.Sequential(nn.Identity())

    with HookSession(model) as session:
        with pytest.raises(TypeError, match="module_path"):
            session.stop(None)
        with pytest.raises(ValueError, match="Invalid submodule path"):
            session.stop("missing")


def test_session_warns_when_early_stop_location_is_a_module_list():
    class ModuleListModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])

        def forward(self, x):
            return self.layers[0](x)

    model = ModuleListModel()
    with HookSession(model) as session:
        with pytest.warns(UserWarning, match="ModuleList"):
            stopped = session.stop("layers")
        output = model(torch.ones(1))

    assert torch.equal(output, torch.ones(1))
    assert not stopped.reached
    assert session.program.stopped_at is None


def test_session_stop_signal_bypasses_model_exception_handlers():
    events = []

    class CatchingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Identity()
            self.second = nn.Identity()

        def forward(self, x):
            try:
                x = self.first(x)
            except Exception:
                events.append("caught")
            events.append("continued")
            return self.second(x)

    model = CatchingModel()
    with HookSession(model) as session:
        stopped = session.stop("first")
        model(torch.ones(1))

    assert stopped.reached
    assert events == []


def test_session_rejects_gradient_operations_with_early_stopping_in_either_order():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    gradient = Target("0", "gradient", -1, (0,))
    message = "cannot be combined with gradient operations"

    with HookSession(model) as session:
        session.capture(gradient)
        with pytest.raises(ValueError, match=message):
            session.stop("1")

    with HookSession(model) as session:
        session.stop("1")
        with pytest.raises(ValueError, match=message):
            session.replace(gradient, 0)


def test_session_partial_output_remains_available_to_autograd_after_stop():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    x = torch.randn(1, 3, requires_grad=True)

    with HookSession(model) as session:
        stopped = session.stop("1")
        model(x)

    assert isinstance(stopped.output, torch.Tensor)
    stopped.output.sum().backward()
    assert x.grad is not None


def test_session_restores_temporary_state_after_early_stop():
    model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.Identity())
    target = Target("0", "parameter", 0, (0,), parameter="weight")
    original = model[0].weight.detach().clone()

    with HookSession(model) as session:
        session.replace(target, -3)
        session.stop("1")
        model(torch.randn(1, 3))

    assert torch.equal(model[0].weight, original)


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

    assert captured.values[-1].shape == (2, 1, 4, 4)
    assert captured_gradient.values[-1].shape == (2, 1, 4, 4)
    assert torch.allclose(model.weight.grad[1], torch.zeros_like(model.weight.grad[1]))


def test_session_captures_parameter_values():
    model = nn.Linear(3, 2, bias=False)
    target = Target("", "parameter", 0, (1,), parameter="weight")

    with HookSession(model) as session:
        captured = session.capture(target)

    assert torch.equal(captured.values[-1], model.weight[1:2])
    assert session.program == HookProgram((HookSpec(target.module_path, "capture", None, target=target),))


def test_session_selects_and_resets_repeated_module_occurrences():
    class RepeatedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Identity()

        def forward(self, x):
            return self.shared(x + 1) + self.shared(x + 2)

    model = RepeatedModule()
    target = Target("shared", "activation", -1, (0,), occurrences=(1,))

    with HookSession(model) as session:
        captured = session.capture(target)
        session.replace(target, 10)
        first = model(torch.zeros(1, 2))
        second = model(torch.ones(1, 2))

    assert len(captured.values) == 2
    assert torch.equal(captured.values[0], torch.tensor([[2.0]]))
    assert torch.equal(captured.values[1], torch.tensor([[3.0]]))
    assert torch.equal(first, torch.tensor([[11.0, 3.0]]))
    assert torch.equal(second, torch.tensor([[12.0, 5.0]]))


def test_session_selects_multiple_occurrences_and_exposes_immutable_evidence():
    class RepeatedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Identity()

        def forward(self, x):
            return torch.cat((self.shared(x + 1), self.shared(x + 2), self.shared(x + 3)), dim=-1)

    model = RepeatedModule()
    target = Target("shared", "activation", -1, (0,), occurrences=(0, 2))

    with HookSession(model) as session:
        captured = session.capture(target)
        first = model(torch.zeros(1, 1))
        second = model(torch.ones(1, 1))

    assert torch.equal(first, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(second, torch.tensor([[2.0, 3.0, 4.0]]))
    assert [value.item() for value in captured.values] == [1.0, 3.0, 2.0, 4.0]
    assert session.program.occurrence_plans[0].target_path == "shared"
    assert session.program.occurrence_plans[0].selected_indices == (0, 2)
    assert tuple(item.root_pass for item in session.occurrence_evidence) == (0, 1)
    assert all(item.selected_indices == (0, 2) for item in session.occurrence_evidence)
    assert all(item.observed_indices == (0, 1, 2) for item in session.occurrence_evidence)
    json.dumps([asdict(item) for item in session.occurrence_evidence])
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_session_selects_root_pre_hook_occurrence_with_prepend():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,), occurrences=(0,))

    with HookSession(model) as session:
        captured = session.capture(target, direction="fwd_pre", prepend=True)
        model(torch.tensor([1.0, 2.0]))

    assert len(captured.values) == 1
    assert torch.equal(captured.values[-1], torch.tensor([1.0]))


def test_session_selects_repeated_gradient_occurrence():
    class RepeatedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Identity()
            self.branch_inputs = ()

        def forward(self, x):
            first_input = x * 2
            second_input = x * 3
            first_input.retain_grad()
            second_input.retain_grad()
            self.branch_inputs = (first_input, second_input)
            return self.shared(first_input) + self.shared(second_input)

    model = RepeatedModule()
    target = Target("shared", "gradient", -1, (0,), occurrences=(1,))
    x = torch.ones(1, 2, requires_grad=True)

    with HookSession(model) as session:
        captured = session.capture(target)
        session.replace(target, 0)
        model(x).sum().backward()

    assert len(captured.values) == 1
    assert torch.equal(captured.values[-1], torch.ones(1, 1))
    assert len(model.branch_inputs) == 2
    branch_gradients = [branch_input.grad for branch_input in model.branch_inputs]
    assert all(gradient is not None for gradient in branch_gradients)
    assert sorted(gradient[0, 0].item() for gradient in branch_gradients) == [0.0, 1.0]
    assert all(gradient[0, 1].item() == 1.0 for gradient in branch_gradients)
    assert x.grad is not None
    assert x.grad[0, 0].item() in {2.0, 3.0}
    assert x.grad[0, 1].item() == 5.0


@pytest.mark.parametrize("operation", ["capture", "replace"])
def test_session_fails_when_requested_occurrence_is_not_reached(operation):
    model = nn.Sequential(nn.Identity())
    target = Target("0", "activation", -1, (0,), occurrences=(1,))

    with (
        pytest.raises(RuntimeError, match=rf"{operation} target '0' requested occurrence 1.*called 1 time"),
        HookSession(model) as session,
    ):
        if operation == "capture":
            session.capture(target)
        else:
            session.replace(target, 0)
        model(torch.ones(1, 2))

    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_session_fails_closed_when_a_multi_occurrence_is_missing():
    model = nn.Sequential(nn.Identity())
    target = Target("0", "activation", -1, (0,), occurrences=(0, 2))

    with (
        pytest.raises(RuntimeError, match=r"requested occurrences \(0, 2\).*called 1 time"),
        HookSession(model) as session,
    ):
        session.capture(target)
        model(torch.ones(1, 2))

    assert session.occurrence_evidence == ()
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


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
