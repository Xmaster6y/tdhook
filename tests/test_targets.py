import json

import pytest
import torch
from torch import nn

from tdhook.targets import Target


def test_target_round_trip_is_json_serializable():
    target = Target("features.0", "activation", -1, (1, 3))

    assert Target.from_dict(target.to_dict()) == target
    assert Target.from_json(target.to_json()) == target
    assert json.loads(target.to_json())["indices"] == [1, 3]


def test_capture_and_replace_mlp_unit(default_test_model):
    target = Target("linear1", "activation", -1, (2,))
    x = torch.randn(3, 10)

    with target.capture(default_test_model) as captured:
        default_test_model(x)
    assert captured.value is not None
    assert captured.value.shape == (3, 1)

    baseline = default_test_model(x)
    with target.replace(default_test_model, 100):
        modified = default_test_model(x)
    assert not torch.allclose(modified, baseline)
    assert torch.allclose(default_test_model(x), baseline)


def test_replace_preserves_single_tensor_tuple_output():
    class TupleModule(nn.Module):
        def forward(self, x):
            return (x + 1,)

    model = TupleModule()
    x = torch.ones(2, 3)
    with Target("", "activation", -1, (1,)).replace(model, 0):
        output = model(x)

    assert isinstance(output, tuple)
    assert torch.equal(output[0][:, 1], torch.zeros(2))


def test_capture_cnn_channel_and_replace_gradient():
    model = nn.Conv2d(2, 3, kernel_size=1, bias=False)
    target = Target("", "activation", 1, (1,))
    x = torch.randn(2, 2, 4, 4, requires_grad=True)
    with target.capture(model) as captured:
        model(x)
    assert captured.value is not None
    assert captured.value.shape == (2, 1, 4, 4)

    gradient_target = Target("", "gradient", 1, (1,))
    with gradient_target.replace(model, 0):
        model(x).sum().backward()
    assert torch.allclose(model.weight.grad[1], torch.zeros_like(model.weight.grad[1]))


def test_capture_gradient_and_parameter_values():
    model = nn.Linear(3, 2, bias=False)
    x = torch.randn(4, 3, requires_grad=True)

    with Target("", "gradient", -1, (0,)).capture(model) as gradient:
        model(x).sum().backward()
    assert gradient.value is not None
    assert gradient.value.shape == (4, 1)

    with Target("", "parameter", 0, (1,), parameter="weight").capture(model) as parameter:
        assert parameter.value is not None
        assert torch.equal(parameter.value, model.weight[1:2])


@pytest.mark.parametrize("axis,indices", [(0, (0,)), (1, (1,))])
def test_parameter_rows_and_columns_restore_after_failure(axis, indices):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    target = Target("0", "parameter", axis, indices, parameter="weight")
    original = model[0].weight.detach().clone()

    with pytest.raises(RuntimeError, match="boom"):
        with target.replace(model, -3):
            assert not torch.equal(model[0].weight, original)
            raise RuntimeError("boom")
    assert torch.equal(model[0].weight, original)

    with target.replace(model, -3):
        assert not torch.equal(model[0].weight, original)
    assert torch.equal(model[0].weight, original)


def test_invalid_targets_have_clear_errors(default_test_model):
    with pytest.raises(ValueError, match="Invalid target kind"):
        Target("linear1", "other", 0, (0,))
    with pytest.raises(ValueError, match="at least one"):
        Target("linear1", "activation", 0, ())
    with pytest.raises(TypeError, match="integers"):
        Target("linear1", "activation", 0, ("unit",))
    with pytest.raises(ValueError, match="parameter targets require"):
        Target("linear1", "parameter", 0, (0,))
    with pytest.raises(ValueError, match="only valid for parameter"):
        Target("linear1", "activation", 0, (0,), parameter="weight")
    with pytest.raises(ValueError, match="missing indices"):
        Target.from_dict({"module_path": "linear1"})
    with pytest.raises(ValueError, match="JSON is invalid"):
        Target.from_json("not json")
    with pytest.raises(ValueError, match="contain an object"):
        Target.from_json("[]")
    with pytest.raises(ValueError, match="does not resolve"):
        Target("missing", "activation", 0, (0,)).validate(default_test_model)

    executed = False

    def dangerous_path():
        nonlocal executed
        executed = True
        return nn.Identity()

    default_test_model.dangerous_path = dangerous_path
    with pytest.raises(ValueError, match="does not resolve"):
        Target("dangerous_path()", "activation", 0, (0,)).validate(default_test_model)
    assert not executed
    with pytest.raises(ValueError, match="has no parameter"):
        Target("linear1", "parameter", 0, (0,), parameter="missing").validate(default_test_model)
    with pytest.raises(ValueError, match="out of bounds"):
        with Target("linear1", "activation", 1, (100,)).capture(default_test_model):
            default_test_model(torch.randn(1, 10))
    with pytest.raises(ValueError, match="feature_axis"):
        Target("linear1", "activation", 2, (0,))._selection(torch.randn(1, 2))
    with pytest.raises(ValueError, match="exactly one tensor"):
        Target._hook_tensor((torch.ones(1), torch.ones(1)))
