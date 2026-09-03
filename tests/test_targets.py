import json

import pytest
import torch
from torch import nn

from tdhook.targets import OccurrenceSelector, Target


def test_target_round_trip_is_json_serializable():
    target = Target("features.0", "activation", -1, (1, 3), output_path=("features", 0), occurrence=1)

    assert Target.from_dict(target.to_dict()) == target
    assert Target.from_json(target.to_json()) == target
    assert json.loads(target.to_json())["indices"] == [1, 3]
    assert json.loads(target.to_json())["output_path"] == ["features", 0]
    assert json.loads(target.to_json())["occurrence"] == 1


def test_multi_occurrence_target_round_trip_is_json_serializable():
    selector = OccurrenceSelector((0, 2))
    target = Target("shared", "activation", -1, (0,), occurrence=selector)

    assert Target.from_dict(target.to_dict()) == target
    assert Target.from_json(target.to_json()) == target
    assert target.occurrence_indices == (0, 2)
    assert json.loads(target.to_json())["occurrence"] == {
        "indices": [0, 2],
        "reset_scope": "root_model_pass",
    }


def test_target_validation_uses_the_shared_path_grammar():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity(), nn.ReLU()])
            setattr(self, "custom/module", nn.Identity())

    model = Model()
    assert Target("layers[-1]", "activation", -1, (0,)).validate(model) is model.layers[-1]
    assert Target("<custom/module>", "activation", -1, (0,)).validate(model) is getattr(model, "custom/module")


def test_target_validation_does_not_invoke_descriptors():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.property_called = False

        @property
        def dangerous_property(self):
            self.property_called = True
            return nn.Identity()

    model = Model()
    with pytest.raises(ValueError, match="does not resolve"):
        Target("dangerous_property", "activation", -1, (0,)).validate(model)
    assert not model.property_called

    model.plain_value = object()
    with pytest.raises(ValueError, match="does not resolve"):
        Target("plain_value", "activation", -1, (0,)).validate(model)


def test_invalid_targets_have_clear_errors(default_test_model):
    with pytest.raises(ValueError, match="Invalid target kind"):
        Target("linear1", "other", 0, (0,))
    with pytest.raises(ValueError, match="at least one"):
        Target("linear1", "activation", 0, ())
    with pytest.raises(TypeError, match="integers"):
        Target("linear1", "activation", 0, ("unit",))
    with pytest.raises(TypeError, match="feature_axis"):
        Target("linear1", "activation", True, (0,))
    with pytest.raises(TypeError, match="feature_axis"):
        Target("linear1", "activation", 1.0, (0,))
    with pytest.raises(TypeError, match="indices"):
        Target("linear1", "activation", 0, (True,))
    with pytest.raises(ValueError, match="parameter targets require"):
        Target("linear1", "parameter", 0, (0,))
    with pytest.raises(ValueError, match="only valid for parameter"):
        Target("linear1", "activation", 0, (0,), parameter="weight")
    with pytest.raises(TypeError, match="output_path components"):
        Target("linear1", "activation", 0, (0,), output_path=(object(),))
    with pytest.raises(ValueError, match="output_path is only valid"):
        Target("linear1", "parameter", 0, (0,), parameter="weight", output_path=(0,))
    with pytest.raises(TypeError, match="occurrence"):
        Target("linear1", "activation", 0, (0,), occurrence=True)
    with pytest.raises(ValueError, match="non-negative"):
        Target("linear1", "activation", 0, (0,), occurrence=-1)
    with pytest.raises(ValueError, match="activation and gradient"):
        Target("linear1", "parameter", 0, (0,), parameter="weight", occurrence=0)
    with pytest.raises(TypeError, match="OccurrenceSelector"):
        Target("linear1", "activation", 0, (0,), occurrence=(0, 1))
    with pytest.raises(ValueError, match="missing indices"):
        Target.from_dict({"module_path": "linear1"})
    with pytest.raises(ValueError, match="JSON is invalid"):
        Target.from_json("not json")
    with pytest.raises(ValueError, match="contain an object"):
        Target.from_json("[]")
    with pytest.raises(ValueError, match="does not resolve"):
        Target("missing", "activation", 0, (0,)).validate(default_test_model)
    with pytest.raises(ValueError, match="does not resolve"):
        Target("linear1.weight", "activation", 0, (0,)).validate(default_test_model)
    with pytest.raises(ValueError, match="does not resolve"):
        Target(0, "activation", 0, (0,)).validate(default_test_model)

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
        Target("linear1", "activation", 1, (100,))._selection(torch.randn(1, 5))
    with pytest.raises(ValueError, match="feature_axis"):
        Target("linear1", "activation", 2, (0,))._selection(torch.randn(1, 2))


@pytest.mark.parametrize(
    ("indices", "exception", "message"),
    [
        ((), ValueError, "at least one"),
        ((True,), TypeError, "integers"),
        ((-1,), ValueError, "non-negative"),
        ((1, 1), ValueError, "unique"),
        ((2, 1), ValueError, "strictly increasing"),
    ],
)
def test_occurrence_selector_rejects_ambiguous_indices(indices, exception, message):
    with pytest.raises(exception, match=message):
        OccurrenceSelector(indices)

    assert OccurrenceSelector((0, 2)).reset_scope == "root_model_pass"
