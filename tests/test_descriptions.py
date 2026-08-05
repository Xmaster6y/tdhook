import json

import pytest
import torch
from tensordict import TensorDict

from tdhook.attribution import IntegratedGradients
from tdhook.execution import GradientMode
from tdhook.latent import ActivationCaching
from tdhook.targets import Target
from tdhook.workflow import Workflow


def test_method_description_is_immutable_json_compatible_and_changes_with_target():
    first = ActivationCaching(Target("linear1", "activation", -1, (1,)))
    second = ActivationCaching(Target("linear1", "activation", -1, (2,)))

    description = first.describe()

    assert description != second.describe()
    assert description.parameters["key_pattern"]["indices"] == (1,)
    assert description.execution == {"model_passes": 1, "gradient_mode": "optional", "autograd_lifetime": "call"}
    json.dumps(description.to_dict(), sort_keys=True)
    with pytest.raises(TypeError, match="immutable"):
        description.parameters["relative"] = False


def test_integrated_gradient_description_captures_baseline_and_step_settings():
    description = IntegratedGradients(baseline_key=("inputs", "baseline"), n_steps=8).describe()

    assert description.parameters["baseline_key"] == ("inputs", "baseline")
    assert description.parameters["n_steps"] == 8
    assert description.execution == {
        "model_passes": 1,
        "gradient_mode": GradientMode.REQUIRED.value,
        "autograd_lifetime": "call",
    }


def test_callbacks_require_explicit_stable_identifiers():
    def select_output(**kwargs):
        return kwargs["output"]

    method = ActivationCaching("linear1", callback=select_output)
    with pytest.raises(TypeError, match="requires an explicit stable identifier"):
        method.describe()

    description = method.describe(callback_identifiers={select_output: "tests.select-output/v1"})
    assert description.parameters["callback"] == {"identifier": "tests.select-output/v1"}


def test_workflow_description_includes_bound_input_and_output_keys(default_test_model):
    workflow = Workflow(ActivationCaching("linear1", cache_key=("activations", "linear1")))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    (description,) = workflow.describe(default_test_model, data)

    assert description.in_keys == ("input",)
    assert description.out_keys == ("output", ("activations", "linear1"))
