import json

import pytest
import torch
from torch import nn

from tdhook.interventions import (
    EarlyStoppingConfig,
    InterventionObjective,
    InterventionSpec,
    OptimizerConfig,
    optimize_intervention,
    optimize_interventions,
)
from tdhook.targets import Target


class StructuredSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Identity()
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2))

    def forward(self, value, *, scale=1.0):
        hidden = self.second(self.first(value)) * scale
        return {"hidden": hidden, "score": hidden.sum()}


def squared_score(target):
    def objective(output, _intervention):
        error = (output["score"] - target).square()
        return InterventionObjective(error, {"score": output["score"]})

    return objective


def test_optimized_intervention_handles_structured_outputs_and_selected_positions():
    model = StructuredSequenceModel()
    inputs = torch.zeros(1, 3, 2)
    target = Target("first", "activation", 1, (1,))
    spec = InterventionSpec(
        target,
        squared_score(4.0),
        "selected-position score",
        max_steps=3,
        optimizer=OptimizerConfig("sgd", 0.0625),
        early_stopping=EarlyStoppingConfig(objective_threshold=0.0),
    )

    result = optimize_intervention(model, (inputs,), spec, model_kwargs={"scale": 2.0})

    assert torch.equal(result.output["hidden"][:, (0, 2)], torch.zeros(1, 2, 2))
    assert result.output["score"].item() == pytest.approx(4.0)
    assert result.interventions[0].value.shape == (1, 1, 2)
    assert result.stages[0].status == "converged"
    assert result.stages[0].steps_completed == 2
    assert result.model_passes == 4
    assert result.model_pass_budget == 5
    artifact = result.to_dict()
    assert artifact["stages"][0]["target"] == target.to_dict()
    assert artifact["stages"][0]["history"][0]["terms"] == {"loss": 16.0, "score": 0.0}
    json.dumps(artifact, sort_keys=True)
    assert not model.first._forward_hooks


def test_sequential_interventions_keep_earlier_optimized_values_active():
    model = nn.Sequential(nn.Identity(), nn.Identity())
    inputs = torch.zeros(1, 2)
    first = Target("0", "activation", -1, (0,))
    second = Target("1", "activation", -1, (1,))
    config = OptimizerConfig("sgd", 0.5)
    stopping = EarlyStoppingConfig(objective_threshold=0.0)

    result = optimize_interventions(
        model,
        (inputs,),
        (
            InterventionSpec(
                first,
                lambda output, _value: (output[:, 0] - 2).square().sum(),
                "set first feature",
                max_steps=3,
                optimizer=config,
                early_stopping=stopping,
            ),
            InterventionSpec(
                second,
                lambda output, _value: (output.sum() - 5).square(),
                "set total with first fixed",
                max_steps=3,
                optimizer=config,
                early_stopping=stopping,
            ),
        ),
    )

    assert torch.equal(result.output, torch.tensor([[2.0, 3.0]]))
    assert [stage.status for stage in result.stages] == ["converged", "converged"]
    assert result.stages[1].history[0].terms["loss"] == pytest.approx(9.0)
    assert not model[0]._forward_hooks
    assert not model[1]._forward_hooks


def test_preservation_regularizer_and_budget_exhaustion_are_reported():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda output, _value: InterventionObjective((output - 2).square().sum(), {"task": output.sum()}),
        "regularized target",
        max_steps=1,
        optimizer=OptimizerConfig("sgd", 0.1),
        preservation_regularizer=lambda value, reference: 0.5 * (value - reference).square().sum(),
    )

    result = optimize_intervention(model, (torch.zeros(1, 1),), spec)

    assert result.stages[0].status == "budget_exhausted"
    assert result.stages[0].history[0].terms == {"loss": 4.0, "task": 0.0, "preservation": 0.0}
    assert result.stages[0].model_passes == 2
    assert result.stages[0].model_pass_budget == 2


def test_non_finite_objective_is_an_explicit_outcome():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(target, lambda output, _value: output.sum() * float("nan"), "nan objective")

    result = optimize_intervention(model, (torch.ones(1, 1),), spec)

    assert result.stages[0].status == "non_finite"
    assert result.stages[0].steps_completed == 1
    assert result.model_passes == 3


def test_stalled_objective_is_an_explicit_early_stop():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda _output, value: value.sum() * 0 + 1,
        "constant objective",
        max_steps=10,
        early_stopping=EarlyStoppingConfig(patience=2),
    )

    result = optimize_intervention(model, (torch.ones(1, 1),), spec)

    assert result.stages[0].status == "stalled"
    assert result.stages[0].steps_completed == 3
    assert result.model_passes == 5


def test_model_and_probe_state_are_restored_after_success_and_failure():
    model = StructuredSequenceModel()
    probe = nn.Linear(2, 1)
    inputs = torch.zeros(1, 3, 2)
    target = Target("first", "activation", 1, (1,))
    original_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_probe = {name: value.detach().clone() for name, value in probe.state_dict().items()}
    model.first.weight.grad = torch.ones_like(model.first.weight)
    original_grad = model.first.weight.grad.clone()

    spec = InterventionSpec(
        target,
        lambda output, _value: probe(output["hidden"][:, 1]).square().sum(),
        "frozen probe",
        max_steps=2,
    )
    optimize_intervention(model, (inputs,), spec, frozen_modules=(probe,))

    assert all(torch.equal(model.state_dict()[name], value) for name, value in original_model.items())
    assert all(torch.equal(probe.state_dict()[name], value) for name, value in original_probe.items())
    assert torch.equal(model.first.weight.grad, original_grad)
    assert model.first.weight.requires_grad
    assert probe.weight.requires_grad

    def failing_objective(_output, _value):
        with torch.no_grad():
            model.first.weight.fill_(123)
            probe.bias.fill_(456)
        raise RuntimeError("injected objective failure")

    failing = InterventionSpec(target, failing_objective, "failure", max_steps=1)
    with pytest.raises(RuntimeError, match="injected objective failure"):
        optimize_intervention(model, (inputs,), failing, frozen_modules=(probe,))

    assert all(torch.equal(model.state_dict()[name], value) for name, value in original_model.items())
    assert all(torch.equal(probe.state_dict()[name], value) for name, value in original_probe.items())
    assert not model.first._forward_hooks


def test_seeded_optimization_is_deterministic_without_advancing_caller_rng():
    model = nn.Sequential(nn.Dropout(0.5), nn.Identity())
    target = Target("1", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda output, _value: (output.sum() - 1).square(),
        "dropout objective",
        max_steps=2,
    )
    inputs = torch.ones(2, 2)
    rng_state = torch.random.get_rng_state()

    first = optimize_intervention(model, (inputs,), spec, seed=7)
    second = optimize_intervention(model, (inputs,), spec, seed=7)

    assert torch.equal(first.interventions[0].value, second.interventions[0].value)
    assert first.to_dict() == second.to_dict()
    assert torch.equal(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize(
    "factory,error,match",
    [
        (lambda: OptimizerConfig("invalid"), ValueError, "unsupported optimizer"),
        (lambda: EarlyStoppingConfig(patience=0), ValueError, "patience must be positive"),
        (
            lambda: InterventionSpec(Target("", "gradient", -1, (0,)), lambda output, value: value.sum(), "gradient"),
            ValueError,
            "activation targets",
        ),
    ],
)
def test_intervention_configuration_validation(factory, error, match):
    with pytest.raises(error, match=match):
        factory()
