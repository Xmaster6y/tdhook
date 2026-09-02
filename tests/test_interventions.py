import json
from collections import namedtuple

import pytest
import torch
from tensordict import TensorDict
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
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2))

    def forward(self, value, *, scale=1.0):
        self.forward_count.add_(1)
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
    optimize_intervention(model, (inputs,), spec, frozen_modules=(model, probe))

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
        optimize_intervention(model, (inputs,), failing, frozen_modules=(model, probe))

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
    different_seed = optimize_intervention(model, (inputs,), spec, seed=8)

    assert torch.equal(first.interventions[0].value, second.interventions[0].value)
    assert first.to_dict() == second.to_dict()
    assert not torch.equal(first.interventions[0].value, different_seed.interventions[0].value)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_explicit_initial_value_skips_capture_and_supports_numeric_terms():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda output, _value: InterventionObjective(output.square().sum(), {"constant": 1.5}),
        "explicit initial value",
        max_steps=1,
        initial_value=torch.ones(1, 1),
    )

    result = optimize_intervention(model, (torch.zeros(1, 1),), spec)

    assert result.stages[0].model_passes == 1
    assert result.stages[0].model_pass_budget == 1
    assert result.model_passes == 2
    assert result.model_pass_budget == 2
    assert result.stages[0].history[0].terms["constant"] == 1.5


def test_non_finite_intervention_gradient_is_an_explicit_outcome():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))

    def objective(_output, value):
        value.register_hook(lambda gradient: gradient * float("nan"))
        return value.sum()

    result = optimize_intervention(model, (torch.ones(1, 1),), InterventionSpec(target, objective, "nan gradient"))

    assert result.stages[0].status == "non_finite"
    assert result.stages[0].steps_completed == 1


def test_non_finite_optimizer_update_restores_last_finite_value():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda _output, value: -value.sum() * 1e38,
        "overflowing update",
        max_steps=1,
        initial_value=torch.zeros(1, 1),
        optimizer=OptimizerConfig("sgd", 10.0),
    )

    result = optimize_intervention(model, (torch.zeros(1, 1),), spec)

    assert result.stages[0].status == "non_finite"
    assert torch.equal(result.interventions[0].value, torch.zeros(1, 1))
    assert torch.equal(result.output, torch.zeros(1, 1))


def test_caller_input_gradient_is_restored_and_nested_inputs_are_discovered():
    class KeywordIdentity(nn.Module):
        def forward(self, value, *, duplicate):
            assert duplicate is value
            return value

    model = KeywordIdentity()
    value = torch.ones(1, 2, requires_grad=True)
    value.grad = torch.full_like(value, 7)
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(target, lambda output, _value: output.sum(), "input gradient", max_steps=1)

    optimize_intervention(model, (value,), spec, model_kwargs={"duplicate": value})

    assert torch.equal(value.grad, torch.full_like(value, 7))


def test_scalar_intervention_value_has_serializable_provenance():
    class ScalarModel(nn.Module):
        def forward(self):
            return torch.ones(1)

    model = ScalarModel()
    target = Target("", "activation", 0, (0,))
    spec = InterventionSpec(
        target,
        lambda output, _value: output.sum(),
        "scalar intervention",
        max_steps=1,
        initial_value=torch.ones(()),
    )

    result = optimize_intervention(model, (), spec)

    assert result.stages[0].value_shape == ()
    assert len(result.stages[0].value_sha256) == 64


def test_final_structured_output_does_not_alias_restored_module_state():
    Aliases = namedtuple("Aliases", ("parameter", "buffers"))

    class AliasingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("running", torch.ones(1))

        def forward(self, value):
            with torch.no_grad():
                self.weight.add_(1)
                self.running.add_(1)
            return {
                "value": value,
                "aliases": Aliases(self.weight, [self.running]),
                "tensor_dict": TensorDict({"buffer": self.running}, batch_size=[]),
                "metadata": "kept",
            }

    model = AliasingModel()
    target = Target("", "activation", -1, (0,), output_path=("value",))
    spec = InterventionSpec(
        target,
        lambda output, _value: output["value"].sum(),
        "aliasing output",
        max_steps=1,
        initial_value=torch.ones(1),
    )

    result = optimize_intervention(model, (torch.ones(1),), spec)

    assert torch.equal(model.weight, torch.ones(1))
    assert torch.equal(model.running, torch.ones(1))
    assert isinstance(result.output["aliases"], Aliases)
    assert torch.equal(result.output["aliases"].parameter, torch.full((1,), 3.0))
    assert torch.equal(result.output["aliases"].buffers[0], torch.full((1,), 3.0))
    assert torch.equal(result.output["tensor_dict"]["buffer"], torch.full((1,), 3.0))
    assert result.output["metadata"] == "kept"


@pytest.mark.parametrize(
    "objective,error,match",
    [
        (lambda _output, _value: object(), TypeError, "tensor or InterventionObjective"),
        (lambda _output, _value: InterventionObjective(torch.ones(()), []), TypeError, "terms must be a mapping"),
        (
            lambda _output, _value: InterventionObjective(torch.ones(()), {"loss": 1}),
            ValueError,
            "reserve the name 'loss'",
        ),
        (lambda _output, _value: InterventionObjective(1.0), TypeError, "objective loss must be a tensor"),
        (lambda _output, _value: torch.ones(2), ValueError, "objective loss must be scalar"),
        (
            lambda _output, _value: InterventionObjective(torch.ones(()), {"": 1}),
            ValueError,
            "term names must be non-empty",
        ),
        (
            lambda _output, _value: InterventionObjective(torch.ones(()), {"invalid": True}),
            TypeError,
            "must be numeric",
        ),
    ],
)
def test_objective_contract_validation(objective, error, match):
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))

    with pytest.raises(error, match=match):
        optimize_intervention(model, (torch.ones(1, 1),), InterventionSpec(target, objective, "invalid objective"))


@pytest.mark.parametrize(
    "regularizer,error,match",
    [
        (lambda _value, _reference: 1.0, TypeError, "regularizer must be a tensor"),
        (lambda _value, _reference: torch.ones(2), ValueError, "regularizer must be scalar"),
    ],
)
def test_preservation_regularizer_contract_validation(regularizer, error, match):
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda output, _value: output.sum(),
        "invalid regularizer",
        preservation_regularizer=regularizer,
    )

    with pytest.raises(error, match=match):
        optimize_intervention(model, (torch.ones(1, 1),), spec)


def test_objective_terms_reserve_preservation_name():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(
        target,
        lambda output, _value: InterventionObjective(output.sum(), {"preservation": output.sum()}),
        "reserved term",
        preservation_regularizer=lambda value, reference: (value - reference).square().sum(),
    )

    with pytest.raises(ValueError, match="reserve the name 'preservation'"):
        optimize_intervention(model, (torch.ones(1, 1),), spec)


@pytest.mark.parametrize(
    "factory,error,match",
    [
        (lambda: OptimizerConfig("invalid"), ValueError, "unsupported optimizer"),
        (lambda: OptimizerConfig(learning_rate=True), TypeError, "learning_rate must be a number"),
        (lambda: OptimizerConfig(learning_rate=0), ValueError, "learning_rate must be positive"),
        (lambda: OptimizerConfig(kwargs=[]), TypeError, "kwargs must be a mapping"),
        (lambda: OptimizerConfig(kwargs={"invalid": object()}), TypeError, "kwargs must be JSON-serializable"),
        (lambda: EarlyStoppingConfig(objective_threshold=True), TypeError, "threshold must be a number"),
        (lambda: EarlyStoppingConfig(objective_threshold=float("inf")), ValueError, "threshold must be finite"),
        (lambda: EarlyStoppingConfig(min_delta=True), TypeError, "min_delta must be a number"),
        (lambda: EarlyStoppingConfig(min_delta=-1), ValueError, "min_delta must be non-negative"),
        (lambda: EarlyStoppingConfig(patience=True), TypeError, "patience must be an integer"),
        (lambda: EarlyStoppingConfig(patience=0), ValueError, "patience must be positive"),
        (
            lambda: InterventionSpec(Target("", "activation", -1, (0,)), 1, "objective"),
            TypeError,
            "objective must be callable",
        ),
        (
            lambda: InterventionSpec(Target("", "activation", -1, (0,)), lambda output, value: value.sum(), 1),
            TypeError,
            "objective_name must be a string",
        ),
        (
            lambda: InterventionSpec(Target("", "activation", -1, (0,)), lambda output, value: value.sum(), ""),
            ValueError,
            "objective_name must be non-empty",
        ),
        (
            lambda: InterventionSpec(
                Target("", "activation", -1, (0,)), lambda output, value: value.sum(), "steps", max_steps=True
            ),
            TypeError,
            "max_steps must be an integer",
        ),
        (
            lambda: InterventionSpec(
                Target("", "activation", -1, (0,)), lambda output, value: value.sum(), "steps", max_steps=0
            ),
            ValueError,
            "max_steps must be positive",
        ),
        (
            lambda: InterventionSpec(
                Target("", "activation", -1, (0,)), lambda output, value: value.sum(), "initial", initial_value=1
            ),
            TypeError,
            "initial_value must be a tensor",
        ),
        (
            lambda: InterventionSpec(
                Target("", "activation", -1, (0,)),
                lambda output, value: value.sum(),
                "regularizer",
                preservation_regularizer=1,
            ),
            TypeError,
            "preservation_regularizer must be callable",
        ),
        (
            lambda: InterventionSpec(
                Target("", "activation", -1, (0,)),
                lambda output, value: value.sum(),
                "direction",
                direction="bwd",
            ),
            ValueError,
            "forward hook direction",
        ),
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


@pytest.mark.parametrize(
    "model,model_args,specs,kwargs,error,match",
    [
        (object(), (), (), {}, TypeError, "model must be a torch.nn.Module"),
        (nn.Identity(), "invalid", (), {}, TypeError, "model_args must be a sequence"),
        (nn.Identity(), (), "invalid", {}, TypeError, "specs must be a sequence"),
        (nn.Identity(), (), (), {}, ValueError, "at least one intervention"),
        (nn.Identity(), (), (object(),), {}, TypeError, "every spec must be an InterventionSpec"),
        (nn.Identity(), (), None, {}, TypeError, "specs must be a sequence"),
    ],
)
def test_optimized_intervention_input_validation(model, model_args, specs, kwargs, error, match):
    with pytest.raises(error, match=match):
        optimize_interventions(model, model_args, specs, **kwargs)


def test_optimized_intervention_keyword_input_validation():
    model = nn.Identity()
    target = Target("", "activation", -1, (0,))
    spec = InterventionSpec(target, lambda output, _value: output.sum(), "valid")

    invalid = (
        ({"model_kwargs": []}, "model_kwargs must be a mapping"),
        ({"frozen_modules": (object(),)}, "frozen_modules must contain"),
        ({"seed": True}, "seed must be an integer"),
    )
    for kwargs, match in invalid:
        with pytest.raises(TypeError, match=match):
            optimize_intervention(model, (torch.ones(1, 1),), spec, **kwargs)
