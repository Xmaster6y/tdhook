import json

import pytest
import torch
from torch import nn

from tdhook.attribution import (
    AttentionSite,
    FeatureSite,
    attention_contributions,
    attribute_feature_circuit,
    feature_contributions,
    logit_contributions,
)
from tdhook.targets import Target


class TargetFeature(nn.Module):
    def forward(self, features, attention):
        return features @ features.new_tensor([[2.0], [-1.0]]) + attention[:, 0]


class ToyCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = nn.Identity()
        self.pattern_hook = nn.Identity()
        self.values_hook = nn.Identity()
        self.attention_output = nn.Identity()
        self.target_feature = TargetFeature()
        self.logits = nn.Linear(1, 2, bias=False)
        self.register_buffer("pattern", torch.tensor([[[[0.25, 0.75]]]]))
        self.register_buffer("values", torch.tensor([[[[2.0]], [[4.0]]]]))
        self.output_weight = nn.Parameter(torch.tensor([[[2.0]]]))
        with torch.no_grad():
            self.logits.weight.copy_(torch.tensor([[2.0], [-1.0]]))

    def forward(self, inputs):
        features = self.upstream(inputs)
        pattern = self.pattern_hook(self.pattern)
        values = self.values_hook(self.values)
        attention = torch.einsum("bhqs,bshd,hdm->bqm", pattern, values, self.output_weight)
        attention = self.attention_output(attention)
        feature = self.target_feature(features, attention)
        return self.logits(feature)[0]


class ReusedUpstreamCircuit(ToyCircuit):
    def forward(self, inputs):
        features = self.upstream(inputs)
        features = self.upstream(features)
        pattern = self.pattern_hook(self.pattern)
        values = self.values_hook(self.values)
        attention = torch.einsum("bhqs,bshd,hdm->bqm", pattern, values, self.output_weight)
        feature = self.target_feature(features, self.attention_output(attention))
        return self.logits(feature)[0]


class UnreachedTargetCircuit(ToyCircuit):
    def forward(self, inputs):
        return self.logits(inputs[:, :1])[0]


class DisconnectedLogitsCircuit(ToyCircuit):
    def __init__(self):
        super().__init__()
        self.disconnected_logits = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, inputs):
        features = self.upstream(inputs)
        pattern = self.pattern_hook(self.pattern)
        values = self.values_hook(self.values)
        attention = torch.einsum("bhqs,bshd,hdm->bqm", pattern, values, self.output_weight)
        self.target_feature(features, self.attention_output(attention))
        return self.disconnected_logits


def target_site(*, indices=(0,), position=(0,)):
    return FeatureSite(1, Target("target_feature", "activation", -1, indices), position=position)


def upstream_site():
    return FeatureSite(0, Target("upstream", "activation", -1, (0, 1)))


def attention_site(model, *, pattern_heads=(0,), value_heads=(0,), target_position=0):
    return AttentionSite(
        layer=0,
        pattern=Target("pattern_hook", "activation", 1, pattern_heads),
        values=Target("values_hook", "activation", 2, value_heads),
        output_gradient=Target("attention_output", "gradient", -1, (0,)),
        output_weight=model.output_weight,
        target_position=target_position,
    )


def test_analytical_toy_circuit_verifies_all_three_attribution_paths():
    model = ToyCircuit()
    model.logits.weight.grad = torch.full_like(model.logits.weight, 7.0)
    original_parameter_grad = model.logits.weight.grad
    inputs = torch.tensor([[2.0, 1.0]], requires_grad=True)
    inputs.grad = torch.tensor([[5.0, 6.0]])
    original_input_grad = inputs.grad

    artifact = attribute_feature_circuit(
        model,
        inputs,
        target_feature=target_site(),
        upstream_features=(upstream_site(),),
        attention_sites=(attention_site(model),),
        output_logits=lambda output: output,
        logit_indices=(0, 1),
    )

    assert artifact.target_activation == pytest.approx(10.0)
    assert [(item.feature_index, item.position, item.score) for item in artifact.upstream_features] == [
        (0, (0,), 4.0),
        (1, (0,), -1.0),
    ]
    assert [(item.source_token, item.score) for item in artifact.attention] == [(1, 6.0), (0, 1.0)]
    assert [(item.token_index, item.score) for item in artifact.output_logits] == [(0, 20.0), (1, -10.0)]
    assert "local autograd Jacobian" in artifact.conventions.jacobian
    assert "frozen" in artifact.conventions.attention
    assert "frozen" in artifact.conventions.nonlinearities
    assert model.logits.weight.grad is not None
    assert model.logits.weight.grad is original_parameter_grad
    assert inputs.grad is original_input_grad
    torch.testing.assert_close(model.logits.weight.grad, torch.full_like(model.logits.weight, 7.0))
    torch.testing.assert_close(inputs.grad, torch.tensor([[5.0, 6.0]]))
    json.dumps(artifact.to_dict(), sort_keys=True)


def test_tensor_attribution_helpers_match_closed_form_scores():
    features = feature_contributions(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[2.0, -1.0]]),
        layer=0,
        feature_indices=(4, 7),
    )
    attention = attention_contributions(
        torch.tensor([[[0.25, 0.75]]]),
        torch.tensor([[[2.0]], [[4.0]]]),
        torch.tensor([[[2.0]]]),
        torch.tensor([[1.0]]),
        layer=0,
        target_position=0,
    )
    logits = logit_contributions(10.0, torch.tensor([2.0, -1.0]), token_indices=(3, 9))

    assert [item.score for item in features] == [4.0, -1.0]
    assert [item.score for item in attention] == [1.0, 6.0]
    assert [item.score for item in logits] == [20.0, -10.0]


def test_attribution_helpers_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="same shape"):
        feature_contributions(torch.ones(2), torch.ones(3), layer=0, feature_indices=(0, 1))
    with pytest.raises(ValueError, match="values must have shape"):
        attention_contributions(
            torch.ones(1, 1, 2),
            torch.ones(3, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1),
            layer=0,
            target_position=0,
        )
    with pytest.raises(ValueError, match="one index"):
        logit_contributions(1.0, torch.ones(2), token_indices=(0,))


def test_workflow_rejects_reused_hook_sites_instead_of_mispairing_captures():
    model = ReusedUpstreamCircuit()

    with pytest.raises(RuntimeError, match="upstream activation.*exactly once.*2 captures"):
        attribute_feature_circuit(
            model,
            torch.tensor([[2.0, 1.0]], requires_grad=True),
            target_feature=target_site(),
            upstream_features=(upstream_site(),),
        )


def test_attention_site_requires_matching_head_selections():
    model = ToyCircuit()

    with pytest.raises(ValueError, match="same heads in the same order"):
        attention_site(model, pattern_heads=(0,), value_heads=(1,))


def test_attention_helper_rejects_negative_target_positions():
    with pytest.raises(IndexError, match="target_position"):
        attention_contributions(
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1),
            layer=0,
            target_position=-1,
        )


def test_workflow_filters_and_limits_ranked_artifacts_without_logits():
    artifact = attribute_feature_circuit(
        ToyCircuit(),
        torch.tensor([[2.0, 1.0]], requires_grad=True),
        target_feature=target_site(),
        upstream_features=(upstream_site(),),
        top_k=1,
        positive_only=True,
    )

    assert [(item.feature_index, item.score) for item in artifact.upstream_features] == [(0, 4.0)]
    assert artifact.output_logits == ()


def test_workflow_infers_the_only_non_feature_position():
    artifact = attribute_feature_circuit(
        ToyCircuit(),
        torch.tensor([[2.0, 1.0]], requires_grad=True),
        target_feature=target_site(position=None),
    )

    assert artifact.target_position == (0,)


def test_workflow_restores_leaf_gradients_passed_in_keyword_mappings():
    inputs = torch.tensor([[2.0, 1.0]], requires_grad=True)
    inputs.grad = torch.tensor([[8.0, 9.0]])

    attribute_feature_circuit(
        ToyCircuit(),
        target_feature=target_site(),
        model_kwargs={"inputs": inputs},
    )

    torch.testing.assert_close(inputs.grad, torch.tensor([[8.0, 9.0]]))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FeatureSite(-1, Target("upstream", "activation", -1, (0,))), "layer"),
        (lambda: FeatureSite(0, Target("upstream", "gradient", -1, (0,))), "activation target"),
        (lambda: FeatureSite(0, Target("upstream", "activation", -1, (0,)), (-1,)), "non-negative"),
    ],
)
def test_feature_site_rejects_invalid_metadata(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer": -1}, "layer"),
        ({"target_position": -1}, "target_position"),
        ({"pattern": Target("pattern_hook", "gradient", 1, (0,))}, "activation targets"),
        ({"output_gradient": Target("attention_output", "activation", -1, (0,))}, "gradient target"),
    ],
)
def test_attention_site_rejects_invalid_metadata(overrides, message):
    model = ToyCircuit()
    arguments = {
        "layer": 0,
        "pattern": Target("pattern_hook", "activation", 1, (0,)),
        "values": Target("values_hook", "activation", 2, (0,)),
        "output_gradient": Target("attention_output", "gradient", -1, (0,)),
        "output_weight": model.output_weight,
        "target_position": 0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        AttentionSite(**arguments)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: feature_contributions(torch.ones(2), torch.ones(2), layer=-1, feature_indices=(0, 1)),
            "layer",
        ),
        (
            lambda: attention_contributions(
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1),
                layer=-1,
                target_position=0,
            ),
            "layer",
        ),
        (
            lambda: feature_contributions(torch.ones(2), torch.ones(2), layer=0, feature_indices=(0,)),
            "feature_indices",
        ),
        (
            lambda: feature_contributions(
                torch.ones(2), torch.ones(2), layer=0, feature_indices=(0, 1), feature_axis=1
            ),
            "feature_axis",
        ),
        (
            lambda: attention_contributions(
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1),
                torch.ones(1, 1),
                layer=0,
                target_position=0,
            ),
            "output_weight",
        ),
        (
            lambda: attention_contributions(
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 2),
                torch.ones(1, 1, 1),
                torch.ones(1, 1),
                layer=0,
                target_position=0,
            ),
            "head and head_dim",
        ),
        (
            lambda: attention_contributions(
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 2),
                torch.ones(1, 1),
                layer=0,
                target_position=0,
            ),
            "output_gradient",
        ),
        (
            lambda: attention_contributions(
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1),
                layer=0,
                target_position=0,
                head_indices=(0, 1),
            ),
            "head_indices",
        ),
        (
            lambda: attention_contributions(
                torch.ones(2, 1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1),
                layer=0,
                target_position=0,
            ),
            "singleton batch",
        ),
        (
            lambda: attention_contributions(
                torch.ones(1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1, 1),
                layer=0,
                target_position=0,
            ),
            "3 dimensions",
        ),
        (lambda: logit_contributions(torch.ones(2), torch.ones(1), token_indices=(0,)), "one scalar"),
        (lambda: logit_contributions(1.0, torch.ones(1), token_indices=(-1,)), "non-negative"),
    ],
)
def test_tensor_helpers_reject_invalid_contracts(call, message):
    with pytest.raises((ValueError, IndexError), match=message):
        call()


@pytest.mark.parametrize(
    ("model", "kwargs", "error", "message"),
    [
        (object(), {}, TypeError, "torch.nn.Module"),
        (ToyCircuit(), {"target_feature": target_site(indices=(0, 1))}, ValueError, "exactly one feature"),
        (ToyCircuit(), {"top_k": 0}, ValueError, "top_k"),
        (ToyCircuit(), {"logit_indices": (0,)}, ValueError, "output_logits"),
        (
            ToyCircuit(),
            {"output_logits": lambda output: output, "logit_indices": (-1,)},
            ValueError,
            "non-negative",
        ),
        (
            ToyCircuit(),
            {"output_logits": lambda output: output[0], "logit_indices": (0,)},
            ValueError,
            "one-dimensional",
        ),
        (
            ToyCircuit(),
            {"output_logits": lambda output: output, "logit_indices": (2,)},
            IndexError,
            "out of bounds",
        ),
        (UnreachedTargetCircuit(), {}, RuntimeError, "target feature activation.*observed 0"),
        (
            DisconnectedLogitsCircuit(),
            {"output_logits": lambda output: output, "logit_indices": (0,)},
            RuntimeError,
            "expected 1, observed 0",
        ),
    ],
)
def test_workflow_rejects_invalid_contracts(model, kwargs, error, message):
    arguments = {"target_feature": target_site(), **kwargs}

    with pytest.raises(error, match=message):
        attribute_feature_circuit(model, torch.tensor([[2.0, 1.0]], requires_grad=True), **arguments)


@pytest.mark.parametrize(
    ("inputs", "position", "message"),
    [
        (torch.tensor([[2.0, 1.0], [3.0, 1.0]], requires_grad=True), None, "position is required"),
        (torch.tensor([[2.0, 1.0]], requires_grad=True), (), "index every non-feature axis"),
        (torch.tensor([[2.0, 1.0]], requires_grad=True), (1,), "outside"),
    ],
)
def test_workflow_validates_target_position(inputs, position, message):
    with pytest.raises((ValueError, IndexError), match=message):
        attribute_feature_circuit(ToyCircuit(), inputs, target_feature=target_site(position=position))
