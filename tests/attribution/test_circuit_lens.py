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


def test_analytical_toy_circuit_verifies_all_three_attribution_paths():
    model = ToyCircuit()
    model.logits.weight.grad = torch.full_like(model.logits.weight, 7.0)
    inputs = torch.tensor([[2.0, 1.0]], requires_grad=True)
    target = FeatureSite(1, Target("target_feature", "activation", -1, (0,)), position=(0,))
    upstream = FeatureSite(0, Target("upstream", "activation", -1, (0, 1)))
    attention = AttentionSite(
        layer=0,
        pattern=Target("pattern_hook", "activation", 1, (0,)),
        values=Target("values_hook", "activation", 2, (0,)),
        output_gradient=Target("attention_output", "gradient", -1, (0,)),
        output_weight=model.output_weight,
        target_position=0,
    )

    artifact = attribute_feature_circuit(
        model,
        inputs,
        target_feature=target,
        upstream_features=(upstream,),
        attention_sites=(attention,),
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
    torch.testing.assert_close(model.logits.weight.grad, torch.full_like(model.logits.weight, 7.0))
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
