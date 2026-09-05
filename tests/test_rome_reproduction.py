"""Tests for the ROME reproduction helpers."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "docs/source/notebooks/tutorials/rome_reproduction.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("rome_reproduction", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_numpy_corruption_matches_the_released_rome_rng_convention():
    helper = _load_helper()
    value = torch.ones(3, 2, 4)
    actual = helper._corrupt_subject(value, seed=1, noise_level=0.1, replace=False)
    expected = value[1:] + torch.from_numpy(np.random.RandomState(1).randn(2, 2, 4)) * 0.1

    assert torch.equal(actual[0], value[0])
    torch.testing.assert_close(actual[1:], expected.to(actual))


def test_temporary_rank_one_edit_is_visible_only_inside_the_session():
    helper = _load_helper()
    model = nn.Linear(3, 2, bias=False)
    original = model.weight.detach().clone()
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([4.0, 5.0])

    with (
        pytest.raises(ValueError, match="sentinel"),
        helper.temporary_rank_one_edit(model, "", "weight", left, right),
    ):
        torch.testing.assert_close(model.weight, original + torch.outer(left, right).T)
        raise ValueError("sentinel")

    assert torch.equal(model.weight, original)


def test_tdhook_trace_runs_clean_and_corrupted_items_together_and_cleans_hooks():
    helper = _load_helper()

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.wte = nn.Embedding(5, 5)
            self.layer = nn.Identity()

        def forward(self, input_ids, attention_mask):
            hidden = self.layer(self.transformer.wte(input_ids))
            return SimpleNamespace(logits=hidden)

    model = TinyLM()
    inputs = {
        "input_ids": torch.tensor([[1, 2], [1, 2], [1, 2]]),
        "attention_mask": torch.ones(3, 2, dtype=torch.long),
    }
    score = helper.trace_with_patch_tdhook(
        model,
        inputs,
        [(1, "layer")],
        answer_token=2,
        subject_range=(0, 1),
        config=helper.CausalTraceConfig(samples=2),
    )

    assert score.ndim == 0 and torch.isfinite(score)
    assert not model.transformer.wte._forward_hooks
    assert not model.layer._forward_hooks

    corrupted = model.transformer.wte(inputs["input_ids"])
    corrupted[:, 1:2] = helper._corrupt_subject(
        corrupted[:, 1:2],
        seed=1,
        noise_level=0.1,
        replace=False,
    )
    expected_corrupted = torch.softmax(corrupted[1:, -1], dim=-1)[:, 2].mean()
    unpatched = helper.trace_with_patch_tdhook(
        model,
        inputs,
        [],
        answer_token=2,
        subject_range=(1, 2),
        config=helper.CausalTraceConfig(samples=2),
    )
    torch.testing.assert_close(unpatched, expected_corrupted)


def test_causal_trace_declares_intervention_and_metric_steps():
    helper = _load_helper()
    workflow = helper.causal_trace_workflow(
        [(1, "layer")],
        answer_token=2,
        subject_range=(0, 1),
        config=helper.CausalTraceConfig(samples=2),
    )

    assert [type(step).__name__ for step in workflow.steps] == ["SteeringVectors", "TensorDictModule"]


def test_residual_grid_selects_the_tensor_from_gpt2_style_block_outputs():
    helper = _load_helper()

    class TupleBlock(nn.Module):
        def forward(self, value):
            return value, None

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.wte = nn.Embedding(5, 5)
            self.layer = TupleBlock()

        def forward(self, input_ids, attention_mask):
            hidden, _ = self.layer(self.transformer.wte(input_ids))
            return SimpleNamespace(logits=hidden)

    model = TinyLM()
    inputs = {
        "input_ids": torch.tensor([[1, 2], [1, 2], [1, 2]]),
        "attention_mask": torch.ones(3, 2, dtype=torch.long),
    }
    scores = helper.causal_trace_grid(
        model,
        inputs,
        answer_token=2,
        subject_range=(0, 1),
        layer_paths=["layer"],
        config=helper.CausalTraceConfig(samples=2),
    )

    assert scores.shape == (2, 1)


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_window_grid_rejects_non_positive_or_non_integer_windows(window):
    helper = _load_helper()

    with pytest.raises(ValueError, match="positive integer"):
        helper.causal_trace_window_grid(
            nn.Identity(),
            {"input": torch.ones(2, 1)},
            answer_token=0,
            subject_range=(0, 1),
            layer_paths=[""],
            component="mlp",
            window=window,
            config=helper.CausalTraceConfig(samples=1),
        )


def test_counterfact_reduction_uses_official_negative_log_likelihood_ordering():
    helper = _load_helper()
    metrics = {
        "rewrite_prompts_probs": [{"target_new": 1.0, "target_true": 2.0}],
        "paraphrase_prompts_probs": [
            {"target_new": 1.0, "target_true": 2.0},
            {"target_new": 3.0, "target_true": 2.0},
        ],
        "neighborhood_prompts_probs": [{"target_new": 2.0, "target_true": 1.0}],
    }
    assert helper.case_score(metrics) == {
        "rewrite_efficacy": 1.0,
        "paraphrase_generalization": 0.5,
        "neighborhood_specificity": 1.0,
    }
