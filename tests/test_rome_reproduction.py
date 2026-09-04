"""Regression contracts for the ROME causal-tracing and editing reproduction."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "docs/source/notebooks/tutorials/rome-research-reproduction.ipynb"
HELPER = REPO_ROOT / "docs/source/notebooks/tutorials/rome_reproduction.py"
RESULT = REPO_ROOT / "docs/source/notebooks/assets/rome-issue-109-results.json"


def _load_helper():
    spec = importlib.util.spec_from_file_location("rome_reproduction", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rome_reproduction_is_linked_parseable_and_resource_bounded():
    tutorials = (REPO_ROOT / "docs/source/tutorials.rst").read_text()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert ":link: notebooks/tutorials/rome-research-reproduction" in tutorials
    assert notebook.metadata["tdhook"] == {
        "ci": False,
        "estimated_download_gb": 7,
        "estimated_vram_gb": 16,
        "network": True,
        "runtime": "cuda",
    }
    assert "issue #109" in markdown
    assert "three published facts" in markdown
    assert "ten counterfact edits" in markdown.lower()
    for required in (
        "causal_trace_grid",
        "temporary_rank_one_edit",
        "execute_rome",
        "compute_rewrite_quality_counterfact",
        "parity_report",
        "summarize_counterfact",
        "--untracked-files=no",
    ):
        assert required in code
    assert "deltas = execute_rome(model, tokenizer, request, hparams)" in code
    assert "deltas = execute_rome(model, tokenizer, [request], hparams)" not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")


def test_documented_rome_reproduction_contains_executed_results():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    result = json.loads(RESULT.read_text())

    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert any(cell.outputs for cell in code_cells)
    assert result["trace"]["case_ids"] == [0, 1, 2]
    assert result["trace"]["allclose"] is True
    assert result["counterfact"]["number_cases"] == 10
    assert result["temporary_edits_restored"] is True
    assert set(result).isdisjoint({"gates", "protocol", "protocol_sha256", "status"})


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

    with pytest.raises(ValueError, match="sentinel"):
        with helper.temporary_rank_one_edit(model, "", "weight", left, right):
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
