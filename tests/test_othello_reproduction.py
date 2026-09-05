"""Tests for the Othello reproduction helpers."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from tdhook.workflow import Workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE4_HELPER = REPO_ROOT / "docs/source/notebooks/tutorials/othello_figure4.py"


def _load_figure4_helper():
    spec = importlib.util.spec_from_file_location("othello_figure4", FIGURE4_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_othello_inverse_map_returns_the_post_update_converged_value():
    helper = _load_figure4_helper()
    initial = torch.zeros(1, 3)
    desired_board = torch.full((1, 64), 2, dtype=torch.long)
    probe = (torch.eye(3).repeat(64, 1), torch.zeros(64 * 3))

    value, steps, final_loss = helper._inverse_map(
        initial,
        desired_board,
        probe,
        max_steps=20,
        learning_rate=0.1,
    )
    returned_logits = helper._probe_logits(value, probe)
    returned_loss = torch.nn.functional.cross_entropy(returned_logits.flatten(0, -2), desired_board.flatten())

    assert steps == 1
    assert torch.equal(returned_logits.argmax(-1), desired_board)
    torch.testing.assert_close(torch.tensor(final_loss), returned_loss)


def test_prediction_view_is_a_serializable_workflow_output():
    helper = _load_figure4_helper()
    logits = torch.randn(1, 3, 61)
    workflow = Workflow(
        TensorDictModule(
            helper.prediction_view,
            ["reference", "clean", "intervention", "sham", "scores"],
            ["view"],
        )
    )
    result = workflow(
        nn.Identity(),
        TensorDict(
            {
                "reference": logits,
                "clean": logits + 1,
                "intervention": logits * 2,
                "sham": logits + 1,
                "scores": NonTensorData({"clean": 0.5, "intervention": 0.9}),
            },
            [],
        ),
    )
    view = json.loads(json.dumps(result["view"]))
    assert view["sham_max_abs_logit_difference"] == 0
    torch.testing.assert_close(torch.tensor(view["probabilities"])[2], (logits[0, -1] * 2).softmax(-1))
    assert view["scores"]["intervention"] == 0.9


def test_othello_capture_and_replacement_use_workflow_artifacts():
    helper = _load_figure4_helper()

    class TinyOthello(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity()])

        def forward(self, inputs):
            return self.blocks[0](inputs), None

    model = TinyOthello()
    inputs = torch.randn(2, 3, 512)
    captured, logits = helper._capture(model, inputs, layer=0)
    replacement = torch.zeros_like(captured)
    replaced = helper._replace(model, inputs, layer=0, replacement=replacement)

    torch.testing.assert_close(captured, inputs)
    torch.testing.assert_close(logits, inputs)
    torch.testing.assert_close(replaced, replacement)
    assert not model.blocks[0]._forward_hooks


def test_othello_metrics_capture_all_layers_of_explicit_tensordict_wrapper():
    helper = _load_figure4_helper()

    class TinyOthello(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity() for _ in range(8)])

        def forward(self, inputs):
            hidden = torch.ones(*inputs.shape, 3)
            for block in self.blocks:
                hidden = block(hidden)
            logits = torch.zeros(*inputs.shape, 61)
            logits[..., 1] = 1
            return logits, None

    class Board:
        def __init__(self):
            self.state = np.zeros((8, 8), dtype=np.int64)

        def umpire(self, move):
            pass

        def get_valid_moves(self):
            return [0]

    bias = torch.tensor([0.0, 1.0, 0.0]).repeat(64)
    probes = [(torch.zeros(192, 3), bias) for _ in range(8)]
    games = np.zeros((100, 59), dtype=np.int64)
    model = TinyOthello()
    metrics = helper._probe_and_behavior_metrics(
        model,
        probes,
        games,
        games,
        SimpleNamespace(OthelloBoardState=Board),
        torch.device("cpu"),
    )
    assert metrics["layer_probe_accuracy"] == [1.0] * 8
    assert metrics["legal_move_rate"] == 1.0
    assert metrics["evaluated_next_move_positions"] == 5800
    assert all(not block._forward_hooks for block in model.blocks)


def test_option_workflow_consumes_replacement_artifacts(monkeypatch):
    helper = _load_figure4_helper()

    class TinyOthello(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity() for _ in range(8)])

        def forward(self, inputs):
            for block in self.blocks:
                inputs = block(inputs)
            return inputs, None

    # Fix the solver result; this test checks composition, not convergence.
    monkeypatch.setattr(helper, "_inverse_map", lambda initial, *args: (initial + 1, 1, 0.0))
    model = TinyOthello()
    inputs = torch.randn(1, 2, 512)
    # The released model's helper reshapes to 512 dimensions.
    probe = (torch.randn(192, 512), torch.zeros(192))
    workflow = helper.option_intervention_workflow(model, 0, probe, probe)
    result = workflow(model, TensorDict({"reference_input": inputs, "alternative_input": inputs.clone()}, []))
    assert len(workflow.steps) == 8
    assert set(result["metrics", "scores"]) == {"clean", "intervention", "sham", "wrong_layer", "randomized_probe"}
    torch.testing.assert_close(result["logits", "intervention"], result["replacement", "intervention"])

    consumer = Workflow(workflow.steps[3])
    with pytest.raises(ValueError, match="missing"):
        consumer(model, TensorDict({"alternative_input": inputs}, []))
    replacement = torch.zeros_like(inputs)
    changed = consumer(
        model,
        TensorDict(
            {
                "alternative_input": inputs,
                ("replacement", "intervention"): replacement,
            },
            [],
        ),
    )
    torch.testing.assert_close(changed["logits", "intervention"], replacement)
    assert all(not block._forward_hooks for block in model.blocks)
