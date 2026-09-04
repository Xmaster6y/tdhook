"""Tests for the Othello reproduction helpers."""

import importlib.util
from pathlib import Path

import torch
from torch import nn

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
