"""Figure exports preserve readable labels and explicit board semantics."""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def figures(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "docs/source/notebooks/tutorials/notebook_figures.py"
    spec = importlib.util.spec_from_file_location("notebook_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    yield module
    plt.close("all")


def test_workflow_exports_editable_svg_and_high_resolution_png(figures, tmp_path):
    figures.workflow_figure(
        ["Capture", "Analyze", "Compute log-probabilities"],
        ["activation", "artifact"],
        "workflow",
    )
    svg = (tmp_path / "figures/workflow.svg").read_text()
    assert "<text" in svg and "activation" in svg and "Compute log-probabilities" in svg
    assert plt.gcf().axes[0].get_title() == ""
    assert not any(text.get_text().startswith("Result:") for text in plt.gcf().axes[0].texts)
    with Image.open(tmp_path / "figures/workflow.png") as image:
        assert image.width >= 3000
    with pytest.raises(ValueError, match="connection"):
        figures.workflow_figure(["Capture", "Analyze"], [], "invalid")


def test_board_labels_and_edit_location_are_preserved(figures):
    figure, axis = plt.subplots()
    board = np.zeros((8, 8), dtype=int)
    board[2, 3], board[3, 4] = 1, 2
    figures.draw_board(axis, board, "Board", highlight=(3, 4))
    assert len(axis.patches) == 3  # Two discs and one highlighted square.
    assert axis.patches[0].center == (3, 2)
    assert axis.patches[1].center == (4, 3)
    assert axis.patches[0].get_facecolor() != axis.patches[1].get_facecolor()
    assert axis.patches[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert [label.get_text() for label in axis.get_xticklabels()] == list("ABCDEFGH")
    with pytest.raises(ValueError, match="8 x 8"):
        figures.draw_board(axis, np.zeros((4, 4)), "Invalid")


def test_intervention_heatmap_uses_a_symmetric_effect_scale(figures, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "docs/source/notebooks/tutorials/othello_figure4.py"
    spec = importlib.util.spec_from_file_location("othello_figure4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "notebook_figures", figures)
    gain = np.full((8, 3), 0.3)
    gain[0, 0] = -0.001
    results = {
        "layers": list(range(1, 9)),
        "game_lengths": [0, 5, 10],
        "behavior_and_probe": {
            "layer_probe_accuracy": [0.95] * 8,
            "paper_targets": {"deep_layer_probe_accuracy": 0.995},
        },
        "summaries": {
            name: {"mean": gain.tolist()}
            for name in ("intervention_gain", "wrong_layer_gain", "randomized_probe_gain", "sham_gain")
        },
    }
    figure = module.plot_figure4_reproduction(results)
    image = figure.axes[1].images[0]
    assert image.norm.vmin == -image.norm.vmax == -0.3
    assert image.norm(0) == 0.5
    np.testing.assert_array_equal(image.get_array(), gain)
