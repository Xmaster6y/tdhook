"""Static contracts for the resource-intensive Othello reproduction."""

import importlib.util
from pathlib import Path

import nbformat
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "docs/source/notebooks/tutorials/othello-research-reproduction.ipynb"
FIGURE4_HELPER = REPO_ROOT / "docs/source/notebooks/tutorials/othello_figure4.py"
FIGURE4_ARTIFACT = REPO_ROOT / "docs/source/notebooks/assets/othello-figure4-results.json"


def _load_figure4_helper():
    spec = importlib.util.spec_from_file_location("othello_figure4", FIGURE4_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_othello_reproduction_uses_the_public_optimized_intervention_api():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")

    for public_api in (
        "InterventionObjective",
        "InterventionSpec",
        "OptimizerConfig",
        "EarlyStoppingConfig",
        "optimize_intervention",
    ):
        assert public_api in code
    assert "probe_board_objective" in code
    assert "optimization_artifact = optimized.to_dict()" in code
    assert "run_figure4_reproduction" in code
    assert "50-game layer-by-game-length causal sweep" in markdown
    assert "issue #107" in markdown
    assert "MPS path may produce silently incorrect results" in markdown


def test_othello_reproduction_code_cells_parse():
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")


def test_othello_figure4_uses_public_tdhook_capture_and_replacement():
    source = FIGURE4_HELPER.read_text()

    assert "with torch.inference_mode(), HookSession(model) as session:" in source
    assert "session.capture(_target(layer))" in source
    assert "session.replace(_target(layer), replacement)" in source
    assert "model.forward_1st_stage" in source
    assert "model.forward_2nd_stage" in source
    compile(source, str(FIGURE4_HELPER), "exec")


def test_othello_figure4_artifact_records_passing_scientific_gates():
    import json

    helper = _load_figure4_helper()
    artifact = json.loads(FIGURE4_ARTIFACT.read_text())

    assert artifact["protocol"]["number_games"] == 50
    assert artifact["protocol"]["controls"] == ["sham", "wrong_layer", "randomized_probe"]
    assert artifact["provenance"]["asset_sha256"] == helper.EXPECTED_SHA256
    assert artifact["ordering"]["mean_paired_gain_difference"] > 0
    assert artifact["ordering"]["bootstrap_95_ci"][0] > 0
    assert artifact["gates"] == {
        "deep_layer_probe_accuracy": True,
        "legal_move_rate": True,
        "tdhook_reference_parity": True,
        "sham_identity": True,
        "middle_over_late_ordering": True,
    }


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
