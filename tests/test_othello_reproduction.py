"""Static contracts for the resource-intensive Othello reproduction."""

from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "docs/source/notebooks/tutorials/othello-research-reproduction.ipynb"
FIGURE4_HELPER = REPO_ROOT / "docs/source/notebooks/tutorials/othello_figure4.py"
FIGURE4_ARTIFACT = REPO_ROOT / "docs/source/notebooks/assets/othello-figure4-results.json"


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

    artifact = json.loads(FIGURE4_ARTIFACT.read_text())

    assert artifact["protocol"]["number_games"] == 50
    assert artifact["protocol"]["controls"] == ["sham", "wrong_layer", "randomized_probe"]
    assert len(artifact["provenance"]["asset_sha256"]) == 10
    assert artifact["ordering"]["mean_paired_gain_difference"] > 0
    assert artifact["ordering"]["bootstrap_95_ci"][0] > 0
    assert all(artifact["gates"].values())
