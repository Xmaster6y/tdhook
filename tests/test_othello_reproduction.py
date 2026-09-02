"""Static contracts for the resource-intensive Othello reproduction."""

from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "docs/source/notebooks/tutorials/othello-research-reproduction.ipynb"


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
    assert "50-game layer/game-length sweep" in markdown
    assert "issue #107" in markdown
    assert "MPS path may produce silently incorrect results" in markdown


def test_othello_reproduction_code_cells_parse():
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")
