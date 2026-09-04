"""Static contracts for the resource-intensive multi-step notebooks."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "docs/source/notebooks/tutorials"
MULTISTEP_NOTEBOOKS = (
    "chess-dimension-estimation.ipynb",
    "concept-attribution.ipynb",
    "othello-research-reproduction.ipynb",
    "rome-research-reproduction.ipynb",
    "weight-circuit-research-reproduction.ipynb",
)


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _compilable_code(path: Path) -> str:
    lines = []
    for line in _code(path).splitlines():
        stripped = line.lstrip()
        lines.append(f"{line[: len(line) - len(stripped)]}pass" if stripped.startswith(("%", "!")) else line)
    return "\n".join(lines)


@pytest.mark.parametrize("filename", MULTISTEP_NOTEBOOKS)
def test_multistep_notebook_code_compiles(filename: str):
    path = NOTEBOOK_DIR / filename
    compile(_compilable_code(path), str(path), "exec")


def test_multistep_reproductions_declare_workflows():
    chess_dimension = _code(NOTEBOOK_DIR / "chess-dimension-estimation.ipynb")
    concept_attribution = _code(NOTEBOOK_DIR / "concept-attribution.ipynb")
    othello = _code(NOTEBOOK_DIR / "othello-research-reproduction.ipynb")
    weight_circuit = _code(NOTEBOOK_DIR / "weight-circuit-research-reproduction.ipynb")
    rome_helper = (NOTEBOOK_DIR / "rome_reproduction.py").read_text(encoding="utf-8")

    assert "capture_workflow = Workflow(" in chess_dimension
    assert "dimension_workflow = Workflow(" in chess_dimension
    assert "relevance_workflow = Workflow(" in concept_attribution
    assert "Workflow(lrp, reduce_map)" in concept_attribution
    assert "representation_workflow = Workflow(" in othello
    assert "prepare_intervention = Workflow(" in othello
    assert "weight_lens_workflow = Workflow(" in weight_circuit
    assert "circuit_workflow = Workflow(" in weight_circuit
    assert "def causal_trace_workflow(" in rome_helper
    assert "return Workflow(" in rome_helper
