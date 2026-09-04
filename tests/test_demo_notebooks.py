"""Execute the deterministic notebooks that define TDHook's demo contract."""

from copy import deepcopy
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DOCUMENTS = (
    ("tutorials.rst", "notebooks/tutorials/hook-session.ipynb"),
    ("tutorials.rst", "notebooks/tutorials/declared-workflows.ipynb"),
    ("tutorials.rst", "notebooks/tutorials/process-and-distributed-workflows.ipynb"),
    ("tutorials.rst", "notebooks/methods/representation-similarity.ipynb"),
)
DEMO_NOTEBOOKS = tuple(REPO_ROOT / "docs/source" / path for _, path in DEMO_DOCUMENTS)
ALL_NOTEBOOKS = tuple((REPO_ROOT / "docs/source/notebooks").rglob("*.ipynb"))


def _stable_outputs(notebook):
    return [
        [
            deepcopy(output)
            for output in cell.outputs
            if not (output.output_type == "stream" and output.name == "stderr")
        ]
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda path: path.stem)
def test_documented_notebook_has_saved_successful_execution(path: Path):
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(cell.source.strip() for cell in code_cells)
    assert all(cell.execution_count is not None for cell in code_cells)
    assert all(output.output_type != "error" for cell in code_cells for output in cell.outputs)


def test_demo_notebooks_are_linked_from_the_top_level_docs():
    for document, notebook in DEMO_DOCUMENTS:
        document_lines = (REPO_ROOT / "docs/source" / document).read_text().splitlines()
        expected_link = f":link: {notebook.removesuffix('.ipynb')}"
        assert expected_link in {line.strip() for line in document_lines}


@pytest.mark.integration
@pytest.mark.parametrize("path", DEMO_NOTEBOOKS, ids=lambda path: path.stem)
def test_demo_notebook_executes_from_a_fresh_kernel(path: Path):
    notebook = nbformat.read(path, as_version=4)
    committed_outputs = _stable_outputs(notebook)

    assert notebook.metadata["tdhook"] == {
        "ci": True,
        "network": False,
        "runtime": "cpu",
    }

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    assert _stable_outputs(notebook) == committed_outputs
