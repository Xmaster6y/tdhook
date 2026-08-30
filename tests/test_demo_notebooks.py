"""Execute the deterministic notebooks that define TDHook's demo contract."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DOCUMENTS = (
    ("tutorials.rst", "notebooks/tutorials/declared-workflows.ipynb"),
    ("tutorials.rst", "notebooks/methods/representation-similarity.ipynb"),
)
DEMO_NOTEBOOKS = tuple(REPO_ROOT / "docs/source" / path for _, path in DEMO_DOCUMENTS)


def test_demo_notebooks_are_linked_from_the_top_level_docs():
    for document, notebook in DEMO_DOCUMENTS:
        document_text = (REPO_ROOT / "docs/source" / document).read_text()
        assert notebook.removesuffix(".ipynb") in document_text


@pytest.mark.integration
@pytest.mark.parametrize("path", DEMO_NOTEBOOKS, ids=lambda path: path.stem)
def test_demo_notebook_executes_from_a_fresh_kernel(path: Path):
    notebook = nbformat.read(path, as_version=4)

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
