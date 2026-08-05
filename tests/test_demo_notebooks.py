"""Execute the deterministic notebooks that define TDHook's demo contract."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_NOTEBOOKS = (
    REPO_ROOT / "docs/source/notebooks/tutorials/declared-workflows.ipynb",
    REPO_ROOT / "docs/source/notebooks/methods/representation-similarity.ipynb",
)


def test_demo_notebooks_are_linked_from_the_top_level_docs():
    demos_page = (REPO_ROOT / "docs/source/demos.rst").read_text()

    for path in DEMO_NOTEBOOKS:
        relative_path = path.relative_to(REPO_ROOT / "docs/source").as_posix()
        assert relative_path in demos_page


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
