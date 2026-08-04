"""Static compatibility checks for the public Colab notebook paths."""

import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    "bilinear-probing.ipynb",
    "dimension-estimation.ipynb",
    "integrated-gradients.ipynb",
    "linear-probing.ipynb",
    "representation-similarity.ipynb",
    "steering-vectors.ipynb",
)
PINNED_SOURCE = "tdhook @ git+https://github.com/Xmaster6y/tdhook.git@66fa52ddb6cb25d593e182c3d21bc21dff83ec56"
NOTEBOOK_DIR = Path("docs/source/notebooks/methods")


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "".join("".join(cell.get("source", ())) for cell in notebook["cells"])


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_public_colab_notebook_pins_its_install_source(filename: str):
    source = notebook_source(NOTEBOOK_DIR / filename)

    assert 'IN_COLAB = importlib.util.find_spec("google.colab") is not None' in source
    assert PINNED_SOURCE in source
    assert "git clone https://github.com/Xmaster6y/tdhook -b main" not in source


def test_dimension_estimation_declares_its_extra_colab_dependency():
    assert "scikit-learn" in notebook_source(NOTEBOOK_DIR / "dimension-estimation.ipynb")
