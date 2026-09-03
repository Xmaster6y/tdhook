"""Static compatibility checks for the public Colab notebook paths."""

import json
from pathlib import Path
import re

import pytest


NOTEBOOKS = (
    "bilinear-probing.ipynb",
    "dimension-estimation.ipynb",
    "integrated-gradients.ipynb",
    "linear-probing.ipynb",
    "representation-similarity.ipynb",
    "steering-vectors.ipynb",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "docs/source/notebooks/methods"


def notebook_code_cells(path: Path) -> tuple[str, ...]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return tuple("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")


def colab_setup_cell(path: Path) -> str:
    return next(source for source in notebook_code_cells(path) if "%pip install" in source)


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_public_colab_notebook_installs_the_released_package(filename: str):
    code_cells = notebook_code_cells(NOTEBOOK_DIR / filename)
    setup = colab_setup_cell(NOTEBOOK_DIR / filename)

    assert any('IN_COLAB = importlib.util.find_spec("google.colab") is not None' in cell for cell in code_cells)
    assert "%pip install -q tdhook" in setup
    assert "git+https://github.com/Xmaster6y/tdhook" not in setup
    assert "git clone https://github.com/Xmaster6y/tdhook -b main" not in setup


def test_dimension_estimation_declares_its_extra_colab_dependency():
    assert "scikit-learn" in colab_setup_cell(NOTEBOOK_DIR / "dimension-estimation.ipynb")


def test_bilinear_probing_does_not_force_absolute_paths():
    code = "\n".join(notebook_code_cells(NOTEBOOK_DIR / "bilinear-probing.ipynb"))

    assert re.search(r"\brelative\s*=\s*False\b", code) is None


def test_readme_colab_badges_load_notebooks_from_main():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for filename in NOTEBOOKS:
        assert f"blob/main/docs/source/notebooks/methods/{filename}" in readme
