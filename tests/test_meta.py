import json
import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import tdhook

REPO_ROOT = Path(__file__).parents[1]


def test_version_matches_metadata() -> None:
    assert tdhook.__version__ == version("tdhook")


def test_release_version_is_consistent_across_metadata_and_docs() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = project["project"]["version"]
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines()
    switcher = json.loads((REPO_ROOT / "docs/source/_static/switcher.json").read_text(encoding="utf-8"))

    assert f"version: {project_version}" in citation
    assert switcher[0] == {
        "version": f"v{project_version}",
        "url": f"https://tdhook.readthedocs.io/en/v{project_version}/",
    }


def test_star_import_exposes_the_documented_core_modules() -> None:
    namespace = {}

    exec("from tdhook import *", namespace)

    assert namespace["modules"] is tdhook.modules


def test_notebook_citations_resolve_to_bibliography_entries() -> None:
    bibliography = (REPO_ROOT / "docs/source/references.bib").read_text(encoding="utf-8")
    bibliography_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))

    citation_keys = set()
    for notebook_path in (REPO_ROOT / "docs/source/notebooks").rglob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for citation_group in re.findall(r'data-cite="([^"]+)"', markdown):
            citation_keys.update(key.strip() for key in citation_group.split(","))

    assert citation_keys <= bibliography_keys
