import json
from importlib.metadata import version
from pathlib import Path
import tomllib

import tdhook


REPO_ROOT = Path(__file__).parents[1]


def test_version_matches_metadata() -> None:
    assert tdhook.__version__ == version("tdhook")


def test_release_version_is_consistent_across_metadata_and_docs() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project_version = project["project"]["version"]
    citation = (REPO_ROOT / "CITATION.cff").read_text().splitlines()
    switcher = json.loads((REPO_ROOT / "docs/source/_static/switcher.json").read_text())

    assert f"version: {project_version}" in citation
    assert switcher[0] == {
        "version": f"v{project_version}",
        "url": f"https://tdhook.readthedocs.io/en/v{project_version}/",
    }


def test_star_import_exposes_the_documented_core_modules() -> None:
    namespace = {}

    exec("from tdhook import *", namespace)

    assert namespace["modules"] is tdhook.modules
