import json
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
SWITCHER_PATH = REPO_ROOT / "docs/source/_static/switcher.json"


def test_documentation_switcher_targets_read_the_docs_version_slugs() -> None:
    entries = json.loads(SWITCHER_PATH.read_text())

    release_entries = [entry for entry in entries if entry["version"] != "dev"]
    assert release_entries
    for entry in release_entries:
        assert entry["version"].startswith("v")
        assert entry["url"] == f"https://tdhook.readthedocs.io/en/{entry['version']}/"

    assert entries[-1] == {
        "version": "dev",
        "url": "https://tdhook.readthedocs.io/en/latest/",
    }
