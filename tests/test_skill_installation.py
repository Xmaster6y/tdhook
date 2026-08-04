"""Smoke checks for the standalone tdhook agent skill."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


SKILL_ROOT = Path(__file__).parents[1] / "skills" / "tdhook"
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def _local_markdown_links(document: Path) -> list[Path]:
    links = []
    for target in MARKDOWN_LINK.findall(document.read_text()):
        target = target.split("#", maxsplit=1)[0]
        if target and "://" not in target and not target.startswith("mailto:"):
            links.append(Path(target))
    return links


def test_skill_references_resolve_outside_the_source_checkout(tmp_path: Path):
    """An installed skill must resolve its own local references without this repo."""
    installed_skill = tmp_path / "consumer-project" / "skills" / "tdhook"
    shutil.copytree(SKILL_ROOT, installed_skill)

    for document in installed_skill.rglob("*.md"):
        for target in _local_markdown_links(document):
            resolved = (document.parent / target).resolve()
            assert resolved.is_relative_to(installed_skill)
            assert resolved.is_file(), f"{document.relative_to(installed_skill)} links to missing {target}"


def test_source_tree_navigation_is_explicitly_contributor_only():
    source_tree_guide = (SKILL_ROOT / "references" / "file_structure.md").read_text()

    assert source_tree_guide.startswith("# Contributor-only Source-tree Navigation")
    assert "not needed to install or use the skill" in source_tree_guide


def test_normal_skill_guidance_does_not_require_a_source_checkout():
    contributor_guide = SKILL_ROOT / "references" / "file_structure.md"

    for document in SKILL_ROOT.rglob("*.md"):
        if document != contributor_guide:
            assert "src/tdhook" not in document.read_text()
