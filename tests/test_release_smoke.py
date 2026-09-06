import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_smoke", REPO_ROOT / "scripts/release_smoke.py")
release_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_smoke)


@pytest.fixture
def installed_package(monkeypatch, tmp_path):
    package_file = Path("tdhook/__init__.py")
    installed = SimpleNamespace(version="1.2.3", files=[package_file], locate_file=lambda path: tmp_path / path)
    monkeypatch.setattr(release_smoke, "distribution", lambda name: installed)
    monkeypatch.setattr(release_smoke.tdhook, "__file__", str(tmp_path / package_file))
    return installed


def test_release_gate_accepts_the_matching_installed_package(installed_package):
    release_smoke.check_installed_distribution("1.2.3")


def test_release_gate_rejects_a_different_version(installed_package):
    with pytest.raises(RuntimeError, match="Expected tdhook 1.2.4, found 1.2.3"):
        release_smoke.check_installed_distribution("1.2.4")


@pytest.mark.parametrize("files", [None, [Path("__editable__.tdhook.pth")]])
def test_release_gate_rejects_an_editable_install(installed_package, files):
    installed_package.files = files
    with pytest.raises(RuntimeError, match="not an editable checkout"):
        release_smoke.check_installed_distribution("1.2.3")


def test_release_gate_rejects_source_shadowing(installed_package, monkeypatch, tmp_path):
    monkeypatch.setattr(release_smoke.tdhook, "__file__", str(tmp_path / "src/tdhook/__init__.py"))
    with pytest.raises(RuntimeError, match="instead of installed distribution"):
        release_smoke.check_installed_distribution("1.2.3")


def test_release_gate_executes_the_readme_and_capture_workflow():
    release_smoke.check_readme(REPO_ROOT / "README.md")
    release_smoke.check_capture_workflow()


def test_release_gate_rejects_a_broken_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text('```python\nraise RuntimeError("broken quickstart")\n```\n')
    with pytest.raises(RuntimeError, match="broken quickstart"):
        release_smoke.check_readme(readme)


def test_release_gate_rejects_hook_leaks():
    model = nn.Linear(3, 4)
    handle = model.register_forward_hook(lambda module, args, output: None)
    try:
        with pytest.raises(RuntimeError, match="left hooks installed"):
            release_smoke.check_hook_cleanup(model)
    finally:
        handle.remove()
