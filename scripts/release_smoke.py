"""Smoke-test an installed TDHook distribution, never the source checkout."""

import re
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

import tdhook
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow


def check_installed_distribution(expected_version: str) -> None:
    installed = distribution("tdhook")
    if installed.version != expected_version:
        raise RuntimeError(f"Expected tdhook {expected_version}, found {installed.version}")

    package_file = Path("tdhook/__init__.py")
    if package_file not in (installed.files or ()):
        raise RuntimeError("TDHook must be installed from a distribution, not an editable checkout")
    expected_path = Path(installed.locate_file(package_file)).resolve()
    if Path(tdhook.__file__).resolve() != expected_path:
        raise RuntimeError(f"Imported {tdhook.__file__} instead of installed distribution {expected_path}")

    print(
        f"Python {sys.version.split()[0]}, tdhook {installed.version}, "
        f"torch {version('torch')}, tensordict {version('tensordict')}\n"
        f"Imported TDHook from {expected_path}",
        flush=True,
    )


def check_hook_cleanup(model: nn.Module) -> None:
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks:
            raise RuntimeError("TDHook left hooks installed after context exit")


def check_readme(readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")
    start = text.index("```python\n") + len("```python\n")
    example = text[start : text.index("\n```", start)]
    namespace = {}
    exec(compile(example, str(readme_path), "exec"), namespace)  # noqa: S102
    attributions = namespace["attributions"]
    if attributions.shape != namespace["inputs"].shape or not torch.isfinite(attributions).all():
        raise RuntimeError("README attribution must be finite and match the input shape")
    check_hook_cleanup(namespace["model"])
    print("README attribution passed", flush=True)


def check_capture_workflow() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    caller = TensorDictModule(model, in_keys=["input"], out_keys=["output"])
    for wrapped, layer_name in ((model, "0"), (caller, "module.0")):
        capture = ActivationCaching(re.escape(layer_name) + "$", cache_key=("activations", "hidden"))
        workflow = Workflow(
            capture,
            TensorDictModule(
                lambda activation: activation.abs().sum(-1),
                in_keys=[("activations", "hidden", layer_name)],
                out_keys=[("metrics", "activation_mass")],
            ),
        )
        for offset in (0.0, 1.0):
            inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset
            with torch.no_grad():
                expected_output = model(inputs)
                expected_hidden = model[0](inputs)
            with capture.prepare(wrapped) as hooked:
                standalone = hooked(TensorDict({"input": inputs}, batch_size=[2]))
            check_hook_cleanup(model)
            result = workflow(wrapped, TensorDict({"input": inputs}, batch_size=[2]))
            for captured in (standalone, result):
                torch.testing.assert_close(captured["output"], expected_output)
                torch.testing.assert_close(captured["activations", "hidden"][layer_name], expected_hidden)
            torch.testing.assert_close(result["metrics", "activation_mass"], expected_hidden.abs().sum(-1))
            if caller.out_keys != ["output"]:
                raise RuntimeError("Activation caching changed the caller's output contract")
            check_hook_cleanup(model)
    print("Repeated standalone capture and capture-to-analysis workflow passed", flush=True)


def main(expected_version: str, readme_path: Path) -> None:
    check_installed_distribution(expected_version)
    torch.manual_seed(0)
    check_readme(readme_path)
    check_capture_workflow()
    print(f"tdhook {expected_version} installed-package smoke passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: release_smoke.py EXPECTED_VERSION README_PATH")
    main(sys.argv[1], Path(sys.argv[2]))
