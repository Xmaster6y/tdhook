"""Versioned correctness-first benchmarks for representative TDHook APIs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor, nn

from tdhook.attribution import Saliency
from tdhook.session import HookSession
from tdhook.targets import Target


SCHEMA_VERSION = 1
SUITE_NAME = "tdhook-maintained"


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    width: int
    depth: int
    selected_features: int
    warmup: int
    repeats: int


CONFIGS = {
    "smoke": BenchmarkConfig(4, 32, 3, 8, 1, 3),
    "full": BenchmarkConfig(128, 512, 6, 64, 5, 30),
}


class BenchmarkMLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend((nn.Linear(width, width), nn.GELU()))
        self.blocks = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(self.blocks(value)).squeeze(-1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing(operation: Callable[[], object], config: BenchmarkConfig, device: torch.device) -> dict[str, object]:
    for _ in range(config.warmup):
        operation()
    _synchronize(device)

    samples: list[int] = []
    for _ in range(config.repeats):
        started = time.perf_counter_ns()
        operation()
        _synchronize(device)
        samples.append(time.perf_counter_ns() - started)
    return {
        "unit": "ns",
        "samples": samples,
        "median": int(statistics.median(samples)),
        "minimum": min(samples),
    }


def _memory(operation: Callable[[], object], device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        _synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        operation()
        _synchronize(device)
        peak = max(0, torch.cuda.max_memory_allocated(device) - baseline)
        return {"unit": "byte", "peak": peak, "method": "torch.cuda.max_memory_allocated_delta"}

    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"unit": "byte", "peak": peak, "method": "tracemalloc_python_allocations"}


def _implementation_result(
    operation: Callable[[], object], config: BenchmarkConfig, device: torch.device
) -> dict[str, object]:
    return {"timing": _timing(operation, config, device), "memory": _memory(operation, device)}


def _agreement(actual: Tensor, expected: Tensor) -> dict[str, object]:
    actual = actual.detach()
    expected = expected.detach()
    max_abs_error = float((actual - expected).abs().max().cpu())
    passed = torch.allclose(actual, expected, rtol=1e-5, atol=1e-7)
    result = {"passed": passed, "max_abs_error": max_abs_error, "rtol": 1e-5, "atol": 1e-7}
    if not passed:
        raise RuntimeError(f"correctness check failed: {result}")
    return result


def _capture_benchmark(
    model: BenchmarkMLP, inputs: Tensor, config: BenchmarkConfig, device: torch.device
) -> dict[str, object]:
    indices = tuple(range(config.selected_features))
    target = Target("blocks.0", "activation", -1, indices)

    def tdhook_operation() -> Tensor:
        with HookSession(model) as session:
            captured = session.capture(target)
            model(inputs)
        assert captured.value is not None
        return captured.value

    def torch_operation() -> Tensor:
        captured: list[Tensor] = []

        def hook(_module: nn.Module, _args: tuple[object, ...], output: Tensor) -> None:
            captured.append(output[:, indices].detach().clone())

        handle = model.blocks[0].register_forward_hook(hook)
        try:
            model(inputs)
        finally:
            handle.remove()
        return captured[0]

    correctness = _agreement(tdhook_operation(), torch_operation())
    return {
        "name": "activation_capture",
        "reference": "torch.nn.Module.register_forward_hook",
        "correctness": correctness,
        "implementations": {
            "tdhook": _implementation_result(tdhook_operation, config, device),
            "reference": _implementation_result(torch_operation, config, device),
        },
    }


def _intervention_benchmark(
    model: BenchmarkMLP, inputs: Tensor, config: BenchmarkConfig, device: torch.device
) -> dict[str, object]:
    indices = tuple(range(config.selected_features))
    target = Target("blocks.0", "activation", -1, indices)

    def tdhook_operation() -> Tensor:
        with HookSession(model) as session:
            session.replace(target, 0.0)
            return model(inputs)

    def torch_operation() -> Tensor:
        def hook(_module: nn.Module, _args: tuple[object, ...], output: Tensor) -> Tensor:
            replaced = output.clone()
            replaced[:, indices] = 0.0
            return replaced

        handle = model.blocks[0].register_forward_hook(hook)
        try:
            return model(inputs)
        finally:
            handle.remove()

    correctness = _agreement(tdhook_operation(), torch_operation())
    return {
        "name": "activation_intervention",
        "reference": "torch.nn.Module.register_forward_hook",
        "correctness": correctness,
        "implementations": {
            "tdhook": _implementation_result(tdhook_operation, config, device),
            "reference": _implementation_result(torch_operation, config, device),
        },
    }


def _attribution_benchmark(
    model: BenchmarkMLP, inputs: Tensor, config: BenchmarkConfig, device: torch.device
) -> dict[str, object]:
    try:
        from captum.attr import Saliency as CaptumSaliency
    except ImportError as exc:  # pragma: no cover - exercised outside development installs
        raise RuntimeError("the benchmark suite requires the 'benchmark' optional dependency") from exc

    tdhook = Saliency()
    captum = CaptumSaliency(model)

    def tdhook_operation() -> Tensor:
        values = inputs.detach().clone().requires_grad_(True)
        batch = TensorDict({"input": values}, batch_size=[config.batch_size])
        with tdhook.prepare(model) as hooked_model:
            result = hooked_model(batch)
        return result.get(("attr", "input"))

    def captum_operation() -> Tensor:
        values = inputs.detach().clone().requires_grad_(True)
        return captum.attribute(values, abs=False)

    correctness = _agreement(tdhook_operation(), captum_operation())
    return {
        "name": "input_saliency",
        "reference": "captum.attr.Saliency",
        "correctness": correctness,
        "implementations": {
            "tdhook": _implementation_result(tdhook_operation, config, device),
            "reference": _implementation_result(captum_operation, config, device),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dirty() -> bool | None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.strip())


def _environment(device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        hardware_name = torch.cuda.get_device_name(device)
    else:
        hardware_name = platform.processor() or platform.machine()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": {"device": str(device), "name": hardware_name},
        "commit": _commit(),
        "dirty": _dirty(),
        "versions": {
            "tdhook": _package_version("tdhook"),
            "torch": torch.__version__,
            "tensordict": _package_version("tensordict"),
            "captum": _package_version("captum"),
        },
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_suite(mode: str, device_name: str = "auto") -> dict[str, Any]:
    config = CONFIGS[mode]
    device = resolve_device(device_name)
    torch.manual_seed(0)
    model = BenchmarkMLP(config.width, config.depth).eval().to(device)
    inputs = torch.randn(config.batch_size, config.width, device=device)

    benchmarks = [
        _capture_benchmark(model, inputs, config, device),
        _intervention_benchmark(model, inputs, config, device),
        _attribution_benchmark(model, inputs, config, device),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "configuration": asdict(config),
        "environment": _environment(device),
        "benchmarks": benchmarks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CONFIGS, default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_suite(args.mode, args.device)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
