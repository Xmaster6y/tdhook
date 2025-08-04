"""
Time benchmark for tdhook.

Run with:

```
uv run --group scripts -m scripts.bench.runtime
```

This script measures the impact of batch size, model width, and model height
on runtime performance across different seeds.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from time import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import json
from pathlib import Path
from loguru import logger

from nnsight import NNsight
from tdhook.module import HookedModule
from .models import MLP

# Fixed seeds for reproducibility
SEEDS = [42, 123, 456]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def benchmark_model(
    model: nn.Module,
    input_data: torch.Tensor,
    seed: int,
    height: int,
) -> Dict[str, float]:
    """Benchmark a single model configuration with given seed."""
    set_seed(seed)

    # NNsight benchmark
    spawn_start = time()
    nnsight_model = NNsight(model)
    spawn_end = time()
    nnsight_spawn_time = spawn_end - spawn_start

    run_start = time()
    with nnsight_model.trace(input_data):
        hidden_states = nnsight_model.layers[height // 2].output
        nnsight_state = nnsight_model.layers[-1](hidden_states).save()
    run_end = time()
    nnsight_run_time = run_end - run_start

    # TDHook benchmark
    spawn_start = time()
    tdhook_model = HookedModule(model, in_keys=["input"], out_keys=["output"])
    spawn_end = time()
    tdhook_spawn_time = spawn_end - spawn_start

    run_start = time()
    with tdhook_model.run(TensorDict({"input": input_data})) as run:
        hidden_states = run.get(
            f"layers[{height // 2}]", callback=lambda **kwargs: tdhook_model.layers[-1](kwargs["output"])
        )
    run_end = time()
    tdhook_run_time = run_end - run_start

    # Verify results are consistent
    assert torch.allclose(nnsight_state, hidden_states.resolve(), atol=1e-6)

    return {
        "nnsight_spawn": nnsight_spawn_time,
        "nnsight_run": nnsight_run_time,
        "tdhook_spawn": tdhook_spawn_time,
        "tdhook_run": tdhook_run_time,
        "total_nnsight": nnsight_spawn_time + nnsight_run_time,
        "total_tdhook": tdhook_spawn_time + tdhook_run_time,
    }


def run_batch_size_benchmark():
    """Benchmark impact of batch size."""
    logger.info("Running batch size benchmark...")

    model = MLP(height=12, width=10)
    batch_sizes = [1, 8, 64, 256, 1024, 4096, 10_000, 20_000]
    results = {}

    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        batch_results = []

        for seed in SEEDS:
            input_data = torch.randn(batch_size, 10)
            result = benchmark_model(model, input_data, seed, height=12)
            batch_results.append(result)

        # Average and std across seeds
        avg_result = {}
        for key in batch_results[0].keys():
            values = [r[key] for r in batch_results]
            avg_result[key] = np.mean(values)
            avg_result[f"{key}_std"] = np.std(values)
        results[batch_size] = avg_result

    return results


def run_model_width_benchmark():
    """Benchmark impact of model width."""
    logger.info("Running model width benchmark...")

    widths = [8, 16, 32, 64, 128, 256, 512]
    results = {}

    for width in widths:
        logger.info(f"Testing model width: {width}")
        model = MLP(height=12, width=width)
        width_results = []

        for seed in SEEDS:
            input_data = torch.randn(1, width)
            result = benchmark_model(model, input_data, seed, height=12)
            width_results.append(result)

        # Average and std across seeds
        avg_result = {}
        for key in width_results[0].keys():
            values = [r[key] for r in width_results]
            avg_result[key] = np.mean(values)
            avg_result[f"{key}_std"] = np.std(values)
        results[width] = avg_result

    return results


def run_model_height_benchmark():
    """Benchmark impact of model height."""
    logger.info("Running model height benchmark...")

    heights = [4, 8, 12, 16, 20, 24, 32]
    results = {}

    for height in heights:
        logger.info(f"Testing model height: {height}")
        model = MLP(height=height, width=10)
        height_results = []

        for seed in SEEDS:
            input_data = torch.randn(1, 10)
            result = benchmark_model(model, input_data, seed, height)
            height_results.append(result)

        # Average and std across seeds
        avg_result = {}
        for key in height_results[0].keys():
            values = [r[key] for r in height_results]
            avg_result[key] = np.mean(values)
            avg_result[f"{key}_std"] = np.std(values)
        results[height] = avg_result

    return results


def plot_results(batch_results: Dict, width_results: Dict, height_results: Dict):
    """Create plots for the benchmark results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Batch size plots
    batch_sizes = list(batch_results.keys())
    nnsight_times = [batch_results[bs]["nnsight_run"] for bs in batch_sizes]
    tdhook_times = [batch_results[bs]["tdhook_run"] for bs in batch_sizes]
    nnsight_errors = [batch_results[bs]["nnsight_run_std"] for bs in batch_sizes]
    tdhook_errors = [batch_results[bs]["tdhook_run_std"] for bs in batch_sizes]

    axes[0, 0].errorbar(
        batch_sizes, nnsight_times, yerr=nnsight_errors, fmt="o-", label="NNsight", color="blue", capsize=3
    )
    axes[0, 0].errorbar(
        batch_sizes, tdhook_times, yerr=tdhook_errors, fmt="s-", label="TDHook", color="red", capsize=3
    )
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Run Time (s)")
    axes[0, 0].set_title("Batch Size Impact")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim(bottom=0)

    # Model width plots
    widths = list(width_results.keys())
    nnsight_times = [width_results[w]["nnsight_run"] for w in widths]
    tdhook_times = [width_results[w]["tdhook_run"] for w in widths]
    nnsight_errors = [width_results[w]["nnsight_run_std"] for w in widths]
    tdhook_errors = [width_results[w]["tdhook_run_std"] for w in widths]

    axes[0, 1].errorbar(widths, nnsight_times, yerr=nnsight_errors, fmt="o-", label="NNsight", color="blue", capsize=3)
    axes[0, 1].errorbar(widths, tdhook_times, yerr=tdhook_errors, fmt="s-", label="TDHook", color="red", capsize=3)
    axes[0, 1].set_xlabel("Model Width")
    axes[0, 1].set_ylabel("Run Time (s)")
    axes[0, 1].set_title("Model Width Impact")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(bottom=0)

    # Model height plots
    heights = list(height_results.keys())
    nnsight_times = [height_results[h]["nnsight_run"] for h in heights]
    tdhook_times = [height_results[h]["tdhook_run"] for h in heights]
    nnsight_errors = [height_results[h]["nnsight_run_std"] for h in heights]
    tdhook_errors = [height_results[h]["tdhook_run_std"] for h in heights]

    axes[0, 2].errorbar(
        heights, nnsight_times, yerr=nnsight_errors, fmt="o-", label="NNsight", color="blue", capsize=3
    )
    axes[0, 2].errorbar(heights, tdhook_times, yerr=tdhook_errors, fmt="s-", label="TDHook", color="red", capsize=3)
    axes[0, 2].set_xlabel("Model Height")
    axes[0, 2].set_ylabel("Run Time (s)")
    axes[0, 2].set_title("Model Height Impact")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].set_ylim(bottom=0)

    # Spawn time plots
    nnsight_spawn_times = [batch_results[bs]["nnsight_spawn"] for bs in batch_sizes]
    tdhook_spawn_times = [batch_results[bs]["tdhook_spawn"] for bs in batch_sizes]
    nnsight_spawn_errors = [batch_results[bs]["nnsight_spawn_std"] for bs in batch_sizes]
    tdhook_spawn_errors = [batch_results[bs]["tdhook_spawn_std"] for bs in batch_sizes]

    axes[1, 0].errorbar(
        batch_sizes, nnsight_spawn_times, yerr=nnsight_spawn_errors, fmt="o-", label="NNsight", color="blue", capsize=3
    )
    axes[1, 0].errorbar(
        batch_sizes, tdhook_spawn_times, yerr=tdhook_spawn_errors, fmt="s-", label="TDHook", color="red", capsize=3
    )
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Spawn Time (s)")
    axes[1, 0].set_title("Spawn Time vs Batch Size")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(bottom=0)

    nnsight_spawn_times = [width_results[w]["nnsight_spawn"] for w in widths]
    tdhook_spawn_times = [width_results[w]["tdhook_spawn"] for w in widths]
    nnsight_spawn_errors = [width_results[w]["nnsight_spawn_std"] for w in widths]
    tdhook_spawn_errors = [width_results[w]["tdhook_spawn_std"] for w in widths]

    axes[1, 1].errorbar(
        widths, nnsight_spawn_times, yerr=nnsight_spawn_errors, fmt="o-", label="NNsight", color="blue", capsize=3
    )
    axes[1, 1].errorbar(
        widths, tdhook_spawn_times, yerr=tdhook_spawn_errors, fmt="s-", label="TDHook", color="red", capsize=3
    )
    axes[1, 1].set_xlabel("Model Width")
    axes[1, 1].set_ylabel("Spawn Time (s)")
    axes[1, 1].set_title("Spawn Time vs Model Width")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(bottom=0)

    nnsight_spawn_times = [height_results[h]["nnsight_spawn"] for h in heights]
    tdhook_spawn_times = [height_results[h]["tdhook_spawn"] for h in heights]
    nnsight_spawn_errors = [height_results[h]["nnsight_spawn_std"] for h in heights]
    tdhook_spawn_errors = [height_results[h]["tdhook_spawn_std"] for h in heights]

    axes[1, 2].errorbar(
        heights, nnsight_spawn_times, yerr=nnsight_spawn_errors, fmt="o-", label="NNsight", color="blue", capsize=3
    )
    axes[1, 2].errorbar(
        heights, tdhook_spawn_times, yerr=tdhook_spawn_errors, fmt="s-", label="TDHook", color="red", capsize=3
    )
    axes[1, 2].set_xlabel("Model Height")
    axes[1, 2].set_ylabel("Spawn Time (s)")
    axes[1, 2].set_title("Spawn Time vs Model Height")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_ylim(bottom=0)

    plt.tight_layout()

    # Create results directory
    results_dir = Path("./results/bench/runtime")
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(results_dir / "benchmark_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def save_results(batch_results: Dict, width_results: Dict, height_results: Dict):
    """Save results to JSON file."""
    results = {
        "batch_size": batch_results,
        "model_width": width_results,
        "model_height": height_results,
        "seeds": SEEDS,
    }

    # Create results directory
    results_dir = Path("./results/bench/runtime")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    """Run all benchmarks."""
    logger.info("Starting comprehensive runtime benchmark...")
    logger.info(f"Using seeds: {SEEDS}")

    # Run benchmarks
    batch_results = run_batch_size_benchmark()
    width_results = run_model_width_benchmark()
    height_results = run_model_height_benchmark()

    # Save results
    save_results(batch_results, width_results, height_results)

    # Create plots
    plot_results(batch_results, width_results, height_results)

    logger.success("Benchmark completed!")
    logger.info("Results saved to './results/bench/runtime/benchmark_results.json'")
    logger.info("Plots saved to './results/bench/runtime/benchmark_results.png'")

    # Print summary
    logger.info("Summary:")
    logger.info("Batch size impact:")
    for bs, result in batch_results.items():
        logger.info(f"  Batch size {bs}: NNsight={result['nnsight_run']:.4f}s, TDHook={result['tdhook_run']:.4f}s")

    logger.info("Model width impact:")
    for w, result in width_results.items():
        logger.info(f"  Width {w}: NNsight={result['nnsight_run']:.4f}s, TDHook={result['tdhook_run']:.4f}s")

    logger.info("Model height impact:")
    for h, result in height_results.items():
        logger.info(f"  Height {h}: NNsight={result['nnsight_run']:.4f}s, TDHook={result['tdhook_run']:.4f}s")


if __name__ == "__main__":
    main()
