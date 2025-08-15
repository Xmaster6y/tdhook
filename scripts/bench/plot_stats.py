"""
Plot runtime and memory benchmark results.

Run with:

```
uv run --group scripts -m scripts.bench.plot_stats
```
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from loguru import logger


def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation of a list of values."""
    if not values:
        return 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values)
    return mean, std


def extract_parameter_data(results: Dict, task: str, script: str, parameter: str, metric: str) -> Dict:
    """Extract data for a specific parameter and metric from results."""
    if task not in results or script not in results[task] or parameter not in results[task][script]:
        return {}

    param_data = results[task][script][parameter]
    extracted_data = {}

    for param_value, seeds_data in param_data.items():
        values = []
        for seed_data in seeds_data.values():
            # Extract the specific metric from spawn or run data
            if metric == "spawn":
                if "spawn" in seed_data and "wall_time" in seed_data["spawn"]:
                    values.append(seed_data["spawn"]["wall_time"])
            elif metric == "run":
                if "run" in seed_data and "wall_time" in seed_data["run"]:
                    values.append(seed_data["run"]["wall_time"])
            elif metric == "spawn_memory":
                if "spawn" in seed_data and "max_memory_used_kb" in seed_data["spawn"]:
                    values.append(seed_data["spawn"]["max_memory_used_kb"] / 1024)  # Convert to MB
            elif metric == "run_memory":
                if "run" in seed_data and "max_memory_used_kb" in seed_data["run"]:
                    values.append(seed_data["run"]["max_memory_used_kb"] / 1024)  # Convert to MB

        if values:
            mean, std = compute_stats(values)
            extracted_data[param_value] = {"mean": mean, "std": std, "values": values}

    return extracted_data


def plot_metric_comparison(
    results: Dict, task: str, parameter: str, metric: str, base_value: float, ax: plt.Axes, title: str, ylabel: str
):
    """Plot comparison between NNsight and TDHook for a specific metric."""
    # Extract data for both scripts
    nnsight_data = extract_parameter_data(results, task, "nnsight", parameter, metric)
    tdhook_data = extract_parameter_data(results, task, "tdhook", parameter, metric)

    if not nnsight_data or not tdhook_data:
        logger.warning(f"No data found for {task}/{parameter}/{metric}")
        return

    # Get common parameter values and sort as integers
    param_values = sorted(set(nnsight_data.keys()) & set(tdhook_data.keys()), key=int)
    if not param_values:
        return

    # Extract means and stds
    nnsight_means = [nnsight_data[pv]["mean"] for pv in param_values]
    nnsight_stds = [nnsight_data[pv]["std"] for pv in param_values]
    tdhook_means = [tdhook_data[pv]["mean"] for pv in param_values]
    tdhook_stds = [tdhook_data[pv]["std"] for pv in param_values]

    # Plot with error bars
    ax.errorbar(
        param_values,
        nnsight_means,
        yerr=nnsight_stds,
        fmt="o-",
        label="NNsight",
        color="blue",
        capsize=3,
        markersize=6,
    )
    ax.errorbar(
        param_values, tdhook_means, yerr=tdhook_stds, fmt="s-", label="TDHook", color="red", capsize=3, markersize=6
    )

    # Add base line
    ax.axhline(y=base_value, color="green", linestyle="--", linewidth=2, label=f"Base: {base_value:.2f}", alpha=0.7)

    ax.set_xlabel(parameter.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_spawn_time_group(results: Dict, task: str = "mlp_intervene"):
    """Create spawn time plots for all parameters."""
    base_spawn_time = results["base"]["spawn"]["wall_time"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot spawn time vs width
    plot_metric_comparison(
        results, task, "width", "spawn", base_spawn_time, axes[0, 0], "Spawn Time vs Model Width", "Spawn Time (s)"
    )

    # Plot spawn time vs height
    plot_metric_comparison(
        results, task, "height", "spawn", base_spawn_time, axes[0, 1], "Spawn Time vs Model Height", "Spawn Time (s)"
    )

    # Plot spawn time vs batch_size
    plot_metric_comparison(
        results, task, "batch_size", "spawn", base_spawn_time, axes[1, 0], "Spawn Time vs Batch Size", "Spawn Time (s)"
    )

    # Plot spawn time vs variations
    plot_variations_metric(
        results, task, "spawn", base_spawn_time, axes[1, 1], "Spawn Time vs Variations", "Spawn Time (s)"
    )

    plt.tight_layout()

    results_dir = Path("./results/bench/stats")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "spawn_time_benchmark_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_run_time_group(results: Dict, task: str = "mlp_intervene"):
    """Create run time plots for all parameters."""
    base_run_time = results["base"]["run"]["wall_time"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot run time vs width
    plot_metric_comparison(
        results, task, "width", "run", base_run_time, axes[0, 0], "Run Time vs Model Width", "Run Time (s)"
    )

    # Plot run time vs height
    plot_metric_comparison(
        results, task, "height", "run", base_run_time, axes[0, 1], "Run Time vs Model Height", "Run Time (s)"
    )

    # Plot run time vs batch_size
    plot_metric_comparison(
        results, task, "batch_size", "run", base_run_time, axes[1, 0], "Run Time vs Batch Size", "Run Time (s)"
    )

    # Plot run time vs variations
    plot_variations_metric(results, task, "run", base_run_time, axes[1, 1], "Run Time vs Variations", "Run Time (s)")

    plt.tight_layout()

    results_dir = Path("./results/bench/stats")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "run_time_benchmark_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_spawn_memory_group(results: Dict, task: str = "mlp_intervene"):
    """Create spawn memory plots for all parameters."""
    base_spawn_memory = results["base"]["spawn"]["max_memory_used_kb"] / 1024

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot spawn memory vs width
    plot_metric_comparison(
        results,
        task,
        "width",
        "spawn_memory",
        base_spawn_memory,
        axes[0, 0],
        "Spawn Memory vs Model Width",
        "Memory (MB)",
    )

    # Plot spawn memory vs height
    plot_metric_comparison(
        results,
        task,
        "height",
        "spawn_memory",
        base_spawn_memory,
        axes[0, 1],
        "Spawn Memory vs Model Height",
        "Memory (MB)",
    )

    # Plot spawn memory vs batch_size
    plot_metric_comparison(
        results,
        task,
        "batch_size",
        "spawn_memory",
        base_spawn_memory,
        axes[1, 0],
        "Spawn Memory vs Batch Size",
        "Memory (MB)",
    )

    # Plot spawn memory vs variations
    plot_variations_metric(
        results, task, "spawn_memory", base_spawn_memory, axes[1, 1], "Spawn Memory vs Variations", "Memory (MB)"
    )

    plt.tight_layout()

    results_dir = Path("./results/bench/stats")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "spawn_memory_benchmark_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_run_memory_group(results: Dict, task: str = "mlp_intervene"):
    """Create run memory plots for all parameters."""
    base_run_memory = results["base"]["run"]["max_memory_used_kb"] / 1024

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot run memory vs width
    plot_metric_comparison(
        results, task, "width", "run_memory", base_run_memory, axes[0, 0], "Run Memory vs Model Width", "Memory (MB)"
    )

    # Plot run memory vs height
    plot_metric_comparison(
        results, task, "height", "run_memory", base_run_memory, axes[0, 1], "Run Memory vs Model Height", "Memory (MB)"
    )

    # Plot run memory vs batch_size
    plot_metric_comparison(
        results,
        task,
        "batch_size",
        "run_memory",
        base_run_memory,
        axes[1, 0],
        "Run Memory vs Batch Size",
        "Memory (MB)",
    )

    # Plot run memory vs variations
    plot_variations_metric(
        results, task, "run_memory", base_run_memory, axes[1, 1], "Run Memory vs Variations", "Memory (MB)"
    )

    plt.tight_layout()

    results_dir = Path("./results/bench/stats")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "run_memory_benchmark_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_variations_metric(
    results: Dict, task: str, metric: str, base_value: float, ax: plt.Axes, title: str, ylabel: str
):
    """Plot variations for a specific metric."""
    # Plot variations for both scripts
    for script in ["nnsight", "tdhook"]:
        if task in results and script in results[task] and "variation" in results[task][script]:
            var_data = results[task][script]["variation"]

            # Extract variation data
            variations = []
            values = []

            for var_name, seeds_data in var_data.items():
                variations.append(var_name)

            # Sort variations as integers
            variations = sorted(variations, key=int)

            # Compute means across seeds
            for var_name in variations:
                seeds_data = var_data[var_name]

                if metric == "spawn":
                    metric_values = [seed_data["spawn"]["wall_time"] for seed_data in seeds_data.values()]
                elif metric == "run":
                    metric_values = [seed_data["run"]["wall_time"] for seed_data in seeds_data.values()]
                elif metric == "spawn_memory":
                    metric_values = [
                        seed_data["spawn"]["max_memory_used_kb"] / 1024 for seed_data in seeds_data.values()
                    ]
                elif metric == "run_memory":
                    metric_values = [
                        seed_data["run"]["max_memory_used_kb"] / 1024 for seed_data in seeds_data.values()
                    ]
                else:
                    continue

                values.append(np.mean(metric_values))

            # Plot with appropriate marker and color
            ax.plot(
                variations,
                values,
                marker="o" if script == "nnsight" else "s",
                label=script.title(),
                color="blue" if script == "nnsight" else "red",
                linewidth=2,
                markersize=6,
            )

    # Add base line
    ax.axhline(y=base_value, color="green", linestyle="--", linewidth=2, label=f"Base: {base_value:.2f}", alpha=0.7)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def print_summary(results: Dict, task: str = "mlp_intervene"):
    """Print a summary of the benchmark results."""
    logger.info("=== Benchmark Results Summary ===")

    # Base results
    base_spawn_time = results["base"]["spawn"]["wall_time"]
    base_run_time = results["base"]["run"]["wall_time"]
    base_spawn_memory = results["base"]["spawn"]["max_memory_used_kb"] / 1024
    base_run_memory = results["base"]["run"]["max_memory_used_kb"] / 1024

    logger.info(f"Base spawn time: {base_spawn_time:.3f}s")
    logger.info(f"Base run time: {base_run_time:.3f}s")
    logger.info(f"Base spawn memory: {base_spawn_memory:.1f}MB")
    logger.info(f"Base run memory: {base_run_memory:.1f}MB")

    if task in results:
        for script in ["nnsight", "tdhook"]:
            if script in results[task]:
                logger.info(f"\n--- {script.upper()} Results ---")

                for parameter in ["width", "height", "batch_size"]:
                    if parameter in results[task][script]:
                        logger.info(f"\n{parameter.title()} impact:")
                        for param_value, data in results[task][script][parameter].items():
                            # Compute averages across seeds
                            spawn_times = [seed_data["spawn"]["wall_time"] for seed_data in data.values()]
                            run_times = [seed_data["run"]["wall_time"] for seed_data in data.values()]
                            spawn_memories = [
                                seed_data["spawn"]["max_memory_used_kb"] / 1024 for seed_data in data.values()
                            ]
                            run_memories = [
                                seed_data["run"]["max_memory_used_kb"] / 1024 for seed_data in data.values()
                            ]

                            avg_spawn_time = np.mean(spawn_times)
                            avg_run_time = np.mean(run_times)
                            avg_spawn_memory = np.mean(spawn_memories)
                            avg_run_memory = np.mean(run_memories)

                            logger.info(
                                f"  {parameter} {param_value}: "
                                f"spawn={avg_spawn_time:.3f}s, run={avg_run_time:.3f}s, "
                                f"spawn_mem={avg_spawn_memory:.1f}MB, run_mem={avg_run_memory:.1f}MB"
                            )


def main(args):
    """Load results and create plots."""
    logger.info("Loading benchmark results...")

    # Load results
    results_path = Path(args.input_file)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    logger.info("Creating plots...")

    # Create all plot groups
    plot_spawn_time_group(results)
    plot_run_time_group(results)
    plot_spawn_memory_group(results)
    plot_run_memory_group(results)

    # Print summary
    print_summary(results, args.task)

    logger.success("Plotting completed!")
    logger.info(f"Plots saved to '{args.output_dir}'")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("plot-stats")
    parser.add_argument(
        "--input-file", type=str, default="./results/bench/results.json", help="Input results file path"
    )
    parser.add_argument("--output-dir", type=str, default="./results/bench/stats", help="Output directory for plots")
    parser.add_argument("--task", type=str, default="mlp_intervene", help="Task to summarize in the output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
