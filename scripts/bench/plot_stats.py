"""
Plot benchmark results in `results/bench/results.json`.

Usage:
    uv run --group scripts -m scripts.bench.plot_stats

The script assumes the JSON hierarchy:
    task -> library -> parameter_name -> parameter_value -> seed -> {spawn_cpu, spawn_gpu, run_cpu, run_gpu}

It flattens this structure, aggregates mean±std over seeds, and creates grouped bar
plots for:
    • spawn_cpu_time  (wall-time, seconds)
    • spawn_gpu_time  (wall-time, seconds)
    • run_cpu_time
    • run_gpu_time
    • spawn_cpu_ram   (host RAM, MB)
    • spawn_gpu_ram   (host RAM, MB)
    • run_cpu_ram
    • run_gpu_ram
    • spawn_cpu_vram (GPU memory, MB)
    • spawn_gpu_vram (GPU memory, MB)
    • run_gpu_vram

Dependencies: pandas, seaborn, matplotlib, loguru
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

from .utils import flatten_results

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Data wrangling helpers
# ---------------------------------------------------------------------------


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean & std per (task, lib, parameter, value) combination."""

    metrics = [
        "spawn_cpu_time",
        "spawn_gpu_time",
        "run_cpu_time",
        "run_gpu_time",
        "spawn_cpu_ram",
        "spawn_gpu_ram",
        "run_cpu_ram",
        "run_gpu_ram",
        "spawn_gpu_vram",
        "run_gpu_vram",
    ]

    grouped = df.groupby(["task", "lib", "parameter", "value"])[metrics]
    means = grouped.mean().reset_index().rename(columns={m: f"{m}_mean" for m in metrics})
    stds = grouped.std().reset_index().rename(columns={m: f"{m}_std" for m in metrics})
    return pd.merge(means, stds, on=["task", "lib", "parameter", "value"], how="left")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_metric(
    agg: pd.DataFrame,
    task: str,
    metric: str,
    output_dir: Path,
    palette: Dict[str, str] | None = None,
):
    """Create a grouped bar plot (lib × varying value) for the given metric."""

    subset = agg[agg["task"] == task]
    if subset.empty:
        logger.warning(f"No rows for task '{task}', skip {metric}")
        return

    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"

    for parameter, df_param in subset.groupby("parameter"):
        plt.figure(figsize=(10, 6))
        # Sort the x-axis values so that numeric entries are ordered numerically and
        # non-numeric entries alphabetically. Numeric values (convertible to ``int``)
        # are displayed first, followed by the remaining values in lexicographic
        # order. This ensures an intuitive left-to-right progression for plots such
        # as batch-sizes (8, 16, 32, …) while still handling categorical settings
        # like "relu" / "tanh" gracefully.

        def _sort_key(v):
            """Return a tuple usable as a sort key.

            The first element indicates whether the value is *non-numeric* (1) or
            *numeric* (0). The second element is the numeric value itself or the
            string representation, so that ``sorted`` arranges numeric values in
            ascending order and strings alphabetically.
            """

            try:
                return (0, int(v))
            except (ValueError, TypeError):
                # Fall back to string comparison for non-numeric values.
                return (1, str(v))

        order = sorted(df_param["value"].unique(), key=_sort_key)

        # Reset index to ensure 0..n sequential for bar/error mapping
        df_plot = df_param.reset_index(drop=True)

        # Create the bar plot
        ax = sns.barplot(
            data=df_plot,
            x="value",
            y=metric_mean,
            hue="lib",
            order=order,
            palette=palette,
            capsize=0.1,
            err_kws={"color": "gray", "linewidth": 1},
            errorbar=None,  # we add manual error bars below
        )

        # Add manual error bars using matplotlib
        patches = ax.patches
        for i, patch in enumerate(patches):
            if i >= len(df_plot):
                continue
            # Each patch corresponds to one bar; get its center x pos
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            # Determine which row this bar corresponds to
            err = df_plot.iloc[i][metric_std]
            if not np.isnan(err):
                ax.errorbar(x, height, yerr=err, ecolor="black", capsize=3, fmt="none", lw=1)

        plt.title(f"{task} – {metric.replace('_', ' ').title()} vs {parameter}")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xlabel(parameter.replace("_", " ").title())
        plt.tight_layout()

        fname = output_dir / f"{task}_{parameter}_{metric}.pdf"
        plt.savefig(fname, dpi=300)
        try:
            rel = fname.relative_to(Path.cwd())
        except ValueError:
            rel = fname
        logger.info(f"Saved {rel}")
        plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("plot-stats")
    parser.add_argument(
        "--input-file", default="./results/bench/results.json", type=str, help="Path to benchmark JSON file"
    )
    parser.add_argument(
        "--output-dir", default="./results/bench/plots", type=str, help="Directory where plots will be saved"
    )
    parser.add_argument(
        "--task", default=None, help="Restrict plotting to a single task label from the JSON (default: all tasks)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Generate plots using the provided command-line arguments."""

    if args is None:
        args = parse_args()

    input_path = Path(args.input_file).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rel = input_path.relative_to(Path.cwd())
    except ValueError:
        rel = input_path
    logger.info(f"Loading results from {rel}")

    with input_path.open() as f:
        raw: Dict = json.load(f)

    df = flatten_results(raw)
    agg = _aggregate(df)

    logger.info(f"Loaded {len(df)} runs → {len(agg)} aggregated rows")

    metrics = [
        "spawn_cpu_time",
        "spawn_gpu_time",
        "run_cpu_time",
        "run_gpu_time",
        "spawn_cpu_ram",
        "spawn_gpu_ram",
        "run_cpu_ram",
        "run_gpu_ram",
        "spawn_gpu_vram",
        "run_gpu_vram",
    ]

    sns.set_palette("tab10")
    palette = dict(zip(sorted(df["lib"].unique()), sns.color_palette()))

    tasks = [args.task] if args.task else sorted(df["task"].unique())

    for metric in metrics:
        for task in tasks:
            _plot_metric(agg, task, metric, output_dir, palette)

    logger.success("All plots generated!")


if __name__ == "__main__":
    main()
