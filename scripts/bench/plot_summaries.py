"""
Plot summary statistics across all benchmark tasks.

This script computes relative performance metrics with TDHook as the baseline (1.0).
For each metric, it computes per-sample relative performance (for each
task/parameter/value/seed) as lib_value / TDHook_value, then aggregates
mean±std across all samples per library.

Usage:
    uv run --group scripts -m scripts.bench.plot_summaries \
        --input-file ./results/bench/results.json \
        --output-dir ./results/bench/summaries

Output: One plot per metric showing relative performance across all libraries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

# ---------------------------------------------------------------------------
# Data processing helpers
# ---------------------------------------------------------------------------


def _flatten_results(raw: Dict) -> pd.DataFrame:
    """Return a tidy DataFrame from the benchmark json structure."""

    if "base" not in raw:
        raise ValueError("results json must contain a 'base' key with baseline numbers")

    rows: List[Dict] = []

    for task, libs in raw.items():
        if task == "base":
            continue

        for lib, params in libs.items():
            for param_name, param_vals in params.items():
                for param_val, seeds in param_vals.items():
                    for seed_str, runs in seeds.items():
                        row = {
                            "task": task,
                            "lib": lib,
                            "parameter": param_name,
                            "value": str(param_val),
                            "seed": int(seed_str),
                            # times (seconds)
                            "spawn_time": runs["spawn"]["wall_time"],
                            "run_cpu_time": runs.get("run_cpu", {}).get("wall_time", np.nan),
                            "run_gpu_time": runs.get("run_gpu", {}).get("wall_time", np.nan),
                            # host RAM (MB)
                            "spawn_ram": runs["spawn"]["max_ram_used_kb"] / 1024,
                            "run_cpu_ram": runs.get("run_cpu", {}).get("max_ram_used_kb", np.nan) / 1024,
                            "run_gpu_ram": runs.get("run_gpu", {}).get("max_ram_used_kb", np.nan) / 1024,
                            # vram (MB)
                            "spawn_gpu_vram": runs["spawn"].get("max_gpu_memory_kb", 0.0) / 1024,
                            "run_gpu_vram": runs.get("run_gpu", {}).get("max_gpu_memory_kb", np.nan) / 1024,
                        }
                        rows.append(row)

    return pd.DataFrame(rows)


def _compute_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample relative metrics vs TDHook for the same configuration.

    For each row identified by (task, parameter, value), we first average over seeds,
    then divide the metric value by the corresponding TDHook value for that same key.
    This avoids dividing means and yields a fair, paired comparison.
    """

    metrics = [
        "spawn_time",
        "run_cpu_time",
        "run_gpu_time",
        "spawn_ram",
        "run_cpu_ram",
        "run_gpu_ram",
        "spawn_gpu_vram",
        "run_gpu_vram",
    ]

    key_cols = ["task", "parameter", "value"]

    # First, average over seeds for each (task, parameter, value, lib) combination
    df_avg = df.groupby(key_cols + ["lib"])[metrics].mean().reset_index()

    # Add run-only metrics: run_metric - spawn_metric for time metrics only
    # Map run time metrics to their spawn counterparts
    run_spawn_mapping = {
        "run_cpu_time": "spawn_time",
        "run_gpu_time": "spawn_time",
    }

    run_only_metrics = []
    for run_metric, spawn_metric in run_spawn_mapping.items():
        if run_metric in metrics and spawn_metric in metrics:
            run_only_name = f"run_only_{run_metric.replace('run_', '')}"
            df_avg[run_only_name] = df_avg[run_metric] - df_avg[spawn_metric]
            run_only_metrics.append(run_only_name)

    # Update metrics list to include run-only metrics
    all_metrics = metrics + run_only_metrics

    # TDHook reference for each configuration
    tdhook_ref = df_avg[df_avg["lib"] == "tdhook"][key_cols + all_metrics].rename(
        columns={m: f"tdh_{m}" for m in all_metrics}
    )

    # Join reference back onto all rows; drop rows without TDHook match
    merged = df_avg.merge(tdhook_ref, on=key_cols, how="inner")

    # Compute relative metrics row-wise; guard against zero/NaN
    for metric in all_metrics:
        denom = merged[f"tdh_{metric}"]
        num = merged[metric]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where((denom > 0) & np.isfinite(denom), num / denom, np.nan)
        merged[f"{metric}_relative"] = rel

    # Keep only relevant columns going forward
    keep_cols = ["lib", "task", "parameter", "value"] + [f"{m}_relative" for m in all_metrics]
    return merged[keep_cols]


def _aggregate_relative_metrics(df_relative: pd.DataFrame) -> pd.DataFrame:
    """Aggregate relative metrics by library, computing mean and std."""

    relative_metrics = [col for col in df_relative.columns if col.endswith("_relative")]

    # Group by library and compute statistics
    agg_stats = []
    for lib in df_relative["lib"].unique():
        lib_data = df_relative[df_relative["lib"] == lib]

        for metric_rel in relative_metrics:
            metric_name = metric_rel.replace("_relative", "")
            values = lib_data[metric_rel].dropna()

            if len(values) > 0:
                agg_stats.append(
                    {
                        "lib": lib,
                        "metric": metric_name,
                        "mean": values.mean(),
                        "std": values.std(),
                        "count": len(values),
                    }
                )

    return pd.DataFrame(agg_stats)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_relative_summary(agg_df: pd.DataFrame, output_dir: Path):
    """Create summary plots showing relative performance across all libraries."""

    # Create one plot per metric
    metrics = sorted(agg_df["metric"].unique())

    for metric in metrics:
        metric_data = agg_df[agg_df["metric"] == metric]

        # Use smaller figure size to maintain good text-to-figure ratio like plot_bundle.py
        plt.figure(figsize=(8, 6))

        # Sort libraries: TDHook first, then alphabetically
        libs = metric_data["lib"].tolist()
        if "tdhook" in libs:
            libs.remove("tdhook")
            libs = ["tdhook"] + sorted(libs)
        else:
            libs = sorted(libs)

        # Reorder metric_data to match the desired order
        metric_data_ordered = []
        for lib in libs:
            lib_row = metric_data[metric_data["lib"] == lib].iloc[0]
            metric_data_ordered.append(lib_row)
        metric_data_ordered = pd.DataFrame(metric_data_ordered).reset_index(drop=True)

        # Create bar plot with consistent style from plot_bundle.py
        bars = plt.bar(
            range(len(metric_data_ordered)),
            metric_data_ordered["mean"],
            yerr=metric_data_ordered["std"],
            capsize=5,
            color="skyblue",
            edgecolor="navy",
            alpha=0.7,
        )

        # Color bars: TDHook in green, others in skyblue (already set above)
        for i, lib in enumerate(metric_data_ordered["lib"]):
            if lib == "tdhook":
                bars[i].set_color("green")
                bars[i].set_edgecolor("darkgreen")

        # Add labels and formatting with consistent font sizes from plot_bundle.py
        plt.xticks(range(len(metric_data_ordered)), metric_data_ordered["lib"], rotation=45, ha="right")
        plt.ylabel(f"Relative {metric.replace('_', ' ').title()} (lower is better)", fontsize=12)

        # Reference line at 1.0 (TDHook)
        plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="TDHook = 1.0")

        # Add value labels on bars with consistent font size from plot_bundle.py
        for i, (mean, std) in enumerate(zip(metric_data_ordered["mean"], metric_data_ordered["std"])):
            lib = metric_data_ordered.iloc[i]["lib"]
            if lib == "tdhook":
                plt.text(i, mean + std + 0.02, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
            else:
                plt.text(i, mean + std + 0.02, f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)

        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # Save plot
        fname = output_dir / f"summary_{metric}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {fname}")
        plt.close()


def _plot_combined_summary(agg_df: pd.DataFrame, output_dir: Path):
    """Create a combined plot showing run_only, spawn, and run memory metrics for all libraries."""

    # Filter to run_only, spawn, and run memory metrics (run_gpu_ram, run_gpu_vram, run_cpu_ram)
    filtered_agg = agg_df[
        agg_df["metric"].str.startswith("run_only_")
        | agg_df["metric"].str.startswith("spawn_")
        | agg_df["metric"].isin(["run_gpu_ram", "run_gpu_vram", "run_cpu_ram"])
    ]

    # Pivot data for easier plotting
    pivot_df = filtered_agg.pivot(index="lib", columns="metric", values="mean")

    # Sort columns: spawn metrics first, then run memory metrics, then run_only metrics
    spawn_cols = [col for col in pivot_df.columns if col.startswith("spawn_")]
    run_memory_cols = [col for col in pivot_df.columns if col in ["run_gpu_ram", "run_gpu_vram", "run_cpu_ram"]]
    run_only_cols = [col for col in pivot_df.columns if col.startswith("run_only_")]
    ordered_cols = sorted(spawn_cols) + sorted(run_memory_cols) + sorted(run_only_cols)
    ordered_cols = [
        "spawn_time",
        "run_only_cpu_time",
        "run_only_gpu_time",
        "spawn_ram",
        "run_cpu_ram",
        "run_gpu_ram",
        "spawn_gpu_vram",
        "run_gpu_vram",
    ]
    pivot_df = pivot_df[ordered_cols]

    # Sort rows: TDHook first, then alphabetically
    libs = pivot_df.index.tolist()
    if "tdhook" in libs:
        libs.remove("tdhook")
        libs = ["tdhook"] + sorted(libs)
    else:
        libs = sorted(libs)
    pivot_df = pivot_df.reindex(libs)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1)

    log_df = pivot_df.applymap(lambda x: np.log(x) if x > 0 else np.nan)

    sns.heatmap(
        log_df,
        annot=pivot_df,
        fmt=".2f",
        cmap="RdYlGn_r",
        center=0.0,
        cbar_kws={"label": "Relative Performance", "shrink": 0.61},
        square=True,
    )

    # Get the colorbar and apply exp formatter to show original values
    ax = plt.gca()
    cbar = ax.collections[0].colorbar

    # Create a formatter that shows the original values (exp of log values)
    from matplotlib.ticker import FuncFormatter

    def exp_formatter(x, pos):
        return f"{np.exp(x):.1f}"

    # Just apply the formatter without changing tick locations
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(exp_formatter))

    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Libraries", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fname = output_dir / "summary_combined_heatmap.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {fname}")
    plt.close()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("plot-summaries")
    parser.add_argument(
        "--input-file", default="./results/bench/results.json", type=str, help="Path to benchmark JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="./results/bench/summaries",
        type=str,
        help="Directory where summary plots will be saved",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Generate summary plots showing relative performance across all libraries."""

    if args is None:
        args = parse_args()

    input_path = Path(args.input_file).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from {input_path}")

    with input_path.open() as f:
        raw = json.load(f)

    # Process data
    df = _flatten_results(raw)
    logger.info(f"Loaded {len(df)} runs")

    # Filter out captum_add library
    df = df[df["lib"] != "captum_add"]
    logger.info(f"Filtered out captum_add, remaining {len(df)} runs")

    df_relative = _compute_relative_metrics(df)
    logger.info("Computed per-sample relative metrics vs TDHook")

    agg_df = _aggregate_relative_metrics(df_relative)
    logger.info(f"Aggregated {len(agg_df)} metric-library combinations")

    # Generate plots
    _plot_relative_summary(agg_df, output_dir)
    _plot_combined_summary(agg_df, output_dir)

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    for metric in sorted(agg_df["metric"].unique()):
        metric_data = agg_df[agg_df["metric"] == metric]
        logger.info(f"\n{metric.replace('_', ' ').title()}:")
        for _, row in metric_data.iterrows():
            if row["lib"] == "tdhook":
                logger.info(f"  {row['lib']}: {row['mean']:.3f} (n={row['count']})")
            else:
                logger.info(f"  {row['lib']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})")

    logger.success("All summary plots generated!")


if __name__ == "__main__":
    main()
