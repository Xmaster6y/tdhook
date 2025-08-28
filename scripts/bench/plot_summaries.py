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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

from .utils import flatten_results

# ---------------------------------------------------------------------------
# Data processing helpers
# ---------------------------------------------------------------------------


def _average_metrics_over_seeds(flattened_df: pd.DataFrame) -> pd.DataFrame:
    """Average raw metric values across seeds for each configuration.

    This collapses the per-seed data so that downstream relative metric
    computations work on a single, more stable estimate for every
    (task, lib, parameter, value) combination.
    """
    # Identify metric columns (everything except metadata)
    metadata_cols = ["task", "lib", "parameter", "value", "seed"]
    metric_cols = [col for col in flattened_df.columns if col not in metadata_cols]

    # Compute mean across seeds for each configuration
    grouped = flattened_df.groupby(["task", "lib", "parameter", "value"], as_index=False)[metric_cols].mean()

    # Keep a dummy seed column so that the rest of the pipeline (which expects
    # it to exist) continues to work without major refactors.
    grouped["seed"] = "avg"

    return grouped


def _compute_relative_metrics(flattened_df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative performance metrics vs TDHook for each metric.

    For better stability, we first average metrics across seeds for each configuration,
    then compute relative metrics using the averaged values. This reduces noise and
    provides more stable comparisons.

    For base experiments, we create new rows where each base row is divided by
    each td_hook row for the same configuration.
    """

    # Get the metric columns (all columns except the metadata columns)
    metadata_cols = ["task", "lib", "parameter", "value", "seed"]
    metric_cols = [col for col in flattened_df.columns if col not in metadata_cols]

    # Define the key columns for grouping
    key_cols = ["task", "parameter", "value"]

    # Separate base and non-base data
    base_data = flattened_df[(flattened_df["task"] == "base") & (flattened_df["parameter"] == "baseline")]
    non_base_data = flattened_df[~((flattened_df["task"] == "base") & (flattened_df["parameter"] == "baseline"))]

    # Process non-base data first
    relative_rows = []

    # Group by configuration (task, parameter, value) and compute relative metrics
    for (task, param, value), group in non_base_data.groupby(key_cols):
        # Get TDHook reference for this configuration
        tdhook_group = group[group["lib"] == "tdhook"]

        if tdhook_group.empty:
            logger.warning(f"No TDHook data found for configuration {task}/{param}/{value}, skipping")
            continue

        # For each library in this configuration (including TDHook)
        for lib in group["lib"].unique():
            lib_group = group[group["lib"] == lib]

            # For each seed of this library
            for _, lib_row in lib_group.iterrows():
                # Find corresponding TDHook row with same seed
                tdhook_row = tdhook_group[tdhook_group["seed"] == lib_row["seed"]]

                if tdhook_row.empty:
                    logger.warning(
                        f"No TDHook data found for seed {lib_row['seed']} in configuration {task}/{param}/{value}, skipping"
                    )
                    continue

                tdhook_row = tdhook_row.iloc[0]

                # Create relative metrics row
                relative_row = {
                    "task": task,
                    "lib": lib,
                    "parameter": param,
                    "value": value,
                    "seed": lib_row["seed"],
                    "is_base": False,
                }

                # Compute relative metrics for each metric
                for metric in metric_cols:
                    tdhook_val = tdhook_row[metric]
                    lib_val = lib_row[metric]

                    # Avoid division by zero or NaN
                    if pd.notna(tdhook_val) and pd.notna(lib_val) and tdhook_val > 0:
                        if lib == "tdhook":
                            # TDHook vs TDHook = 1.0
                            relative_row[f"{metric}_relative"] = 1.0
                        else:
                            # Other library vs TDHook
                            relative_row[f"{metric}_relative"] = lib_val / tdhook_val
                    else:
                        relative_row[f"{metric}_relative"] = np.nan

                relative_rows.append(relative_row)

    # Process base data - create new rows for each td_hook run across ALL configurations
    # Get ALL TDHook runs from all configurations
    all_tdhook_data = non_base_data[non_base_data["lib"] == "tdhook"]

    if not all_tdhook_data.empty:
        # For each base row
        for _, base_row in base_data.iterrows():
            # For each TDHook row across ALL configurations
            for _, tdhook_row in all_tdhook_data.iterrows():
                # Create relative metrics row for base vs TDHook
                relative_row = {
                    "task": tdhook_row["task"],  # Use TDHook task for reference
                    "lib": "base",
                    "parameter": tdhook_row["parameter"],  # Use TDHook parameter for reference
                    "value": tdhook_row["value"],  # Use TDHook value for reference
                    "seed": tdhook_row["seed"],  # Use TDHook seed for reference
                    "is_base": True,
                }

                # Compute relative metrics: base / td_hook
                for metric in metric_cols:
                    tdhook_val = tdhook_row[metric]
                    base_val = base_row[metric]

                    # Avoid division by zero or NaN
                    if pd.notna(tdhook_val) and pd.notna(base_val) and tdhook_val > 0:
                        relative_row[f"{metric}_relative"] = base_val / tdhook_val
                    else:
                        relative_row[f"{metric}_relative"] = np.nan

                relative_rows.append(relative_row)

    return pd.DataFrame(relative_rows)


def _aggregate_relative_metrics(relative_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate relative metrics across libraries and tasks.

    Now that relative metrics are computed per-sample, we can properly
    average over seeds for each (lib, task, parameter, value) combination.
    """

    # Get the relative metric columns
    relative_metric_cols = [col for col in relative_df.columns if col.endswith("_relative")]

    # Aggregate across seeds for each (lib, task, parameter, value) combination
    agg_data = []

    # Group by library, task, parameter, and value, then average over seeds
    for (lib, task, param, value), group in relative_df.groupby(["lib", "task", "parameter", "value"]):
        # Check if this is the base experiment
        is_baseline = False if group.empty else group["is_base"].iloc[0]

        # For each metric, compute mean and std across seeds
        for metric_rel in relative_metric_cols:
            metric_name = metric_rel.replace("_relative", "")
            metric_data = group[metric_rel].dropna()

            if len(metric_data) > 0:
                agg_data.append(
                    {
                        "lib": lib,
                        "task": task,
                        "parameter": param,
                        "value": value,
                        "metric": metric_name,
                        "mean": metric_data.mean(),
                        "std": metric_data.std(),
                        "n": len(metric_data),
                        "is_baseline": is_baseline,
                    }
                )

    return pd.DataFrame(agg_data)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_relative_summary(agg_df: pd.DataFrame, output_dir: Path):
    """Create summary plots showing relative performance across all libraries."""

    # Debug: Check what's in the DataFrame
    logger.info(f"Available libraries in agg_df: {sorted(agg_df['lib'].unique())}")
    logger.info(f"Available metrics in agg_df: {sorted(agg_df['metric'].unique())}")

    # Further aggregate across tasks, parameters, and values to get one value per library per metric
    # This is needed because the new structure has separate rows for each (lib, task, parameter, value, metric)
    final_agg_data = []

    for (lib, metric), group in agg_df.groupby(["lib", "metric"]):
        # Check if this is the base experiment
        is_baseline = False if group.empty else group["is_baseline"].iloc[0]

        # Aggregate across all configurations for this lib-metric combination
        all_means = group["mean"].dropna()
        all_stds = group["std"].dropna()

        if len(all_means) > 0:
            # For the final aggregation, we need to handle the fact that we're combining
            # means from different configurations. We'll use a weighted average approach.
            # Since all configurations should have the same number of seeds, we can just
            # take the mean of means and compute a combined std.

            # Combined mean (average of means from different configurations)
            combined_mean = all_means.mean()

            # Combined std (pooled variance approach)
            if len(all_stds) > 1:
                # Pooled standard deviation
                n_values = group["n"].sum()
                combined_variance = ((group["n"] - 1) * group["std"] ** 2).sum() / (n_values - len(group))
                combined_std = np.sqrt(combined_variance)
            else:
                combined_std = all_stds.iloc[0] if len(all_stds) > 0 else 0.0

            final_agg_data.append(
                {
                    "lib": lib,
                    "metric": metric,
                    "mean": combined_mean,
                    "std": combined_std,
                    "n": group["n"].sum(),
                    "is_baseline": is_baseline,
                }
            )

    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(final_agg_data)

    # Create one plot per metric
    metrics = sorted(plot_df["metric"].unique())

    for metric in metrics:
        metric_data = plot_df[plot_df["metric"] == metric]

        # Debug: Check what's available for this metric
        logger.info(f"Metric {metric}: libraries {sorted(metric_data['lib'].unique())}")

        # Use smaller figure size to maintain good text-to-figure ratio like plot_bundle.py
        plt.figure(figsize=(8, 6))

        # Sort libraries: base first (baseline), then TDHook, then alphabetically
        libs = metric_data["lib"].tolist()
        if "base" in libs:
            libs.remove("base")
            libs = ["base"] + libs
        if "tdhook" in libs:
            libs.remove("tdhook")
            libs = ["base", "tdhook"] + [lib for lib in libs if lib != "base"]
        else:
            libs = ["base"] + sorted([lib for lib in libs if lib != "base"])

        # Reorder metric_data to match the desired order, but exclude base from bars
        metric_data_ordered = []
        for lib in libs:
            if lib == "base":
                continue  # Skip base for bars, only use it for the red line
            lib_data = metric_data[metric_data["lib"] == lib]
            if not lib_data.empty:
                lib_row = lib_data.iloc[0]
                metric_data_ordered.append(lib_row)

        if not metric_data_ordered:
            logger.warning(f"No data available for metric {metric}, skipping plot")
            plt.close()
            continue

        metric_data_ordered = pd.DataFrame(metric_data_ordered).reset_index(drop=True)

        # Create bar plot with consistent style from plot_bundle.py
        # For TDHook, no error bars since relative metrics are always exactly 1.0
        yerr_values = []
        for i, lib in enumerate(metric_data_ordered["lib"]):
            if lib == "tdhook":
                yerr_values.append(np.nan)  # No error bars for TDHook
            else:
                yerr_values.append(metric_data_ordered.iloc[i]["std"])

        bars = plt.bar(
            range(len(metric_data_ordered)),
            metric_data_ordered["mean"],
            yerr=yerr_values,
            capsize=5,
            color="skyblue",
            edgecolor="navy",
            alpha=0.7,
            width=0.6,  # Use same width as bundle plots
        )

        # Color bars: TDHook in green, others in skyblue
        for i, lib in enumerate(metric_data_ordered["lib"]):
            if lib == "tdhook":
                bars[i].set_color("green")
                bars[i].set_edgecolor("darkgreen")

        # Add base experiment as a red horizontal line for baseline reference AFTER all other elements
        base_row = metric_data[metric_data["lib"] == "base"]
        if not base_row.empty:
            base_value = base_row.iloc[0]["mean"]

            # Use exact same style as plot_bundle.py
            plt.axhline(
                y=base_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Base (torch import): {base_value:.2f}",
                zorder=10,
            )

            plt.legend()
        else:
            logger.warning(f"No base data found for metric {metric}")
            logger.warning(f"Available libs in metric_data: {metric_data['lib'].tolist()}")

        # Add labels and formatting with consistent font sizes from plot_bundle.py
        plt.xticks(range(len(metric_data_ordered)), metric_data_ordered["lib"], rotation=45, ha="right")
        plt.ylabel(f"Relative {metric.replace('_', ' ').title()} (lower is better)", fontsize=12)

        # Add value labels on bars with consistent font size from plot_bundle.py
        for i, (mean, std) in enumerate(zip(metric_data_ordered["mean"], metric_data_ordered["std"])):
            lib = metric_data_ordered.iloc[i]["lib"]
            if lib == "tdhook":
                # For TDHook, use a fixed offset since std might be 0 or NaN
                plt.text(i, mean + 0.02, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
            else:
                plt.text(i, mean + std + 0.02, f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)

        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # Save plot
        fname = output_dir / f"summary_{metric}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {fname}")
        plt.close()


def _plot_combined_summary(agg_df: pd.DataFrame, output_dir: Path):
    """Create a combined plot showing run_only, spawn, and run memory metrics for all libraries."""

    # First, aggregate across tasks, parameters, and values to get one value per library per metric
    # This is needed because the new structure has separate rows for each (lib, task, parameter, value, metric)
    final_agg_data = []

    for (lib, metric), group in agg_df.groupby(["lib", "metric"]):
        # Aggregate across all configurations for this lib-metric combination
        all_means = group["mean"].dropna()
        all_stds = group["std"].dropna()

        if len(all_means) > 0:
            # Combined mean (average of means from different configurations)
            combined_mean = all_means.mean()

            # Combined std (pooled variance approach)
            if len(all_stds) > 1:
                # Pooled standard deviation
                n_values = group["n"].sum()
                combined_variance = ((group["n"] - 1) * group["std"] ** 2).sum() / (n_values - len(group))
                combined_std = np.sqrt(combined_variance)
            else:
                combined_std = all_stds.iloc[0] if len(all_stds) > 0 else 0.0

            final_agg_data.append(
                {"lib": lib, "metric": metric, "mean": combined_mean, "std": combined_std, "n": group["n"].sum()}
            )

    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(final_agg_data)

    # Filter to run_only, spawn, and run memory metrics (run_gpu_ram, run_gpu_vram, run_cpu_ram)
    # Exclude base experiment as it's just a baseline reference
    filtered_agg = plot_df[
        (
            plot_df["metric"].str.startswith("run_only_")
            | plot_df["metric"].str.startswith("spawn_")
            | plot_df["metric"].isin(["run_gpu_ram", "run_gpu_vram", "run_cpu_ram"])
        )
        & (plot_df["lib"] != "base")
    ]

    # Pivot data for easier plotting
    pivot_df = filtered_agg.pivot(index="lib", columns="metric", values="mean")

    # Hardcoded column order for logical grouping
    ordered_cols = [
        "spawn_cpu_time",
        "run_only_cpu_time",
        "spawn_gpu_time",
        "run_only_gpu_time",
        "spawn_cpu_ram",
        "run_cpu_ram",
        "spawn_gpu_ram",
        "run_gpu_ram",
        "spawn_gpu_vram",
        "run_gpu_vram",
    ]

    # Only use columns that actually exist in the data, maintaining logical order
    available_cols = [col for col in ordered_cols if col in pivot_df.columns]
    pivot_df = pivot_df[available_cols]

    # Sort rows: TDHook first, then alphabetically (base excluded)
    libs = pivot_df.index.tolist()
    if "tdhook" in libs:
        libs.remove("tdhook")
        libs = ["tdhook"] + sorted(libs)
    else:
        libs = sorted(libs)
    pivot_df = pivot_df.reindex(libs)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1)

    log_df = pivot_df.map(lambda x: np.log(x) if x > 0 else np.nan)

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


def main():
    """Main function to generate summary plots."""
    args = parse_args()

    # Load results
    logger.info(f"Loading results from {args.input_file}")
    with open(args.input_file, "r") as f:
        raw_results = json.load(f)

    # Flatten the nested results structure
    df = flatten_results(raw_results)
    logger.info(f"Loaded {len(df)} runs")

    # First, average raw metrics across seeds for stability
    df = _average_metrics_over_seeds(df)
    logger.info("Averaged raw metrics across seeds")

    # Filter out captum_add if present (it's a duplicate)
    if "captum_add" in df["lib"].values:
        df = df[df["lib"] != "captum_add"]
        logger.info(f"Filtered out captum_add, remaining {len(df)} runs")

    # Compute relative metrics vs TDHook
    df_relative = _compute_relative_metrics(df)
    logger.info("Computed per-sample relative metrics vs TDHook")

    # Aggregate relative metrics
    agg_df = _aggregate_relative_metrics(df_relative)
    logger.info(f"Aggregated {len(agg_df)} metric-library combinations")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    _plot_relative_summary(agg_df, output_dir)
    _plot_combined_summary(agg_df, output_dir)

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")

    # First aggregate across tasks, parameters, and values to get one value per library per metric
    final_agg_data = []

    for (lib, metric), group in agg_df.groupby(["lib", "metric"]):
        # Aggregate across all configurations for this lib-metric combination
        all_means = group["mean"].dropna()
        all_stds = group["std"].dropna()

        if len(all_means) > 0:
            # Combined mean (average of means from different configurations)
            combined_mean = all_means.mean()

            # Combined std (pooled variance approach)
            if len(all_stds) > 1:
                # Pooled standard deviation
                n_values = group["n"].sum()
                combined_variance = ((group["n"] - 1) * group["std"] ** 2).sum() / (n_values - len(group))
                combined_std = np.sqrt(combined_variance)
            else:
                combined_std = all_stds.iloc[0] if len(all_stds) > 0 else 0.0

            final_agg_data.append(
                {"lib": lib, "metric": metric, "mean": combined_mean, "std": combined_std, "n": group["n"].sum()}
            )

    # Convert to DataFrame for summary printing
    summary_df = pd.DataFrame(final_agg_data)

    for metric in sorted(summary_df["metric"].unique()):
        metric_data = summary_df[summary_df["metric"] == metric]
        logger.info(f"\n{metric.replace('_', ' ').title()}:")

        for _, row in metric_data.iterrows():
            if row["lib"] == "tdhook":
                logger.info(f"  {row['lib']}: {row['mean']:.3f} (n={row['n']})")
            else:
                logger.info(f"  {row['lib']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['n']})")

    logger.success("All summary plots generated!")


if __name__ == "__main__":
    main()
