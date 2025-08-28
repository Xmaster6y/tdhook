"""
Script to measure memory sizes and inode counts for different packages and plot results.

Run with:

```
uv run --group scripts -m scripts.bench.plot_bundle
```

"""

import re
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from .utils import run_command


def parse_du_output(output):
    """Parse du -sh output and return size in MB."""
    if not output:
        return 0

    # Extract the size from output like "585M    ."
    match = re.search(r"(\d+(?:\.\d+)?)([KMGT]?)", output)
    if not match:
        logger.warning(f"Could not parse du output: {output}")
        return 0

    size = float(match.group(1))
    unit = match.group(2)

    # Convert to MB
    multipliers = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    return size * multipliers.get(unit, 1)


def parse_inode_output(output):
    """Parse find output and return number of files."""
    if not output:
        return 0

    try:
        return int(output.strip())
    except ValueError:
        logger.warning(f"Could not parse inode count: {output}")
        return 0


def measure_memory_size():
    """Measure the current memory size."""
    output = run_command("du -sh", cwd="./scripts/bundle-test/.venv")
    return parse_du_output(output)


def measure_inode_count():
    """Measure the current number of files (inodes) in .venv."""
    output = run_command("find . -type f | wc -l", cwd="./scripts/bundle-test/.venv")
    return parse_inode_output(output)


def measure_metrics():
    """Measure both memory size and inode count."""
    memory_size = measure_memory_size()
    inodes = measure_inode_count()
    return {"memory_size": memory_size, "inodes": inodes}


def install_package(package):
    """Install a package and return the new metrics."""
    logger.info(f"Installing {package}...")

    # Install the package
    if package == "tdhook":
        install_cmd = "uv pip install ../.. -p .venv/bin/python"
    else:
        install_cmd = f"uv sync --group {package} --locked -p .venv/bin/python"
    result = run_command(install_cmd, cwd="./scripts/bundle-test")
    if result is None:
        logger.error(f"Failed to install {package}")
        return None

    # Measure new metrics
    new_metrics = measure_metrics()
    logger.info(f"Memory size after installing {package}: {new_metrics['memory_size']:.1f}M")
    logger.info(f"Inode count after installing {package}: {new_metrics['inodes']}")

    return new_metrics


def reset_environment():
    """Reset the environment to clean state."""
    logger.info("Resetting environment...")
    result = run_command("uv sync", cwd="./scripts/bundle-test")
    if result is None:
        logger.error("Failed to reset environment")
        return False
    return True


def main(args):
    """Main function to measure memory sizes and inode counts and create plots."""
    logger.info("Starting memory size and inode count measurement...")

    # Packages to test
    packages = args.packages

    # Store results
    results = {}
    base_metrics = None

    # First, measure base metrics (torch without tensordict)
    logger.info("\n--- Measuring base metrics (torch only) ---")
    if not reset_environment():
        logger.error("Failed to reset environment for base measurement")
        return

    base_metrics = measure_metrics()
    logger.info(f"Base memory size (torch only): {base_metrics['memory_size']:.1f}M")
    logger.info(f"Base inode count (torch only): {base_metrics['inodes']}")

    # Test each package
    for package in packages:
        logger.info(f"\n--- Testing {package} ---")

        # Reset environment
        if not reset_environment():
            logger.error(f"Failed to reset environment for {package}")
            continue

        # Install package and measure
        new_metrics = install_package(package)
        if new_metrics is not None:
            results[package] = new_metrics
            logger.info(f"{package} memory size: {new_metrics['memory_size']:.1f}M")
            logger.info(f"{package} inode count: {new_metrics['inodes']}")
        else:
            logger.error(f"Failed to measure {package}")

    # Create plots
    if results and base_metrics is not None:
        create_plots(results, base_metrics, args.output_dir)
    else:
        logger.error("No results to plot")


def create_plots(results, base_metrics, output_dir):
    """Create bar charts for both memory size and inode count."""
    # Sort packages by memory size in increasing order
    sorted_packages = sorted(results.keys(), key=lambda pkg: results[pkg]["memory_size"])

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create memory size plot
    create_memory_size_plot(results, base_metrics, output_path)

    # Create inode count plot
    create_inode_count_plot(results, base_metrics, output_path)

    # Print summary
    print_summary(results, base_metrics, sorted_packages)


def create_memory_size_plot(results, base_metrics, output_path):
    """Create a bar chart for memory sizes."""
    # Sort packages by memory size in increasing order
    sorted_packages = sorted(results.keys(), key=lambda pkg: results[pkg]["memory_size"])
    memory_sizes = [results[pkg]["memory_size"] for pkg in sorted_packages]

    # Create the plot
    plt.figure(figsize=(6, 6))
    bars = plt.bar(sorted_packages, memory_sizes, color="skyblue", edgecolor="navy", alpha=0.7)

    # Add a red line for base memory size
    base_memory_size = base_metrics["memory_size"]
    plt.axhline(
        y=base_memory_size,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Base (torch only): {int(base_memory_size)}M",
    )
    plt.legend()

    # Add value labels on bars with percentage increase
    for bar, memory_size in zip(bars, memory_sizes):
        height = bar.get_height()
        increase_pct = ((memory_size - base_memory_size) / base_memory_size) * 100
        label = f"+{int(increase_pct)}%"

        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, label, ha="center", va="bottom", fontsize=9)

    plt.xlabel("Package", fontsize=12)
    plt.ylabel("Memory Size (MB)", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")

    # Reduce space between bars by adjusting x-axis limits
    plt.xlim(-0.5, len(sorted_packages) - 0.5)

    # Adjust layout with tighter spacing
    plt.tight_layout()

    # Save the plot
    plot_path = output_path / "memory_sizes.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Memory size plot saved to {plot_path}")
    plt.close()


def create_inode_count_plot(results, base_metrics, output_path):
    """Create a bar chart for inode counts."""
    # Sort packages by inode count in increasing order
    sorted_packages = sorted(results.keys(), key=lambda pkg: results[pkg]["inodes"])
    inodes = [results[pkg]["inodes"] for pkg in sorted_packages]

    # Create the plot
    plt.figure(figsize=(6, 6))
    bars = plt.bar(sorted_packages, inodes, color="skyblue", edgecolor="navy", alpha=0.7)

    # Add a red line for base inode count
    base_inodes = base_metrics["inodes"]
    plt.axhline(y=base_inodes, color="red", linestyle="--", linewidth=2, label=f"Base (torch only): {base_inodes}")
    plt.legend()

    # Add value labels on bars with percentage increase
    for bar, inode_count in zip(bars, inodes):
        height = bar.get_height()
        increase_pct = ((inode_count - base_inodes) / base_inodes) * 100
        label = f"+{int(increase_pct)}%"

        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, label, ha="center", va="bottom", fontsize=9)

    plt.xlabel("Package", fontsize=12)
    plt.ylabel("Inode Count", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")

    # Reduce space between bars by adjusting x-axis limits
    plt.xlim(-0.5, len(sorted_packages) - 0.5)

    # Adjust layout with tighter spacing
    plt.tight_layout()

    # Save the plot
    plot_path = output_path / "inode_counts.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Inode count plot saved to {plot_path}")
    plt.close()


def print_summary(results, base_metrics, sorted_packages):
    """Print summary of results in sorted order."""
    logger.info("\n--- Summary (sorted by memory size) ---")
    logger.info(f"Base (torch only): {int(base_metrics['memory_size'])}M, {base_metrics['inodes']} files")

    for package in sorted_packages:
        memory_size = results[package]["memory_size"]
        inodes = results[package]["inodes"]

        memory_increase_pct = ((memory_size - base_metrics["memory_size"]) / base_metrics["memory_size"]) * 100
        inode_increase_pct = ((inodes - base_metrics["inodes"]) / base_metrics["inodes"]) * 100

        logger.info(
            f"{package}: {int(memory_size)}M (+{int(memory_increase_pct)}%), {inodes} files (+{int(inode_increase_pct)}%)"
        )


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("plot-bundle")
    parser.add_argument(
        "--packages",
        nargs="+",
        default=["tdhook", "nnsight", "transformer_lens", "captum", "zennit", "pyvene", "inseq"],
        help="Packages to test",
    )
    parser.add_argument("--output-dir", type=str, default="results/bench/bundle", help="Output directory for plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
