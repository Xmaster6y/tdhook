"""
Script to measure bundle sizes for different packages and plot results.

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


def measure_bundle_size():
    """Measure the current bundle size."""
    output = run_command("du -sh", cwd="./scripts/bundle-test/.venv")
    return parse_du_output(output)


def install_package(package):
    """Install a package and return the new bundle size."""
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

    # Measure new bundle size
    new_size = measure_bundle_size()
    logger.info(f"Bundle size after installing {package}: {new_size:.1f}M")

    return new_size


def reset_environment():
    """Reset the environment to clean state."""
    logger.info("Resetting environment...")
    result = run_command("uv sync", cwd="./scripts/bundle-test")
    if result is None:
        logger.error("Failed to reset environment")
        return False
    return True


def main(args):
    """Main function to measure bundle sizes and create plot."""
    logger.info("Starting bundle size measurement...")

    # Packages to test
    packages = args.packages

    # Store results
    results = {}
    base_size = None

    # First, measure base size (torch without tensordict)
    logger.info("\n--- Measuring base size (torch only) ---")
    if not reset_environment():
        logger.error("Failed to reset environment for base measurement")
        return

    base_size = measure_bundle_size()
    logger.info(f"Base bundle size (torch only): {base_size:.1f}M")

    # Test each package
    for package in packages:
        logger.info(f"\n--- Testing {package} ---")

        # Reset environment
        if not reset_environment():
            logger.error(f"Failed to reset environment for {package}")
            continue

        # Install package and measure
        new_size = install_package(package)
        if new_size is not None:
            results[package] = {"size": new_size}
            logger.info(f"{package} bundle size: {new_size:.1f}M")
        else:
            logger.error(f"Failed to measure {package}")

    # Create plot
    if results and base_size is not None:
        create_plot(results, base_size, args.output_dir)
    else:
        logger.error("No results to plot")


def create_plot(results, base_size, output_dir):
    """Create a bar chart of the results."""
    # Sort packages by size in increasing order
    sorted_packages = sorted(results.keys(), key=lambda pkg: results[pkg]["size"])
    sizes = [results[pkg]["size"] for pkg in sorted_packages]

    # Create the plot
    plt.figure(figsize=(6, 6))
    bars = plt.bar(sorted_packages, sizes, color="skyblue", edgecolor="navy", alpha=0.7)

    # Add a red line for base size
    if base_size is not None:
        plt.axhline(
            y=base_size, color="red", linestyle="--", linewidth=2, label=f"Base (torch only): {int(base_size)}M"
        )
        plt.legend()

    # Add value labels on bars with percentage increase
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        if base_size is not None:
            increase_pct = ((size - base_size) / base_size) * 100
            label = f"{int(size)}M (+{int(increase_pct)}%)"
        else:
            label = f"{int(size)}M"

        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, label, ha="center", va="bottom", fontsize=9)

    plt.title("Bundle Size by Package", fontsize=14, fontweight="bold")
    plt.xlabel("Package", fontsize=12)
    plt.ylabel("Bundle Size (MB)", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")

    # Reduce space between bars by adjusting x-axis limits
    plt.xlim(-0.5, len(sorted_packages) - 0.5)

    # Adjust layout with tighter spacing
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "bundle_sizes.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")

    # Show the plot
    plt.show()

    # Print summary in sorted order
    logger.info("\n--- Summary (sorted by size) ---")
    logger.info(f"Base (torch only): {int(base_size)}M")
    for package in sorted_packages:
        size = results[package]["size"]
        if base_size is not None:
            increase_pct = ((size - base_size) / base_size) * 100
            logger.info(f"{package}: {int(size)}M (+{int(increase_pct)}%)")
        else:
            logger.info(f"{package}: {int(size)}M")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("plot-bundle")
    parser.add_argument(
        "--packages",
        nargs="+",
        default=["tdhook", "nnsight", "transformer_lens", "captum", "zennit"],
        help="Packages to test",
    )
    parser.add_argument("--output-dir", type=str, default="results/bench/bundle", help="Output directory for plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
