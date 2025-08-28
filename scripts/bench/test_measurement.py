"""
Test script for the measurement system.

Run with:

```
uv run --group scripts -m scripts.bench.test_measurement
```
"""

from loguru import logger

from .utils import Measurer


def log_stats(stats: dict):
    """Log the stats."""
    logger.info("Spawn CPU stats:")
    for key, value in stats["spawn_cpu"].items():
        logger.info(f"  {key}: {value}")
    logger.info("Spawn GPU stats:")
    for key, value in stats["spawn_gpu"].items():
        logger.info(f"  {key}: {value}")
    logger.info("Run CPU stats:")
    for key, value in stats["run_cpu"].items():
        logger.info(f"  {key}: {value}")
    logger.info("Run GPU stats:")
    for key, value in stats["run_gpu"].items():
        logger.info(f"  {key}: {value}")


def main(args):
    """Test the measurement system."""
    logger.info("Testing measurement system...")

    measurer = Measurer()

    for i in range(args.iterations):
        logger.info(f"Measuring `_base` ({i + 1})...")
        stats = measurer.measure_script("scripts/bench/tasks/_base.py", {})
        log_stats(stats)

    for i in range(args.iterations):
        logger.info(f"Measuring `_test` ({i + 1})...")
        stats = measurer.measure_script("scripts/bench/tasks/_test.py", {})
        log_stats(stats)

    for i in range(args.iterations):
        logger.info(f"Measuring `mlp_intervene._nnsight` ({i + 1})...")
        stats = measurer.measure_script("scripts/bench/tasks/mlp_intervene/_nnsight.py", {})
        log_stats(stats)

    for i in range(args.iterations):
        logger.info(f"Measuring `mlp_intervene._tdhook` ({i + 1})...")
        stats = measurer.measure_script("scripts/bench/tasks/mlp_intervene/_tdhook.py", {})
        log_stats(stats)

    logger.success("Measurement test completed!")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("test-measurement")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to run for each test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
