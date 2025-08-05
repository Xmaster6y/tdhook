"""
Test script for the measurement system.

Run with:

```
uv run --group scripts -m scripts.bench.test_measurement
```
"""

from loguru import logger

from .utils import Measurer


def main():
    """Test the measurement system."""
    logger.info("Testing measurement system...")

    measurer = Measurer()

    logger.info("Measuring...")
    stats = measurer.measure_script("scripts/bench/tasks/_test.py", {})

    logger.info("Spawn stats:")
    for key, value in stats["spawn"].items():
        logger.info(f"  {key}: {value}")
    logger.info("Run stats:")
    for key, value in stats["run"].items():
        logger.info(f"  {key}: {value}")

    logger.success("Measurement test completed!")


if __name__ == "__main__":
    main()
