"""
Test task - Standalone script.
"""

import argparse
import time
from loguru import logger
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024 * 1024)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    logger.info("Running test task with:")
    logger.info(f"  Size: {args.size}")
    logger.info(f"  Sleep: {args.sleep}")

    # Run the benchmark
    if args.run:
        time.sleep(args.sleep / 2)
        torch.randn(args.size)
        time.sleep(args.sleep / 2)


if __name__ == "__main__":
    main()
