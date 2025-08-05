"""
Base task - Standalone script.
"""

from loguru import logger
import torch


def main():
    logger.info("Loaded base task with:")
    logger.info(f"  Torch version: {torch.__version__}")


if __name__ == "__main__":
    main()
