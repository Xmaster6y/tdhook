"""
Base task - Standalone script.
"""

from loguru import logger
import torch


def main():
    logger.info("Loaded base task with:")
    logger.info(f"  Torch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")


if __name__ == "__main__":
    main()
