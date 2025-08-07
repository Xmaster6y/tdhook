"""
Test task - Standalone script.
"""

import argparse
from loguru import logger
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

cuda_available = torch.cuda.is_available()


class MLP(nn.Module):
    def __init__(self, height: int = 10, width: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(height)])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


def prepare(
    batch_size: int,
    use_cuda: bool,
) -> Tuple[nn.Module, torch.Tensor]:
    """Prepare the model and input data."""
    model = MLP().to("cuda" if use_cuda else "cpu")
    input_data = torch.randn(batch_size, 10).to("cuda" if use_cuda else "cpu")
    return model, input_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024 * 1024)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = cuda_available and args.cuda

    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    logger.info("Running test task with:")
    logger.info(f"  Spawn max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.0f} KB")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Use CUDA: {use_cuda}")

    model, input_data = prepare(args.batch_size, use_cuda)

    if args.run:
        result = model(input_data)
        logger.info(f"  Data: {result.sum()}")

    logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")


if __name__ == "__main__":
    main()
