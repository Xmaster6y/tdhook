"""
Integrated gradients task - Standalone script.
"""

import torch
import torch.nn as nn
from typing import Tuple
import argparse
import numpy as np
from loguru import logger

from captum.attr import IntegratedGradients


cuda_available = torch.cuda.is_available()


class MLP(nn.Module):
    def __init__(self, height: int = 10, width: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(height)])
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x[:, 4] + x[:, 7]


def prepare(
    height: int,
    width: int,
    batch_size: int,
    use_cuda: bool,
    variation: str,
) -> Tuple[IntegratedGradients, torch.Tensor, torch.Tensor]:
    """Prepare the model and input data."""
    model = MLP(height=height, width=width).to("cuda" if use_cuda else "cpu")
    input_data = torch.randn(batch_size, width).to("cuda" if use_cuda else "cpu")
    baseline = torch.zeros_like(input_data)
    return IntegratedGradients(model, multiply_by_inputs=variation == "multiply"), input_data, baseline


def run(
    ig: IntegratedGradients,
    input_data: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple:
    """Benchmark a single model configuration."""
    attributions, delta = ig.attribute(input_data, baseline, target=None, return_convergence_delta=True)
    return attributions, delta


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    use_cuda = cuda_available and args.cuda
    logger.info("Running integrated gradients task with Captum:")
    logger.info(f"  Spawn max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.0f} KB")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Width: {args.width}")
    logger.info(f"  Height: {args.height}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Variation: {args.variation}")
    logger.info(f"  Use CUDA: {use_cuda}")

    ig, input_data, baseline = prepare(args.height, args.width, args.batch_size, use_cuda, args.variation)

    if args.run:
        try:
            result = run(ig, input_data, baseline)
            logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    else:
        logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("integrated-gradients-captum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--variation", type=str, default="multiply")
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
