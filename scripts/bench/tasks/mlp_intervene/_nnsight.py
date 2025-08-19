"""
MLP intervention task - Standalone script.
"""

import torch
import torch.nn as nn
from typing import Tuple
import argparse
import numpy as np
from loguru import logger
from nnsight import NNsight


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
    height: int,
    width: int,
    batch_size: int,
    use_cuda: bool,
) -> Tuple[NNsight, torch.Tensor]:
    """Prepare the model and input data."""
    model = MLP(height=height, width=width).to("cuda" if use_cuda else "cpu")
    input_data = torch.randn(batch_size, width).to("cuda" if use_cuda else "cpu")
    return NNsight(model), input_data


def run(
    model: NNsight,
    input_data: torch.Tensor,
    variation: int,
) -> tuple:
    """Benchmark a single model configuration."""
    with model.trace(input_data):
        state1 = model.layers[0].output.save()
        model.layers[-2].output[:] = state1
        for _ in range(variation):
            hidden_states = model.layers[-2].output + 1
            model.layers[-2].output[:] = hidden_states

        state2 = model.layers[-2].output.save()
        state3 = model.layers[-1](state2).save()

    return (state1, state2, state3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("mlp-intervene-nnsight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--variation", type=int, default=10)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    use_cuda = cuda_available and args.cuda
    logger.info("Running MLP intervention task with NNsight:")
    logger.info(f"  Spawn max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.0f} KB")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Width: {args.width}")
    logger.info(f"  Height: {args.height}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Variation: {args.variation}")
    logger.info(f"  Use CUDA: {use_cuda}")

    model, input_data = prepare(args.height, args.width, args.batch_size, use_cuda)

    if args.run:
        try:
            result = run(model, input_data, args.variation)
            logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    else:
        logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")


if __name__ == "__main__":
    main()
