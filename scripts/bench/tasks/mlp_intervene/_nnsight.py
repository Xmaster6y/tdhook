"""
MLP intervention task - Standalone script.
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
from loguru import logger
from nnsight import NNsight


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
) -> NNsight:
    """Prepare the model and input data."""
    model = MLP(height=height, width=width)
    input_data = torch.randn(batch_size, width)
    return model, input_data


def spawn(
    model: MLP,
) -> NNsight:
    """Spawn the model."""
    return NNsight(model)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--variation", type=int, default=10)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare model and input
    model, input_data = prepare(args.height, args.width, args.batch_size)
    nnsight_model = spawn(model)

    logger.info("Running MLP intervention task with NNsight:")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Width: {args.width}")
    logger.info(f"  Height: {args.height}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Variation: {args.variation}")

    # Run the benchmark
    if args.run:
        try:
            result = run(nnsight_model, input_data, args.variation)
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise


if __name__ == "__main__":
    main()
