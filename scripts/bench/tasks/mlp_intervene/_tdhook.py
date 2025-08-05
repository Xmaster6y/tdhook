"""
MLP intervention task - Standalone script.
"""

import torch
import torch.nn as nn
from typing import Dict
import argparse
import numpy as np
from loguru import logger
from tensordict import TensorDict

from tdhook.module import HookedModule


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
) -> HookedModule:
    """Prepare the model and input data."""
    model = MLP(height=height, width=width)
    input_data = torch.randn(batch_size, width)
    return model, input_data


def spawn(
    model: MLP,
) -> HookedModule:
    """Spawn the model."""
    return HookedModule(model, in_keys=["input"], out_keys=["output"])


def run(
    model: HookedModule,
    input_data: torch.Tensor,
    variation: int,
) -> Dict[str, float]:
    """Benchmark a single model configuration."""

    with model.run(TensorDict({"input": input_data})) as run:
        state1 = run.get("layers[0]")
        run.set("layers[-2]", state1)
        for _ in range(variation):
            run.set("layers[-2]", None, callback=lambda **kwargs: kwargs["output"] + 1)
        state2 = run.get("layers[-2]", cache_key="tdhook_state2")
        state3 = run.get(
            "layers[-2]", cache_key="tdhook_state3", callback=lambda **kwargs: model.layers[-1](kwargs["output"])
        )

    return (state1.resolve(), state2.resolve(), state3.resolve())


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
    tdhook_model = spawn(model)

    logger.info("Running MLP intervention task with TDhook:")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Width: {args.width}")
    logger.info(f"  Height: {args.height}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Variation: {args.variation}")

    # Run the benchmark
    if args.run:
        try:
            result = run(tdhook_model, input_data, args.variation)
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise


if __name__ == "__main__":
    main()
