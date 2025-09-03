"""
MLP intervention task - Standalone script.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import argparse
import numpy as np
from loguru import logger
from tensordict import TensorDict

from tdhook.modules import HookedModule


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
) -> Tuple[HookedModule, torch.Tensor]:
    """Prepare the model and input data."""
    model = MLP(height=height, width=width).to("cuda" if use_cuda else "cpu")
    input_data = torch.randn(batch_size, width).to("cuda" if use_cuda else "cpu")
    return HookedModule.from_module(model, in_keys=["input"], out_keys=["output"]), input_data


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("mlp-intervene-tdhook")
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
    logger.info("Running MLP intervention task with TDhook:")
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
