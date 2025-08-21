"""
Integrated gradients task - Standalone script.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import argparse
import numpy as np
from loguru import logger
from tensordict import TensorDict


from tdhook.attribution.gradient_attribution.integrated_gradients import IntegratedGradients
from tdhook.module import HookedModule
from tdhook.contexts import HookingContext

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
    variation: str,
) -> Tuple[HookedModule, torch.Tensor, torch.Tensor]:
    """Prepare the model and input data."""
    model = MLP(height=height, width=width).to("cuda" if use_cuda else "cpu")
    input_data = torch.randn(batch_size, width).to("cuda" if use_cuda else "cpu")
    baseline = torch.zeros_like(input_data)
    context_factory = IntegratedGradients(
        init_attr_targets=lambda td, _: td.apply(lambda t: t[..., (4, 7)]),
        multiply_by_inputs=variation == "multiply",
        compute_convergence_delta=True,
    )
    return context_factory.prepare(model), input_data, baseline


def run(
    hooking_context: HookingContext,
    input_data: torch.Tensor,
    baseline: torch.Tensor,
) -> Dict[str, float]:
    """Benchmark a single model configuration."""
    with hooking_context as hooked_module:
        output = hooked_module(
            TensorDict(
                {"input": input_data, ("baseline", "input"): baseline},
                batch_size=input_data.shape[0],
                device=input_data.device,
            )
        )
        return output[("attr", "input")], output["convergence_delta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("integrated-gradients-tdhook")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--variation", type=str, default="multiply")
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
    logger.info("Running integrated gradients task with TDhook:")
    logger.info(f"  Spawn max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.0f} KB")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Width: {args.width}")
    logger.info(f"  Height: {args.height}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Variation: {args.variation}")
    logger.info(f"  Use CUDA: {use_cuda}")

    hooking_context, input_data, baseline = prepare(args.height, args.width, args.batch_size, use_cuda, args.variation)

    if args.run:
        try:
            result = run(hooking_context, input_data, baseline)
            logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    else:
        logger.info(f"  Max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.2f} KB")


if __name__ == "__main__":
    main()
