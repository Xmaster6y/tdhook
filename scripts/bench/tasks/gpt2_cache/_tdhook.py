"""
MLP intervention task - Standalone script.
"""

import torch
from typing import Dict, Tuple
import argparse
import numpy as np
from loguru import logger
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from tdhook.module import HookedModule
from tdhook.latent.activation_caching import ActivationCaching


cuda_available = torch.cuda.is_available()


def prepare(
    model_size: str,
    batch_size: int,
    use_cuda: bool,
) -> Tuple[HookedModule, torch.Tensor]:
    """Prepare the model and input data."""
    model = AutoModelForCausalLM.from_pretrained(model_size, device_map="cuda" if use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_size)
    sentences = ["Hello, world!"] * batch_size
    input_data = tokenizer(sentences, return_tensors="pt")
    input_data = {k: v.to("cuda" if use_cuda else "cpu") for k, v in input_data.items()}
    return HookedModule.from_module(model, in_keys={k: k for k in input_data.keys()}, out_keys=["output"]), input_data


def run(
    model: HookedModule,
    input_data: Dict[str, torch.Tensor],
    variation: str,
) -> Dict[str, float]:
    """Benchmark a single model configuration."""

    def callback(**kwargs):
        if isinstance(kwargs["output"], tuple):
            return kwargs["output"][0]
        return kwargs["output"]

    context = ActivationCaching(key_pattern="^(?!transformer$).*", callback=callback, relative=True)
    with context._hook_module(model):
        shuttle = TensorDict(input_data)
        model(shuttle)
    return shuttle["output", "logits"], context.cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("gpt2-cache-tdhook")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_size", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--variation", type=str, default="specific")
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
    logger.info("Running GPT2 cache task with TDhook:")
    logger.info(f"  Spawn max GPU memory: {torch.cuda.max_memory_allocated() / 1024:.0f} KB")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Variation: {args.variation}")
    logger.info(f"  Use CUDA: {use_cuda}")

    model, input_data = prepare(args.model_size, args.batch_size, use_cuda)

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
