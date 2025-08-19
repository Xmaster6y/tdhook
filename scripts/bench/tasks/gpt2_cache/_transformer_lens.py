"""
MLP intervention task - Standalone script.
"""

import torch
import argparse
import numpy as np
from loguru import logger
from typing import Dict, Tuple

from transformer_lens import HookedTransformer


cuda_available = torch.cuda.is_available()


def prepare(
    model_size: str,
    batch_size: int,
    use_cuda: bool,
) -> Tuple[HookedTransformer, Dict[str, torch.Tensor]]:
    """Prepare the model and input data."""
    model = HookedTransformer.from_pretrained(
        model_size, device="cuda" if use_cuda else "cpu", center_writing_weights=False, center_unembed=False
    )
    sentences = ["Hello, world!"] * batch_size
    input_data = model.tokenizer(sentences, return_tensors="pt")
    input_data = {
        k if k != "input_ids" else "input": v.to("cuda" if use_cuda else "cpu") for k, v in input_data.items()
    }
    return model, input_data


def run(
    model: HookedTransformer,
    input_data: Dict[str, torch.Tensor],
    variation: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Benchmark a single model configuration."""
    return model.run_with_cache(**input_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("gpt2-cache-transformer-lens")
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
    logger.info("Running GPT2 cache task with TransformerLens:")
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
