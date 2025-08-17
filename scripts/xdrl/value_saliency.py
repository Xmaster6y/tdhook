"""
Script to compute saliency maps for a value network using tdhook.

Run with:

```
uv run --group scripts -m scripts.xdrl.value_saliency
```
"""

import argparse
from loguru import logger
import torch
from torch import nn
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter, DeviceCastTransform
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ValueOperator

from tdhook.attribution.gradient_attribution import Saliency
from tdhook.module import get_best_device

device = get_best_device()


@torch.no_grad()
def setup(args):
    base_env = GymEnv(args.env_name)
    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
            DeviceCastTransform(device=device),
        ),
    )

    value_net = nn.Sequential(
        nn.LazyLinear(args.num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(args.num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(args.num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_head = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
    return env, value_head


def compute_saliency(value_head, batch):
    """Compute saliency maps using tdhook."""
    saliency_context = Saliency()

    with saliency_context.prepare(value_head) as hooked_policy:
        output = hooked_policy(batch)

        saliency_maps = output.get(("attr", "observation"))

        return saliency_maps


def main(args):
    """Run saliency map demo."""
    logger.info("Starting value saliency demo...")

    env, value_head = setup(args)

    logger.info(f"Value created with {len(list(value_head.parameters()))} parameters")

    batch = env.rollout(1000)

    saliency_maps = compute_saliency(value_head, batch)

    logger.info(f"Saliency maps computed: {saliency_maps.shape}")
    logger.info(f"Mean absolute importance: {saliency_maps.abs().mean().item():.4f}")
    logger.info("Demo completed!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("value-saliency")
    parser.add_argument("--env_name", type=str, default="InvertedDoublePendulum-v4")
    parser.add_argument("--num_cells", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
