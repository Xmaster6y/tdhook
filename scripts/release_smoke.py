"""Smoke-test an installed TDHook distribution, never the source checkout."""

from importlib.metadata import version
import sys

import torch
from tensordict import TensorDict
from torch import nn

from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow


def main(expected_version: str) -> None:
    installed_version = version("tdhook")
    if installed_version != expected_version:
        raise RuntimeError(f"Expected tdhook {expected_version}, found {installed_version}")

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    workflow = Workflow(ActivationCaching("0", cache_key=("activations", "hidden")))
    data = TensorDict({"input": torch.ones(2, 3)}, batch_size=[2])

    plan = workflow.plan(model, data)
    result = workflow(model, data)

    if plan.model_passes != 1:
        raise RuntimeError(f"Expected one model pass, found {plan.model_passes}")
    if result["output"].shape != (2, 2):
        raise RuntimeError(f"Unexpected model output shape: {result['output'].shape}")
    if result["activations", "hidden"]["0"].shape != (2, 4):
        raise RuntimeError("Installed workflow did not publish the captured activation")

    print(f"tdhook {installed_version} wheel smoke passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: release_smoke.py EXPECTED_VERSION")
    main(sys.argv[1])
