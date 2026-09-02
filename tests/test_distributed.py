import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tdhook.latent import ActivationCaching
from tdhook.session import HookSession
from tdhook.targets import Target
from tdhook.workflow import Workflow


class _DistributedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Identity()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, value):
        return self.linear(self.module(value))


def _run_rank_local_sessions(rank: int, world_size: int, rendezvous: str, output_directory: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _DistributedModel()
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))
        model = DistributedDataParallel(model)
        target = Target("module", "activation", -1, (0,))
        value = torch.tensor([[float(rank + 1), float(rank + 2)]])
        target_module = model.module.module
        assert Target("", "activation", -1, (0,)).validate(model) is model.module
        assert target.validate(model) is target_module
        original_hook_count = len(target_module._forward_hooks)

        with HookSession(model) as session:
            captured = session.capture(target)
            session.replace(target, rank + 10)
            replaced = model(value)

        assert captured.value is not None
        assert len(target_module._forward_hooks) == original_hook_count

        workflow = Workflow(ActivationCaching(target, cache_key="activations"))
        workflow_result = workflow(model, TensorDict({"input": value}, batch_size=[1]))
        workflow_capture = workflow_result["activations", "module"]
        assert len(target_module._forward_hooks) == original_hook_count

        with HookSession(model) as session:
            stopped = session.stop("module")
            model(value)

        assert stopped.reached
        assert isinstance(stopped.output, torch.Tensor)
        assert len(target_module._forward_hooks) == original_hook_count

        result = {
            "rank": rank,
            "captured": captured.value.tolist(),
            "replaced": replaced.tolist(),
            "workflow_capture": workflow_capture.tolist(),
            "stopped": stopped.output.tolist(),
        }
        Path(output_directory, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_hook_sessions_and_workflows_are_rank_local_under_ddp(tmp_path):
    """Exercise the public DDP contract in two independent CPU processes."""

    world_size = 2
    rendezvous = tmp_path / "gloo-rendezvous"
    mp.spawn(
        _run_rank_local_sessions,
        args=(world_size, str(rendezvous), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(world_size)]
    assert [result["rank"] for result in results] == [0, 1]
    assert [result["captured"] for result in results] == [[[1.0]], [[2.0]]]
    assert [result["replaced"] for result in results] == [[[10.0, 2.0]], [[11.0, 3.0]]]
    assert [result["workflow_capture"] for result in results] == [[[1.0]], [[2.0]]]
    assert [result["stopped"] for result in results] == [[[1.0, 2.0]], [[2.0, 3.0]]]
