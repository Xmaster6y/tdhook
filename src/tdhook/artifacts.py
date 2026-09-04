"""Validated transport for TensorDict workflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torch import Tensor

from tdhook._types import tensor_leaf_items


class WorkflowArtifactError(RuntimeError):
    """A TensorDict does not satisfy a distributed workflow artifact contract."""


@dataclass(frozen=True)
class WorkflowArtifactTensorSpec:
    """Expected metadata for one tensor in a distributed workflow artifact."""

    key: NestedKey
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class WorkflowArtifactSchema:
    """The fixed layout used by TensorDict-native distributed transport."""

    tensors: tuple[WorkflowArtifactTensorSpec, ...]
    batch_size: torch.Size
    device: torch.device | None

    @classmethod
    def from_tensordict(cls, data: TensorDictBase) -> "WorkflowArtifactSchema":
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow artifact must be a TensorDict, got {type(data).__name__}")
        tensors = []
        for key, value in tensor_leaf_items(data):
            if not isinstance(value, Tensor):
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} must contain a Tensor, got {type(value).__name__}"
                )
            tensors.append(WorkflowArtifactTensorSpec(key, value.shape, value.dtype, value.device))
        return cls(tuple(tensors), data.batch_size, data.device)

    @property
    def keys(self) -> tuple[NestedKey, ...]:
        return tuple(tensor.key for tensor in self.tensors)

    def validate(self, data: TensorDictBase) -> None:
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow artifact must be a TensorDict, got {type(data).__name__}")
        if data.batch_size != self.batch_size:
            raise WorkflowArtifactError(
                f"Workflow artifact batch size is {data.batch_size}, expected {self.batch_size}"
            )
        if data.device != self.device:
            raise WorkflowArtifactError(f"Workflow artifact device is {data.device}, expected {self.device}")

        items = tuple(tensor_leaf_items(data))
        keys = tuple(key for key, _ in items)
        if keys != self.keys:
            raise WorkflowArtifactError(f"Workflow artifact keys are {keys!r}, expected {self.keys!r}")
        for expected, (key, value) in zip(self.tensors, items, strict=True):
            if not isinstance(value, Tensor):
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} must contain a Tensor, got {type(value).__name__}"
                )
            if value.shape != expected.shape:
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} shape is {value.shape}, expected {expected.shape}"
                )
            if value.dtype != expected.dtype:
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} dtype is {value.dtype}, expected {expected.dtype}"
                )
            if value.device != expected.device:
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} device is {value.device}, expected {expected.device}"
                )
            if value.requires_grad:
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} requires gradients; detach it before transport"
                )


def _validate_distributed_artifact(
    data: TensorDictBase,
    schema: WorkflowArtifactSchema,
    group: dist.ProcessGroup | None,
) -> None:
    schema.validate(data)
    if not dist.is_available() or not dist.is_initialized():
        raise WorkflowArtifactError("An externally managed torch.distributed process group must be initialized")
    if not data.is_consolidated():
        raise WorkflowArtifactError("Distributed workflow artifacts must use consolidated TensorDict storage")
    if dist.get_backend(group) == dist.Backend.NCCL:
        raise WorkflowArtifactError(
            "NCCL does not support point-to-point tags; Workflow artifact transport requires a tag-capable backend"
        )


def send_workflow_artifact(
    data: TensorDictBase,
    dst: int,
    *,
    schema: WorkflowArtifactSchema,
    group: dist.ProcessGroup | None = None,
    init_tag: int = 0,
) -> None:
    """Send a validated, consolidated TensorDict through an existing group."""

    _validate_distributed_artifact(data, schema, group)
    data.send(dst, group=group, init_tag=init_tag)


def receive_workflow_artifact(
    data: TensorDictBase,
    src: int,
    *,
    schema: WorkflowArtifactSchema,
    group: dist.ProcessGroup | None = None,
    init_tag: int = 0,
) -> TensorDictBase:
    """Receive tensor values into a validated, preallocated artifact template."""

    _validate_distributed_artifact(data, schema, group)
    data.recv(src, group=group, init_tag=init_tag)
    schema.validate(data)
    if not data.is_consolidated():
        raise WorkflowArtifactError("TensorDict transport changed consolidated artifact storage")
    return data


__all__ = [
    "WorkflowArtifactError",
    "WorkflowArtifactSchema",
    "WorkflowArtifactTensorSpec",
    "receive_workflow_artifact",
    "send_workflow_artifact",
]
