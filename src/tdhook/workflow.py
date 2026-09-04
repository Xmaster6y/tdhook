"""Direct TensorDict-native composition of configured methods."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import Tensor, nn

from tdhook._types import tensor_leaf_items
from tdhook.methods import BoundMethod
from tdhook.execution import AutogradLifetime, ExecutionSpec, GradientMode


@runtime_checkable
class WorkflowMethod(Protocol):
    """The configured-method surface consumed by :class:`Workflow`."""

    @property
    def execution_spec(self) -> ExecutionSpec: ...

    def bind(self, module: nn.Module) -> BoundMethod: ...


@dataclass(frozen=True)
class WorkflowUpdate:
    """Explicitly permit one step to replace an earlier workflow-owned output."""

    step: WorkflowMethod | TensorDictModuleBase


class WorkflowHandoffError(RuntimeError):
    """A workflow cannot safely preserve a process-handoff artifact."""


WorkflowStep = WorkflowMethod | TensorDictModuleBase | WorkflowUpdate


@dataclass(frozen=True)
class _BoundMethodNode:
    name: str
    in_keys: tuple[NestedKey, ...]
    out_keys: tuple[NestedKey, ...]
    model_in_keys: tuple[NestedKey, ...]
    model_out_keys: tuple[NestedKey, ...]
    execution_spec: ExecutionSpec
    module: TensorDictModuleBase
    binding: BoundMethod
    allow_output_overwrite: bool = False


@dataclass(frozen=True)
class _OperatorNode:
    name: str
    operator: TensorDictModuleBase
    in_keys: tuple[NestedKey, ...]
    out_keys: tuple[NestedKey, ...]
    allow_output_overwrite: bool = False


_ExecutionNode = _BoundMethodNode | _OperatorNode


def _artifact_leaf_keys(data: TensorDictBase) -> frozenset[NestedKey]:
    return frozenset(key for key, _ in tensor_leaf_items(data))


@dataclass(frozen=True)
class _HandoffArtifact:
    """Storage facts for a shared or consolidated TensorDict."""

    data: TensorDictBase
    keys: frozenset[NestedKey]
    batch_size: torch.Size
    device: torch.device | None
    shared: bool
    consolidated: bool

    @classmethod
    def inspect(cls, data: TensorDictBase) -> "_HandoffArtifact | None":
        shared = data.is_shared()
        consolidated = data.is_consolidated()
        if not (shared or consolidated):
            return None

        for key, value in tensor_leaf_items(data):
            if not isinstance(value, Tensor):
                raise WorkflowHandoffError(
                    f"Workflow handoff artifact key {key!r} must contain a Tensor, got {type(value).__name__}"
                )
            if value.device.type != "cpu":
                raise WorkflowHandoffError(
                    f"Workflow handoff artifact key {key!r} is on {value.device}; local process handoff requires CPU tensors"
                )
            if value.requires_grad:
                raise WorkflowHandoffError(
                    f"Workflow handoff artifact key {key!r} requires gradients; detach tensors before process handoff"
                )

        return cls(
            data=data,
            keys=_artifact_leaf_keys(data),
            batch_size=data.batch_size,
            device=data.device,
            shared=shared,
            consolidated=consolidated,
        )

    def working_copy(self) -> TensorDictBase:
        # Clone only the TensorDict containers. Tensor leaves retain their native
        # shared or consolidated storage until a declared output is committed.
        return self.data.clone(recurse=False)

    def validate_step(self, out_keys: Sequence[NestedKey], execution_spec: ExecutionSpec | None) -> None:
        if execution_spec is not None and execution_spec.autograd_lifetime is AutogradLifetime.BACKWARD:
            raise WorkflowHandoffError(
                "Workflow handoff artifacts cannot retain an autograd graph for deferred backward execution"
            )
        missing = tuple(key for key in out_keys if key != "_" and key not in self.keys)
        if missing:
            raise WorkflowHandoffError(
                "Workflow handoff artifact storage requires every output to be preallocated; "
                f"missing output keys: {list(dict.fromkeys(missing))!r}"
            )

    def commit(
        self,
        current: TensorDictBase,
        working: TensorDictBase,
        step_name: str,
        out_keys: Sequence[NestedKey],
    ) -> None:
        if current is not working:
            raise WorkflowHandoffError(
                f"Workflow step {step_name!r} returned a different TensorDict; "
                "handoff workflows must mutate their input TensorDict in place"
            )
        if current.batch_size != self.batch_size or current.device != self.device:
            raise WorkflowHandoffError(f"Workflow step {step_name!r} changed artifact batch size or device metadata")
        if _artifact_leaf_keys(current) != self.keys:
            raise WorkflowHandoffError(f"Workflow step {step_name!r} changed artifact keys")

        for key in out_keys:
            if key == "_":
                continue
            source = current.get(key)
            destination = self.data.get(key)
            if not isinstance(source, Tensor) or not isinstance(destination, Tensor):
                raise WorkflowHandoffError(f"Workflow handoff output {key!r} must remain a Tensor")
            if (source.shape, source.dtype, source.device) != (
                destination.shape,
                destination.dtype,
                destination.device,
            ):
                raise WorkflowHandoffError(
                    f"Workflow handoff output {key!r} has shape/dtype/device "
                    f"{(source.shape, source.dtype, source.device)!r}, expected "
                    f"{(destination.shape, destination.dtype, destination.device)!r}"
                )
            with torch.no_grad():
                destination.copy_(source.detach())
                if self.consolidated:
                    self._consolidated_destination(key, destination).copy_(source.detach())

    def _consolidated_destination(self, key: NestedKey, destination: Tensor) -> Tensor:
        """Return the canonical consolidated-storage view for ``key``.

        TensorDict multiprocessing reconstruction can leave leaf objects detached
        from its canonical consolidated buffer. Updating the buffer as well keeps
        subsequent native TensorDict handoffs consistent.
        """

        key = (key,) if isinstance(key, str) else key
        consolidated = getattr(self.data, "_consolidated", None)
        if not isinstance(consolidated, Mapping):
            raise WorkflowHandoffError("TensorDict reported consolidated storage without storage metadata")
        metadata = consolidated.get("metadata")
        storage = consolidated.get("storage")
        if not isinstance(metadata, Mapping) or not isinstance(storage, Tensor):
            raise WorkflowHandoffError("Consolidated TensorDict storage metadata is unavailable")
        for part in key[:-1]:
            metadata = metadata.get(part)
            if not isinstance(metadata, Mapping):
                raise WorkflowHandoffError(f"Consolidated storage metadata is missing output {key!r}")
        leaves = metadata.get("leaves")
        if not isinstance(leaves, Mapping) or key[-1] not in leaves:
            raise WorkflowHandoffError(f"Consolidated storage metadata is missing output {key!r}")
        _, _, start, stop, _ = leaves[key[-1]]
        return storage[start:stop].view(destination.dtype)[: destination.numel()].view(destination.shape)

    def result(self) -> TensorDictBase:
        if (
            self.data.batch_size != self.batch_size
            or self.data.device != self.device
            or self.data.is_shared() != self.shared
            or self.data.is_consolidated() != self.consolidated
            or _artifact_leaf_keys(self.data) != self.keys
        ):
            raise WorkflowHandoffError("Workflow changed native handoff artifact storage or metadata")
        return self.data


def _step_name(index: int, step: WorkflowStep) -> str:
    return f"{index}:{type(step).__name__}"


def _method_step(step: object) -> bool:
    return isinstance(step, WorkflowMethod) and not isinstance(step, TensorDictModuleBase)


def _method_spec(method: WorkflowMethod) -> ExecutionSpec:
    spec = method.execution_spec
    if not isinstance(spec, ExecutionSpec):
        raise TypeError(f"{type(method).__name__}.execution_spec must be an ExecutionSpec, got {type(spec).__name__}")
    return spec


def _bind_method(
    index: int,
    method: WorkflowMethod,
    model: nn.Module,
    *,
    allow_output_overwrite: bool = False,
) -> _BoundMethodNode:
    binding = method.bind(model)
    if not isinstance(binding, BoundMethod):
        raise TypeError(f"{type(method).__name__}.bind(model) must return a BoundMethod, got {type(binding).__name__}")
    bound_module = binding.module
    if not isinstance(bound_module, TensorDictModuleBase):
        raise TypeError(
            f"{type(method).__name__}.bind(model) must bind a TensorDictModuleBase, got {type(bound_module).__name__}"
        )
    if not isinstance(getattr(bound_module, "td_module", None), TensorDictModuleBase) or not callable(
        getattr(bound_module, "finalize_tensordict", None)
    ):
        raise TypeError(
            f"{type(method).__name__}.bind(model) returned an invalid bound module, got {type(bound_module).__name__}"
        )
    return _BoundMethodNode(
        name=_step_name(index, method),
        in_keys=tuple(bound_module.in_keys),
        out_keys=tuple(bound_module.out_keys),
        model_in_keys=binding.model_in_keys,
        model_out_keys=binding.model_out_keys,
        execution_spec=_method_spec(method),
        module=bound_module,
        binding=binding,
        allow_output_overwrite=allow_output_overwrite,
    )


def _available_keys(data: TensorDictBase) -> set[NestedKey]:
    return set(data.keys(include_nested=True, leaves_only=False))


def _key_path(key: NestedKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _key_paths_overlap(left: NestedKey, right: NestedKey) -> bool:
    left_path = _key_path(left)
    right_path = _key_path(right)
    common = min(len(left_path), len(right_path))
    return left_path[:common] == right_path[:common]


def _validate_workflow_method(node: _BoundMethodNode) -> None:
    program = node.binding.program
    if (
        node.execution_spec.gradient_mode is not GradientMode.REQUIRED
        and program is not None
        and any(spec.direction in {"bwd", "bwd_pre"} for spec in program.hooks)
    ):
        raise ValueError(
            f"Workflow step {node.name!r} installs backward hooks but does not own an autograd-enabled execution"
        )


class _DeferredAutogradCleanup:
    """Close method bindings after the caller's autograd engine finishes."""

    def __init__(self, stack: ExitStack):
        self._stack = stack
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._queued = False
        self._closed = False

    def arm(self, data: TensorDictBase, keys: Sequence[NestedKey]) -> None:
        tensors = []
        for key in keys:
            value = data.get(key)
            if isinstance(value, Tensor) and value.requires_grad:
                tensors.append(value)
        if not tensors:
            self.close()
            raise RuntimeError("Deferred-backward workflow execution did not produce an autograd-enabled model output")
        self._handles = [tensor.register_hook(self._queue_cleanup) for tensor in tensors]

    def _queue_cleanup(self, gradient: Tensor) -> Tensor:
        if not self._queued:
            self._queued = True
            torch.autograd.Variable._execution_engine.queue_callback(self.close)
        return gradient

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_error = None
        try:
            for handle in self._handles:
                try:
                    handle.remove()
                except BaseException as error:
                    cleanup_error = cleanup_error or error
        finally:
            self._handles = []
            try:
                self._stack.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise cleanup_error


def _provided_namespace_covers(provided: NestedKey, required: NestedKey) -> bool:
    provided_path = _key_path(provided)
    required_path = _key_path(required)
    return len(provided_path) < len(required_path) and required_path[: len(provided_path)] == provided_path


def _validate_dependencies(nodes: Sequence[_ExecutionNode], data: TensorDictBase) -> None:
    available = _available_keys(data)
    provided_namespaces: set[NestedKey] = set()
    owned_outputs: set[NestedKey] = set()
    for node in nodes:
        missing = tuple(
            key
            for key in node.in_keys
            if key not in available
            and not any(_provided_namespace_covers(provided, key) for provided in provided_namespaces)
        )
        if missing:
            raise ValueError(f"Workflow step {node.name!r} requires missing TensorDict keys: {missing!r}")
        owned = (
            node.out_keys
            if isinstance(node, _OperatorNode)
            else tuple(key for key in node.out_keys if key not in node.model_out_keys)
        )
        collisions = tuple(
            (key, previous) for key in owned for previous in owned_outputs if _key_paths_overlap(key, previous)
        )
        if collisions and not node.allow_output_overwrite:
            raise ValueError(
                f"Workflow step {node.name!r} overlaps earlier workflow-owned outputs: {collisions!r}; "
                "wrap an intentional replacement in WorkflowUpdate"
            )
        available.update(node.out_keys)
        provided_namespaces.update(node.out_keys)
        owned_outputs.update(owned)


def _validate_runtime_dependencies(nodes: Sequence[_ExecutionNode], data: TensorDictBase) -> None:
    for node in nodes:
        missing = tuple(key for key in node.in_keys if key not in data)
        if missing:
            raise ValueError(f"Workflow step {node.name!r} requires missing TensorDict keys: {missing!r}")


class Workflow:
    """Execute configured methods and ordinary TensorDict modules in order.

    Execution is rank-local when ``model`` is a ``DistributedDataParallel``
    instance. TDHook does not initialize process groups, launch workers, or
    aggregate TensorDicts or captured values.
    """

    def __init__(self, *steps: WorkflowStep):
        self.steps = tuple(steps)
        for index, step in enumerate(self.steps):
            actual = step.step if isinstance(step, WorkflowUpdate) else step
            if not isinstance(actual, TensorDictModuleBase) and not _method_step(actual):
                raise TypeError(
                    f"Workflow step {index} must be a configured method or TensorDictModuleBase, "
                    f"got {type(step).__name__}"
                )

    def run(self, model: nn.Module, data: TensorDictBase) -> TensorDictBase:
        """Execute each step once and return the resulting TensorDict.

        TensorDicts prepared with :meth:`TensorDict.share_memory_` or
        :meth:`TensorDictBase.consolidate` retain their native storage. Their
        output keys must already exist with the expected tensor metadata, and
        each workflow step must use in-place TensorDict semantics.
        """

        if not isinstance(model, nn.Module):
            raise TypeError(f"Workflow model must be a torch.nn.Module, got {type(model).__name__}")
        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow data must be a TensorDict, got {type(data).__name__}")
        nodes: list[_ExecutionNode] = []
        for index, step in enumerate(self.steps):
            overwrite = isinstance(step, WorkflowUpdate)
            actual = step.step if overwrite else step
            name = _step_name(index, actual)
            if isinstance(actual, TensorDictModuleBase):
                nodes.append(
                    _OperatorNode(
                        name=name,
                        operator=actual,
                        in_keys=tuple(actual.in_keys),
                        out_keys=tuple(actual.out_keys),
                        allow_output_overwrite=overwrite,
                    )
                )
            else:
                nodes.append(_bind_method(index, actual, model, allow_output_overwrite=overwrite))

        _validate_dependencies(nodes, data)
        for index, node in enumerate(nodes[:-1]):
            if (
                isinstance(node, _BoundMethodNode)
                and node.execution_spec.autograd_lifetime is AutogradLifetime.BACKWARD
            ):
                if any(isinstance(later, _BoundMethodNode) for later in nodes[index + 1 :]):
                    raise ValueError(
                        "A deferred-backward workflow method cannot precede another model-executing method"
                    )

        handoff = _HandoffArtifact.inspect(data)
        if handoff is not None:
            for node in nodes:
                spec = node.execution_spec if isinstance(node, _BoundMethodNode) else None
                handoff.validate_step(node.out_keys, spec)
        current = handoff.working_copy() if handoff is not None else data
        working = current

        for node in nodes:
            _validate_runtime_dependencies((node,), current)
            if isinstance(node, _OperatorNode):
                current = node.operator(current)
                if not isinstance(current, TensorDictBase):
                    raise TypeError(
                        f"Workflow operator {node.name!r} must return a TensorDict, got {type(current).__name__}"
                    )
                if handoff is not None:
                    handoff.commit(current, working, node.name, node.out_keys)
                continue

            with ExitStack() as stack:
                bound_module = stack.enter_context(node.binding)
                if not isinstance(bound_module, TensorDictModuleBase):
                    raise TypeError(
                        f"Workflow method {node.name!r} must enter a TensorDictModuleBase, "
                        f"got {type(bound_module).__name__}"
                    )
                if bound_module is not node.module:
                    raise TypeError(
                        f"Workflow method {node.name!r} entered an invalid bound module, "
                        f"got {type(bound_module).__name__}"
                    )
                _validate_workflow_method(node)
                result = node.module(current)
                if result is not None:
                    current = result
                if not isinstance(current, TensorDictBase):
                    raise TypeError(
                        f"Workflow method execution must return a TensorDict, got {type(current).__name__}"
                    )
                if node.execution_spec.autograd_lifetime is AutogradLifetime.BACKWARD:
                    cleanup = _DeferredAutogradCleanup(stack.pop_all())
                    node.binding.on_hook_failure(cleanup.close)
                    cleanup.arm(current, node.model_out_keys)
            if handoff is not None:
                handoff.commit(current, working, node.name, node.out_keys)
        if handoff is not None:
            current = handoff.result()
        return current

    def __call__(self, model: nn.Module, data: TensorDictBase) -> TensorDictBase:
        return self.run(model, data)


__all__ = [
    "Workflow",
    "WorkflowHandoffError",
    "WorkflowMethod",
    "WorkflowUpdate",
]
