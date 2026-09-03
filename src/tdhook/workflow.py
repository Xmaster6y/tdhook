"""TensorDict-native composition and conservative method execution planning."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Generator, Protocol, Sequence, runtime_checkable

import torch
import torch.distributed as dist
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import Tensor, nn

from tdhook.contexts import HookingContext
from tdhook.execution import AutogradLifetime, ExecutionSpec, GradientMode
from tdhook.runtime import HookProgram, TargetOccurrenceEvidence
from tdhook.session import HookSession


@runtime_checkable
class WorkflowMethod(Protocol):
    """The configured-method surface consumed by :class:`Workflow`."""

    @property
    def execution_spec(self) -> ExecutionSpec: ...

    def prepare(self, module: nn.Module) -> HookingContext: ...


@dataclass(frozen=True)
class WorkflowUpdate:
    """Explicitly permit one step to replace an earlier workflow-owned output."""

    step: WorkflowMethod | TensorDictModuleBase


class WorkflowHandoffError(RuntimeError):
    """A workflow cannot safely preserve a process-handoff artifact."""


class WorkflowArtifactError(RuntimeError):
    """A TensorDict does not satisfy a distributed workflow artifact contract."""


WorkflowStep = WorkflowMethod | TensorDictModuleBase | WorkflowUpdate


class MethodBinding(Protocol):
    """Prepared method surface used for execution after facts are validated."""

    in_keys: list[NestedKey]
    out_keys: list[NestedKey]

    @property
    def td_module(self) -> TensorDictModuleBase: ...

    def __call__(self, data: TensorDictBase) -> TensorDictBase: ...

    def finalize_tensordict(self, data: TensorDictBase) -> TensorDictBase: ...


@dataclass(frozen=True)
class PlannedExecution:
    """One deterministic execution in a workflow plan."""

    steps: tuple[str, ...]
    kind: str
    in_keys: tuple[NestedKey, ...]
    out_keys: tuple[NestedKey, ...]
    model_passes: int
    gradient_mode: GradientMode | None
    autograd_lifetime: AutogradLifetime | None = None
    coexecuted: bool = False


@dataclass(frozen=True)
class CompatibilityDecision:
    """The result of considering one method for an existing model execution."""

    existing_steps: tuple[str, ...]
    candidate_step: str
    compatible: bool
    reason: str


@dataclass(frozen=True)
class WorkflowPlan:
    """An immutable plan created before workflow model execution."""

    executions: tuple[PlannedExecution, ...]
    compatibility: tuple[CompatibilityDecision, ...] = ()

    @property
    def model_passes(self) -> int:
        """Return the total number of declared model passes."""

        return sum(execution.model_passes for execution in self.executions)


@dataclass(frozen=True)
class WorkflowResult:
    """The TensorDict and exact plan produced by one workflow execution.

    Use :meth:`Workflow.run_with_plan` when consumers need execution metadata.
    :meth:`Workflow.run` continues to return only the TensorDict for existing
    callers. ``program`` identifies the imperative operations that wrapped a
    managed :class:`WorkflowSession` run and is ``None`` for direct execution.
    ``occurrence_evidence`` contains only root passes whose selected target
    occurrences were validated before the result was returned.
    """

    data: TensorDictBase
    plan: WorkflowPlan
    program: HookProgram | None = None
    occurrence_evidence: tuple[TargetOccurrenceEvidence, ...] = ()


@dataclass(frozen=True)
class _BoundMethodNode:
    index: int
    name: str
    method: WorkflowMethod
    in_keys: tuple[NestedKey, ...]
    out_keys: tuple[NestedKey, ...]
    model_in_keys: tuple[NestedKey, ...]
    model_out_keys: tuple[NestedKey, ...]
    execution_spec: ExecutionSpec
    program: HookProgram | None
    direct_execution: bool
    allow_output_overwrite: bool = False


@dataclass(frozen=True)
class _OperatorNode:
    index: int
    name: str
    operator: TensorDictModuleBase
    in_keys: tuple[NestedKey, ...]
    out_keys: tuple[NestedKey, ...]
    allow_output_overwrite: bool = False


_ExecutionNode = _BoundMethodNode | _OperatorNode


def _artifact_leaf_items(data: TensorDictBase):
    return (
        (key, value)
        for key, value in data.items(include_nested=True, leaves_only=False)
        if not isinstance(value, TensorDictBase)
    )


def _artifact_leaf_keys(data: TensorDictBase) -> frozenset[NestedKey]:
    return frozenset(key for key, _ in _artifact_leaf_items(data))


@dataclass(frozen=True)
class WorkflowArtifactTensorSpec:
    """Expected metadata for one tensor in a distributed workflow artifact."""

    key: NestedKey
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class WorkflowArtifactSchema:
    """Validate the fixed layout used by TensorDict-native distributed transport.

    Create the same schema on sending and receiving ranks. Leaf order is part of
    the contract because :meth:`TensorDictBase.send` and
    :meth:`TensorDictBase.recv` assign communication tags in traversal order.
    """

    tensors: tuple[WorkflowArtifactTensorSpec, ...]
    batch_size: torch.Size
    device: torch.device | None

    @classmethod
    def from_tensordict(cls, data: TensorDictBase) -> "WorkflowArtifactSchema":
        """Capture keys, shapes, dtypes, and devices from a TensorDict template."""

        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow artifact must be a TensorDict, got {type(data).__name__}")
        tensors = []
        for key, value in _artifact_leaf_items(data):
            if not isinstance(value, Tensor):
                raise WorkflowArtifactError(
                    f"Workflow artifact key {key!r} must contain a Tensor, got {type(value).__name__}"
                )
            tensors.append(WorkflowArtifactTensorSpec(key, value.shape, value.dtype, value.device))
        return cls(tuple(tensors), data.batch_size, data.device)

    @property
    def keys(self) -> tuple[NestedKey, ...]:
        """Return the ordered tensor keys required by the transport."""

        return tuple(tensor.key for tensor in self.tensors)

    def validate(self, data: TensorDictBase) -> None:
        """Fail before use when an artifact differs from this schema."""

        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow artifact must be a TensorDict, got {type(data).__name__}")
        if data.batch_size != self.batch_size:
            raise WorkflowArtifactError(
                f"Workflow artifact batch size is {data.batch_size}, expected {self.batch_size}"
            )
        if data.device != self.device:
            raise WorkflowArtifactError(f"Workflow artifact device is {data.device}, expected {self.device}")

        items = tuple(_artifact_leaf_items(data))
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
    """Send only a validated, consolidated TensorDict through an existing group.

    Process-group lifecycle, workflow execution, models, and hook sessions remain
    local to the caller. Tensor values are transferred with TensorDict's native
    point-to-point transport.
    """

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

        for key, value in _artifact_leaf_items(data):
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

    def working_copy(self, plan: WorkflowPlan) -> TensorDictBase:
        if any(execution.autograd_lifetime is AutogradLifetime.BACKWARD for execution in plan.executions):
            raise WorkflowHandoffError(
                "Workflow handoff artifacts cannot retain an autograd graph for deferred backward execution"
            )
        missing = tuple(
            key for execution in plan.executions for key in execution.out_keys if key != "_" and key not in self.keys
        )
        if missing:
            raise WorkflowHandoffError(
                "Workflow handoff artifact storage requires every output to be preallocated; "
                f"missing output keys: {list(dict.fromkeys(missing))!r}"
            )
        # Clone only the TensorDict containers. Tensor leaves retain their native
        # shared or consolidated storage until a declared output is committed.
        return self.data.clone(recurse=False)

    def commit(self, current: TensorDictBase, working: TensorDictBase, execution: PlannedExecution) -> None:
        if current is not working:
            raise WorkflowHandoffError(
                f"Workflow step {execution.steps!r} returned a different TensorDict; "
                "handoff workflows must mutate their input TensorDict in place"
            )
        if current.batch_size != self.batch_size or current.device != self.device:
            raise WorkflowHandoffError(
                f"Workflow step {execution.steps!r} changed artifact batch size or device metadata"
            )
        if _artifact_leaf_keys(current) != self.keys:
            raise WorkflowHandoffError(f"Workflow step {execution.steps!r} changed artifact keys")

        for key in execution.out_keys:
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


@contextmanager
def _bind_method(
    index: int,
    method: WorkflowMethod,
    model: nn.Module,
    *,
    for_inspection: bool = False,
    allow_output_overwrite: bool = False,
) -> Generator[tuple[_BoundMethodNode, MethodBinding], None, None]:
    context = method.prepare(model)
    if not isinstance(context, HookingContext):
        raise TypeError(
            f"{type(method).__name__}.prepare(model) must return a HookingContext, got {type(context).__name__}"
        )
    binding = context.inspect() if for_inspection else context
    with binding as prepared:
        if not isinstance(prepared, TensorDictModuleBase):
            raise TypeError(
                f"{type(method).__name__}.prepare(model) must bind a TensorDictModuleBase, "
                f"got {type(prepared).__name__}"
            )
        if not isinstance(getattr(prepared, "td_module", None), TensorDictModuleBase) or not callable(
            getattr(prepared, "finalize_tensordict", None)
        ):
            raise TypeError(
                f"{type(method).__name__}.prepare(model) must bind the MethodBinding protocol, "
                f"got {type(prepared).__name__}"
            )
        node = _BoundMethodNode(
            index=index,
            name=_step_name(index, method),
            method=method,
            in_keys=tuple(prepared.in_keys),
            out_keys=tuple(prepared.out_keys),
            model_in_keys=context.model_in_keys,
            model_out_keys=context.model_out_keys,
            execution_spec=_method_spec(method),
            program=context.program,
            direct_execution=context.executes_model_directly,
            allow_output_overwrite=allow_output_overwrite,
        )
        yield node, prepared


def _coexecution_incompatibility(
    existing: Sequence[_BoundMethodNode],
    candidate: _BoundMethodNode,
) -> str | None:
    """Return ``None`` only when bound facts prove one shared call safe."""

    group = (*existing, candidate)
    if any(node.execution_spec.model_passes != 1 for node in group):
        return "co-execution requires every method to declare exactly one model pass"
    if any(node.execution_spec.gradient_mode != group[0].execution_spec.gradient_mode for node in group[1:]):
        return "methods require different autograd modes"
    if any(node.execution_spec.autograd_lifetime != group[0].execution_spec.autograd_lifetime for node in group[1:]):
        return "methods require different autograd lifetimes"
    if any(
        (node.model_in_keys, node.model_out_keys) != (group[0].model_in_keys, group[0].model_out_keys)
        for node in group[1:]
    ):
        return "prepared methods bind different model TensorDict signatures"
    if any(not node.direct_execution for node in group):
        return "a prepared method transforms model execution rather than wrapping it directly"
    if any(node.program is None for node in group):
        return "a prepared method did not expose its bound hook program"
    if any(not node.program.hooks for node in group if node.program is not None):
        return "an empty hook program is not evidence of shared execution compatibility"
    method_outputs: list[NestedKey] = []
    for node in group:
        owned = [key for key in node.out_keys if key not in node.model_out_keys]
        if any(_key_paths_overlap(key, existing) for key in owned for existing in method_outputs):
            return "method-owned output namespaces overlap"
        method_outputs.extend(owned)
    if any(
        spec.operation != "capture" or spec.direction is None
        for node in group
        if node.program is not None
        for spec in node.program.hooks
    ):
        return "only bound read-only capture programs currently prove shared execution safe"
    return None


def _same_bound_facts(planned: _BoundMethodNode, rebound: _BoundMethodNode) -> bool:
    return (
        planned.in_keys,
        planned.out_keys,
        planned.model_in_keys,
        planned.model_out_keys,
        planned.execution_spec,
        planned.program,
        planned.direct_execution,
        planned.allow_output_overwrite,
    ) == (
        rebound.in_keys,
        rebound.out_keys,
        rebound.model_in_keys,
        rebound.model_out_keys,
        rebound.execution_spec,
        rebound.program,
        rebound.direct_execution,
        rebound.allow_output_overwrite,
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
    if (
        node.execution_spec.gradient_mode is not GradientMode.REQUIRED
        and node.program is not None
        and any(spec.direction in {"bwd", "bwd_pre"} for spec in node.program.hooks)
    ):
        raise ValueError(
            f"Workflow step {node.name!r} installs backward hooks but does not own an autograd-enabled execution"
        )


class _DeferredAutogradCleanup:
    """Close prepared contexts after the caller's autograd engine finishes."""

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
    """Compose configured methods and ordinary TensorDict modules.

    Methods are bound temporarily to ``model`` so the planner can use their
    actual TensorDict signatures and installed hook programs. Unknown method
    compatibility always produces separate model executions.

    Execution is rank-local when ``model`` is a ``DistributedDataParallel``
    instance. TDHook does not initialize process groups, launch workers, or
    aggregate the returned plans, programs, TensorDicts, or captured values.
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

    def _inspect(self, model: nn.Module) -> tuple[_ExecutionNode, ...]:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Workflow model must be a torch.nn.Module, got {type(model).__name__}")
        nodes: list[_ExecutionNode] = []
        for index, step in enumerate(self.steps):
            allow_output_overwrite = isinstance(step, WorkflowUpdate)
            actual = step.step if allow_output_overwrite else step
            if isinstance(actual, TensorDictModuleBase):
                nodes.append(
                    _OperatorNode(
                        index=index,
                        name=_step_name(index, step),
                        operator=actual,
                        in_keys=tuple(actual.in_keys),
                        out_keys=tuple(actual.out_keys),
                        allow_output_overwrite=allow_output_overwrite,
                    )
                )
            else:
                with _bind_method(
                    index,
                    actual,
                    model,
                    for_inspection=True,
                    allow_output_overwrite=allow_output_overwrite,
                ) as (node, _):
                    nodes.append(node)
        return tuple(nodes)

    @staticmethod
    def _build_plan(nodes: Sequence[_ExecutionNode]) -> WorkflowPlan:
        executions: list[PlannedExecution] = []
        decisions: list[CompatibilityDecision] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if isinstance(node, _OperatorNode):
                executions.append(
                    PlannedExecution(
                        steps=(node.name,),
                        kind="operator",
                        in_keys=node.in_keys,
                        out_keys=node.out_keys,
                        model_passes=0,
                        gradient_mode=None,
                        autograd_lifetime=None,
                    )
                )
                index += 1
                continue

            group = [node]
            _validate_workflow_method(node)
            while index + len(group) < len(nodes):
                candidate = nodes[index + len(group)]
                if not isinstance(candidate, _BoundMethodNode):
                    break
                _validate_workflow_method(candidate)
                reason = _coexecution_incompatibility(group, candidate)
                decisions.append(
                    CompatibilityDecision(
                        existing_steps=tuple(item.name for item in group),
                        candidate_step=candidate.name,
                        compatible=reason is None,
                        reason="bound read-only capture programs are compatible" if reason is None else reason,
                    )
                )
                if reason is not None:
                    break
                group.append(candidate)

            executions.append(
                PlannedExecution(
                    steps=tuple(item.name for item in group),
                    kind="method",
                    in_keys=tuple(dict.fromkeys(key for item in group for key in item.in_keys)),
                    out_keys=tuple(dict.fromkeys(key for item in group for key in item.out_keys)),
                    model_passes=1 if len(group) > 1 else group[0].execution_spec.model_passes,
                    gradient_mode=group[0].execution_spec.gradient_mode,
                    autograd_lifetime=group[0].execution_spec.autograd_lifetime,
                    coexecuted=len(group) > 1,
                )
            )
            index += len(group)
        for index, execution in enumerate(executions):
            if execution.autograd_lifetime is AutogradLifetime.BACKWARD and any(
                later.kind == "method" for later in executions[index + 1 :]
            ):
                raise ValueError("A deferred-backward workflow execution cannot precede a later model execution")
        return WorkflowPlan(tuple(executions), tuple(decisions))

    def plan(self, model: nn.Module, data: TensorDictBase) -> WorkflowPlan:
        """Bind, validate, and return a plan without executing the model."""

        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow data must be a TensorDict, got {type(data).__name__}")
        nodes = self._inspect(model)
        _validate_dependencies(nodes, data)
        return self._build_plan(nodes)

    def run_with_plan(self, model: nn.Module, data: TensorDictBase) -> WorkflowResult:
        """Execute the workflow and return its TensorDict with the executed plan.

        TensorDicts prepared with :meth:`TensorDict.share_memory_` or
        :meth:`TensorDictBase.consolidate` retain their native storage. Their
        output keys must already exist with the expected tensor metadata, and
        each workflow step must use in-place TensorDict semantics.
        """

        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow data must be a TensorDict, got {type(data).__name__}")
        nodes = self._inspect(model)
        _validate_dependencies(nodes, data)
        plan = self._build_plan(nodes)
        handoff = _HandoffArtifact.inspect(data)
        current = handoff.working_copy(plan) if handoff is not None else data
        working = current
        by_name = {node.name: node for node in nodes}
        for execution in plan.executions:
            execution_nodes = [by_name[name] for name in execution.steps]
            _validate_runtime_dependencies(execution_nodes, current)
            first = execution_nodes[0]
            if isinstance(first, _OperatorNode):
                current = first.operator(current)
                if not isinstance(current, TensorDictBase):
                    raise TypeError(
                        f"Workflow operator {first.name!r} must return a TensorDict, got {type(current).__name__}"
                    )
                if handoff is not None:
                    handoff.commit(current, working, execution)
                continue

            with ExitStack() as stack:
                bound = []
                for node in execution_nodes:
                    assert isinstance(node, _BoundMethodNode)
                    rebound, prepared = stack.enter_context(
                        _bind_method(
                            node.index,
                            node.method,
                            model,
                            allow_output_overwrite=node.allow_output_overwrite,
                        )
                    )
                    if not _same_bound_facts(node, rebound):
                        raise RuntimeError(f"Bound facts for workflow method {node.name!r} changed after planning")
                    bound.append(prepared)
                if len(bound) == 1:
                    result = bound[0](current)
                    if result is not None:
                        current = result
                else:
                    result = bound[0].td_module(current)
                    if result is not None:
                        current = result
                    for prepared in bound:
                        current = prepared.finalize_tensordict(current)
                if not isinstance(current, TensorDictBase):
                    raise TypeError(
                        f"Workflow method execution must return a TensorDict, got {type(current).__name__}"
                    )
                if execution.autograd_lifetime is AutogradLifetime.BACKWARD:
                    cleanup = _DeferredAutogradCleanup(stack.pop_all())
                    for prepared in bound:
                        assert prepared.hooking_context is not None
                        prepared.hooking_context.on_hook_failure(cleanup.close)
                    cleanup.arm(current, first.model_out_keys)
            if handoff is not None:
                handoff.commit(current, working, execution)
        if handoff is not None:
            current = handoff.result()
        return WorkflowResult(data=current, plan=plan)

    def run(self, model: nn.Module, data: TensorDictBase) -> TensorDictBase:
        """Execute the workflow and return its TensorDict.

        Use :meth:`run_with_plan` to also receive the exact execution plan.
        """

        return self.run_with_plan(model, data).data

    def session(self, model: nn.Module) -> "WorkflowSession":
        """Bind this workflow and ``model`` to one managed hook session.

        Operations registered on the returned session wrap the complete
        workflow run without participating in planning or co-execution.
        """

        return WorkflowSession(self, model)

    def __call__(self, model: nn.Module, data: TensorDictBase) -> TensorDictBase:
        return self.run(model, data)


class WorkflowSession(HookSession):
    """A :class:`HookSession` whose operations wrap a complete workflow run."""

    def __init__(self, workflow: Workflow, model: nn.Module):
        super().__init__(model)
        self._workflow = workflow

    def run(self, data: TensorDictBase) -> WorkflowResult:
        """Execute the bound workflow and associate its plan with this program.

        The session must be active. Managed early stopping aborts the workflow
        run and exits the surrounding context, leaving partial results on the
        corresponding :class:`~tdhook.session.EarlyStopResult`.
        """

        self._active_state()
        evidence_start = len(self.occurrence_evidence)
        result = self._workflow.run_with_plan(self._model(), data)
        return WorkflowResult(
            data=result.data,
            plan=result.plan,
            program=self.program,
            occurrence_evidence=self.occurrence_evidence[evidence_start:],
        )

    def __call__(self, data: TensorDictBase) -> WorkflowResult:
        return self.run(data)


__all__ = [
    "CompatibilityDecision",
    "MethodBinding",
    "PlannedExecution",
    "WorkflowArtifactError",
    "WorkflowArtifactSchema",
    "WorkflowArtifactTensorSpec",
    "Workflow",
    "WorkflowHandoffError",
    "WorkflowMethod",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowSession",
    "WorkflowUpdate",
    "receive_workflow_artifact",
    "send_workflow_artifact",
]
