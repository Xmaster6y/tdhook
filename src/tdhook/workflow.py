"""TensorDict-native composition and conservative method execution planning."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Generator, Protocol, Sequence, runtime_checkable

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch import nn

from tdhook._types import UnraveledKey
from tdhook.contexts import HookingContext
from tdhook.execution import ExecutionSpec, GradientMode
from tdhook.runtime import HookProgram


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


WorkflowStep = WorkflowMethod | TensorDictModuleBase | WorkflowUpdate


class MethodBinding(Protocol):
    """Prepared method surface used for execution after facts are validated."""

    in_keys: list[UnraveledKey]
    out_keys: list[UnraveledKey]

    @property
    def td_module(self) -> TensorDictModuleBase: ...

    def __call__(self, data: TensorDictBase) -> TensorDictBase: ...

    def finalize_tensordict(self, data: TensorDictBase) -> TensorDictBase: ...


@dataclass(frozen=True)
class PlannedExecution:
    """One deterministic execution in a workflow plan."""

    steps: tuple[str, ...]
    kind: str
    in_keys: tuple[UnraveledKey, ...]
    out_keys: tuple[UnraveledKey, ...]
    model_passes: int
    gradient_mode: GradientMode | None
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
class _BoundMethodNode:
    index: int
    name: str
    method: WorkflowMethod
    in_keys: tuple[UnraveledKey, ...]
    out_keys: tuple[UnraveledKey, ...]
    model_in_keys: tuple[UnraveledKey, ...]
    model_out_keys: tuple[UnraveledKey, ...]
    execution_spec: ExecutionSpec
    program: HookProgram | None
    direct_execution: bool
    allow_output_overwrite: bool = False


@dataclass(frozen=True)
class _OperatorNode:
    index: int
    name: str
    operator: TensorDictModuleBase
    in_keys: tuple[UnraveledKey, ...]
    out_keys: tuple[UnraveledKey, ...]
    allow_output_overwrite: bool = False


_ExecutionNode = _BoundMethodNode | _OperatorNode


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
    method_outputs: list[UnraveledKey] = []
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


def _available_keys(data: TensorDictBase) -> set[UnraveledKey]:
    return set(data.keys(include_nested=True, leaves_only=False))


def _key_path(key: UnraveledKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _key_paths_overlap(left: UnraveledKey, right: UnraveledKey) -> bool:
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


def _provided_namespace_covers(provided: UnraveledKey, required: UnraveledKey) -> bool:
    provided_path = _key_path(provided)
    required_path = _key_path(required)
    return len(provided_path) < len(required_path) and required_path[: len(provided_path)] == provided_path


def _validate_dependencies(nodes: Sequence[_ExecutionNode], data: TensorDictBase) -> None:
    available = _available_keys(data)
    provided_namespaces: set[UnraveledKey] = set()
    owned_outputs: set[UnraveledKey] = set()
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
                    coexecuted=len(group) > 1,
                )
            )
            index += len(group)
        return WorkflowPlan(tuple(executions), tuple(decisions))

    def plan(self, model: nn.Module, data: TensorDictBase) -> WorkflowPlan:
        """Bind, validate, and return a plan without executing the model."""

        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow data must be a TensorDict, got {type(data).__name__}")
        nodes = self._inspect(model)
        _validate_dependencies(nodes, data)
        return self._build_plan(nodes)

    def run(self, model: nn.Module, data: TensorDictBase) -> TensorDictBase:
        """Execute the validated workflow and return its TensorDict."""

        if not isinstance(data, TensorDictBase):
            raise TypeError(f"Workflow data must be a TensorDict, got {type(data).__name__}")
        nodes = self._inspect(model)
        _validate_dependencies(nodes, data)
        plan = self._build_plan(nodes)
        current = data
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
                raise TypeError(f"Workflow method execution must return a TensorDict, got {type(current).__name__}")
        return current

    def __call__(self, model: nn.Module, data: TensorDictBase) -> TensorDictBase:
        return self.run(model, data)


__all__ = [
    "CompatibilityDecision",
    "MethodBinding",
    "PlannedExecution",
    "Workflow",
    "WorkflowMethod",
    "WorkflowPlan",
    "WorkflowUpdate",
]
