"""A small, linear API for TensorDict interpretability workflows.

Pipelines make the data exchanged between independently executed methods
explicit.  They deliberately do not schedule a graph or convert artifacts:
each stage receives the same :class:`~tensordict.TensorDict` and declares the
keys it reads and writes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Sequence

from torch import nn
from tensordict import TensorDictBase

from tdhook._types import UnraveledKey
from tdhook.artifacts import ArtifactAdapter, ArtifactContract, ArtifactProvenance, ArtifactRegistry, make_provenance
from tdhook.contexts import HookGroup, HookingContextFactory


PipelineKey = UnraveledKey
RESERVED_KEYS = frozenset({"_pipeline"})


def _keys(keys: Iterable[PipelineKey]) -> tuple[PipelineKey, ...]:
    result = tuple(keys)
    for key in result:
        if not isinstance(key, UnraveledKey):
            raise TypeError(f"Pipeline keys must be strings or non-empty tuples of strings, got {key!r}")
    if len(set(result)) != len(result):
        raise ValueError(f"Stage declares duplicate keys: {result!r}")
    return result


def _path(key: PipelineKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _keys_conflict(first: PipelineKey, second: PipelineKey) -> bool:
    """Whether two TensorDict keys are identical or ancestor/descendant paths."""
    first_path, second_path = _path(first), _path(second)
    return first_path[: len(second_path)] == second_path or second_path[: len(first_path)] == first_path


@dataclass(frozen=True)
class StageResult:
    """Metadata recorded for one completed stage."""

    name: str
    provided_keys: tuple[PipelineKey, ...]
    effects: frozenset[str]


@dataclass(frozen=True)
class PipelineResult:
    """Final artifacts and metadata from a pipeline execution."""

    artifacts: TensorDictBase
    stages: tuple[StageResult, ...]
    provenance: tuple[ArtifactProvenance, ...] = ()
    plan: ExecutionPlan | None = None


@dataclass(frozen=True)
class PlannedRun:
    """One inspectable execution unit produced by pipeline preflight."""

    stages: tuple[str, ...]
    kind: Literal["model", "transform"]
    model_passes: int
    gradient_mode: str
    device_batch_constraints: tuple[str, ...]
    effects: frozenset[str]
    required_keys: tuple[PipelineKey, ...]
    provided_keys: tuple[PipelineKey, ...]
    coalesced: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    """A deterministic plan created before hooks or model execution."""

    runs: tuple[PlannedRun, ...]

    @property
    def model_passes(self) -> int:
        """Total model-pass budget declared by the planned runs."""
        return sum(run.model_passes for run in self.runs)


class Stage(ABC):
    """One ordered pipeline operation with an explicit artifact contract."""

    def __init__(
        self,
        name: str,
        *,
        required_keys: Iterable[PipelineKey] = (),
        provided_keys: Iterable[PipelineKey] = (),
        effects: Iterable[str] = (),
        incompatible_effects: Iterable[str] = (),
        artifact_contract: ArtifactContract | None = None,
        method_id: str | None = None,
        model_passes: int | None = None,
        gradient_mode: str = "optional",
        device_batch_constraints: Iterable[str] = (),
        coexecution_key: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("A stage must have a non-empty name")
        self.name = name
        if artifact_contract is not None and (tuple(required_keys) or tuple(provided_keys)):
            raise ValueError("Use either artifact_contract or storage keys, not both")
        self.artifact_contract = artifact_contract
        self.method_id = type(self).__name__ if method_id is None else method_id
        if not self.method_id:
            raise ValueError("A stage method identifier must be non-empty")
        self.required_keys = _keys(artifact_contract.required_keys if artifact_contract else required_keys)
        self.provided_keys = _keys(artifact_contract.provided_keys if artifact_contract else provided_keys)
        self.effects = frozenset(effects)
        self.incompatible_effects = frozenset(incompatible_effects)
        inferred_passes = 1 if "model_execution" in self.effects else 0
        self.model_passes = inferred_passes if model_passes is None else model_passes
        if self.model_passes < 0:
            raise ValueError("Stage model_passes must be non-negative")
        if "model_execution" in self.effects and self.model_passes == 0:
            raise ValueError("Model-executing stages require positive model_passes")
        self.gradient_mode = gradient_mode
        self.device_batch_constraints = tuple(device_batch_constraints)
        self.coexecution_key = coexecution_key

    @abstractmethod
    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        """Execute the stage and return its TensorDict artifacts."""

    def coexecution_factory(self) -> HookingContextFactory | None:
        """Return the hook factory used for an explicitly compatible shared run."""
        return None

    def coexecution_storage_kind(self) -> str | None:
        """Identify the runtime storage layout used by a shared run."""
        return None

    def coexecution_bindings(self) -> Mapping[PipelineKey, PipelineKey]:
        """Map runtime input keys to their public artifact keys."""
        return {}

    def coexecution_model_keys(
        self,
    ) -> tuple[tuple[PipelineKey, ...] | None, tuple[PipelineKey, ...] | None]:
        """Return the model signature used by a shared run."""
        return None, None

    def prepare_coexecution(
        self, artifacts: TensorDictBase, execution: TensorDictBase | None = None
    ) -> TensorDictBase:
        """Prepare or extend the TensorDict passed to a shared model run."""
        raise TypeError(f"Stage {self.name!r} does not support shared execution")

    def finalize_coexecution(self, artifacts: TensorDictBase, execution: TensorDictBase) -> TensorDictBase:
        """Publish this stage's products after a shared model run."""
        return artifacts


class MethodStage(Stage):
    """Execute a model once under an existing :class:`HookingContextFactory`."""

    def __init__(
        self,
        name: str,
        factory: HookingContextFactory,
        *,
        required_keys: Iterable[PipelineKey] = (),
        provided_keys: Iterable[PipelineKey] = (),
        effects: Iterable[str] = (),
        incompatible_effects: Iterable[str] = (),
        model_in_keys: Iterable[PipelineKey] | None = None,
        model_out_keys: Iterable[PipelineKey] | None = None,
        artifact_contract: ArtifactContract | None = None,
        method_id: str | None = None,
        model_passes: int = 1,
        gradient_mode: str = "optional",
        device_batch_constraints: Iterable[str] = (),
        coexecution_key: str | None = None,
    ) -> None:
        super().__init__(
            name,
            required_keys=required_keys,
            provided_keys=provided_keys,
            effects=("model_execution", *effects),
            incompatible_effects=incompatible_effects,
            artifact_contract=artifact_contract,
            method_id=type(factory).__name__ if method_id is None else method_id,
            model_passes=model_passes,
            gradient_mode=gradient_mode,
            device_batch_constraints=device_batch_constraints,
            coexecution_key=coexecution_key,
        )
        self.factory = factory
        self.model_in_keys = None if model_in_keys is None else _keys(model_in_keys)
        self.model_out_keys = None if model_out_keys is None else _keys(model_out_keys)

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        with self.factory.prepare(
            model,
            in_keys=None if self.model_in_keys is None else list(self.model_in_keys),
            out_keys=None if self.model_out_keys is None else list(self.model_out_keys),
        ) as method:
            result = method(artifacts)
        return artifacts if result is None else result

    def coexecution_factory(self) -> HookingContextFactory | None:
        if type(self).run is not MethodStage.run:
            return None
        return self.factory

    def coexecution_storage_kind(self) -> str:
        return "public-artifacts"

    def coexecution_bindings(self) -> Mapping[PipelineKey, PipelineKey]:
        return {key: key for key in self.required_keys}

    def coexecution_model_keys(
        self,
    ) -> tuple[tuple[PipelineKey, ...] | None, tuple[PipelineKey, ...] | None]:
        return self.model_in_keys, self.model_out_keys

    def prepare_coexecution(
        self, artifacts: TensorDictBase, execution: TensorDictBase | None = None
    ) -> TensorDictBase:
        if execution is not None and execution is not artifacts:
            raise ValueError("MethodStage cannot share a legacy adapter's execution storage")
        return artifacts

    def finalize_coexecution(self, artifacts: TensorDictBase, execution: TensorDictBase) -> TensorDictBase:
        return execution


class TransformStage(Stage):
    """Apply a pure TensorDict-to-TensorDict transformation."""

    def __init__(
        self,
        name: str,
        transform: Callable[[TensorDictBase], TensorDictBase],
        *,
        required_keys: Iterable[PipelineKey] = (),
        provided_keys: Iterable[PipelineKey] = (),
        effects: Iterable[str] = (),
        incompatible_effects: Iterable[str] = (),
        artifact_contract: ArtifactContract | None = None,
        method_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            required_keys=required_keys,
            provided_keys=provided_keys,
            effects=effects,
            incompatible_effects=incompatible_effects,
            artifact_contract=artifact_contract,
            method_id=_callable_identity(transform) if method_id is None else method_id,
        )
        self.transform = transform

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        result = self.transform(artifacts)
        if not isinstance(result, TensorDictBase):
            raise TypeError(f"Transform stage {self.name!r} must return a TensorDict, got {type(result).__name__}")
        return result


def _callable_identity(transform: Callable[..., object]) -> str:
    """Return a useful identifier for provenance."""
    module = getattr(transform, "__module__", type(transform).__module__)
    name = getattr(transform, "__qualname__", type(transform).__qualname__)
    return f"{module}.{name}"


class AdapterStage(Stage):
    """Run a legacy method against adapter-managed storage.

    ``execute`` receives the model, public artifacts, and a TensorDict using
    the legacy method's keys. It may return replacement public artifacts or
    ``None`` after mutating them in place.
    """

    def __init__(
        self,
        name: str,
        adapter: ArtifactAdapter,
        execute: Callable[[nn.Module, TensorDictBase, TensorDictBase], TensorDictBase | None],
        *,
        effects: Iterable[str] = (),
        incompatible_effects: Iterable[str] = (),
    ) -> None:
        super().__init__(
            name,
            artifact_contract=adapter.contract,
            effects=effects,
            incompatible_effects=incompatible_effects,
            method_id=adapter.method,
        )
        self.adapter = adapter
        self.execute = execute

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        storage = self.adapter.prepare(artifacts)
        result = self.execute(model, artifacts, storage)
        return self.adapter.finalize(artifacts if result is None else result, storage)


class Pipeline:
    """Validate and execute an ordered sequence of stages."""

    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        reserved_keys: Iterable[PipelineKey] = RESERVED_KEYS,
        artifact_registry: ArtifactRegistry | None = None,
    ) -> None:
        self.stages = tuple(stages)
        self.reserved_keys = frozenset(_keys(reserved_keys))
        self.artifact_registry = artifact_registry
        self._validate_static()

    def _validate_static(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline stage names must be unique")
        produced: dict[PipelineKey, str] = {}
        for stage in self.stages:
            for key in stage.provided_keys:
                if any(_keys_conflict(key, reserved_key) for reserved_key in self.reserved_keys):
                    raise ValueError(f"Stage {stage.name!r} writes reserved pipeline key {key!r}")
                conflicting_key = next((output for output in produced if _keys_conflict(key, output)), None)
                if conflicting_key is not None:
                    raise ValueError(
                        f"Stage {stage.name!r} duplicates output key {key!r} already provided by "
                        f"{produced[conflicting_key]!r}"
                    )
                produced[key] = stage.name

    @staticmethod
    def _artifact_keys(artifacts: TensorDictBase) -> set[PipelineKey]:
        return set(artifacts.keys(include_nested=True, leaves_only=False))

    @staticmethod
    def _writes_existing_artifact(key: PipelineKey, existing: PipelineKey, artifacts: TensorDictBase) -> bool:
        """Return whether writing *key* replaces an existing value rather than extending a namespace."""
        if not _keys_conflict(key, existing):
            return False
        if key == existing:
            return True
        key_path, existing_path = _path(key), _path(existing)
        if key_path[: len(existing_path)] == existing_path:
            return not isinstance(artifacts.get(existing), TensorDictBase)
        return True

    def validate(self, artifacts: TensorDictBase) -> None:
        """Fail before model execution when a stage dependency is unavailable."""
        if not isinstance(artifacts, TensorDictBase):
            raise TypeError(f"Pipeline artifacts must be a TensorDict, got {type(artifacts).__name__}")
        available = self._artifact_keys(artifacts)
        for stage in self.stages:
            missing = [key for key in stage.required_keys if key not in available]
            if missing:
                raise ValueError(f"Stage {stage.name!r} requires missing artifact keys: {missing!r}")
            collisions = [
                key
                for key in stage.provided_keys
                if any(self._writes_existing_artifact(key, existing, artifacts) for existing in available)
            ]
            if collisions:
                raise ValueError(f"Stage {stage.name!r} writes existing artifact keys: {collisions!r}")
            available.update(stage.provided_keys)

    @staticmethod
    def _can_coalesce(stages: Sequence[Stage]) -> bool:
        """Return whether explicit capabilities prove one shared model run safe."""
        if not stages or any(stage.coexecution_factory() is None for stage in stages):
            return False
        first = stages[0]
        if not first.coexecution_key or any(stage.coexecution_key != first.coexecution_key for stage in stages):
            return False
        if any(stage.model_passes != 1 for stage in stages):
            return False
        if any(stage.coexecution_model_keys() != first.coexecution_model_keys() for stage in stages[1:]):
            return False
        if any(stage.coexecution_storage_kind() != first.coexecution_storage_kind() for stage in stages[1:]):
            return False
        if any(
            (stage.gradient_mode, stage.device_batch_constraints)
            != (first.gradient_mode, first.device_batch_constraints)
            for stage in stages[1:]
        ):
            return False
        produced: set[PipelineKey] = set()
        effects: set[str] = set()
        incompatible: set[str] = set()
        runtime_bindings: dict[PipelineKey, PipelineKey] = {}
        for stage in stages:
            if produced.intersection(stage.required_keys):
                return False
            if stage.incompatible_effects.intersection(effects) or stage.effects.intersection(incompatible):
                return False
            produced.update(stage.provided_keys)
            effects.update(stage.effects)
            incompatible.update(stage.incompatible_effects)
            for runtime_key, public_key in stage.coexecution_bindings().items():
                if runtime_key in runtime_bindings and runtime_bindings[runtime_key] != public_key:
                    return False
                runtime_bindings[runtime_key] = public_key
            factory = stage.coexecution_factory()
            if factory is None or factory.planner_coexecution_incompatibility() is not None:
                return False
        try:
            HookGroup(*(stage.coexecution_factory() for stage in stages))
        except ValueError:
            return False
        return True

    @staticmethod
    def _planned_run(stages: Sequence[Stage], *, coalesced: bool = False) -> PlannedRun:
        kind = "model" if any(stage.model_passes for stage in stages) else "transform"
        gradient_modes = {stage.gradient_mode for stage in stages}
        gradient_mode = next(iter(gradient_modes)) if len(gradient_modes) == 1 else "mixed"
        return PlannedRun(
            stages=tuple(stage.name for stage in stages),
            kind=kind,
            model_passes=1 if coalesced else sum(stage.model_passes for stage in stages),
            gradient_mode=gradient_mode,
            device_batch_constraints=tuple(
                dict.fromkeys(constraint for stage in stages for constraint in stage.device_batch_constraints)
            ),
            effects=frozenset(effect for stage in stages for effect in stage.effects),
            required_keys=tuple(dict.fromkeys(key for stage in stages for key in stage.required_keys)),
            provided_keys=tuple(dict.fromkeys(key for stage in stages for key in stage.provided_keys)),
            coalesced=coalesced,
        )

    def plan(self, artifacts: TensorDictBase | None = None) -> ExecutionPlan:
        """Return the conservative execution plan without preparing the model.

        Unknown method pairs are split. Adjacent :class:`MethodStage` objects
        share a run only when they declare the same non-empty
        ``coexecution_key`` and their remaining execution contracts agree.
        """
        if artifacts is not None:
            self.validate(artifacts)
        runs: list[PlannedRun] = []
        index = 0
        while index < len(self.stages):
            candidate = [self.stages[index]]
            if candidate[0].coexecution_factory() is not None and candidate[0].coexecution_key:
                while index + len(candidate) < len(self.stages):
                    next_stage = self.stages[index + len(candidate)]
                    if not self._can_coalesce([*candidate, next_stage]):
                        break
                    candidate.append(next_stage)
            coalesced = len(candidate) > 1
            runs.append(self._planned_run(candidate, coalesced=coalesced))
            index += len(candidate)
        return ExecutionPlan(tuple(runs))

    @staticmethod
    def _run_coalesced(stages: Sequence[Stage], model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        first = stages[0]
        execution = None
        for stage in stages:
            execution = stage.prepare_coexecution(artifacts, execution)
        factory = HookGroup(*(stage.coexecution_factory() for stage in stages))
        model_in_keys, model_out_keys = first.coexecution_model_keys()
        with factory.prepare(
            model,
            in_keys=None if model_in_keys is None else list(model_in_keys),
            out_keys=None if model_out_keys is None else list(model_out_keys),
        ) as method:
            result = method(execution)
        execution = execution if result is None else result
        current = artifacts
        for stage in stages:
            current = stage.finalize_coexecution(current, execution)
        return current

    def run(
        self,
        model: nn.Module,
        artifacts: TensorDictBase,
        *,
        model_id: str | None = None,
        seed: int | None = None,
        stage_configurations: Mapping[str, Mapping[str, object]] | None = None,
    ) -> PipelineResult:
        plan = self.plan(artifacts)
        registry = self.artifact_registry or ArtifactRegistry()
        generation = registry.begin_generation()
        # Only declared pipeline inputs are owned by the caller for this
        # execution. Retained but undeclared artifacts can belong to another
        # pipeline sharing this registry and must not cause an ownership clash.
        initial_required_keys = {
            key for stage in self.stages for key in stage.required_keys if key in self._artifact_keys(artifacts)
        }
        for key in initial_required_keys:
            registry.claim(key, "<pipeline-input>", generation=generation)
        stage_results: list[StageResult] = []
        provenance: list[ArtifactProvenance] = []
        current = artifacts
        stage_by_name = {stage.name: stage for stage in self.stages}
        for planned_run in plan.runs:
            run_stages = [stage_by_name[name] for name in planned_run.stages]
            for stage in run_stages:
                missing = [key for key in stage.required_keys if key not in self._artifact_keys(current)]
                if missing:
                    raise ValueError(f"Stage {stage.name!r} requires missing artifact keys: {missing!r}")
                for key in stage.required_keys:
                    registry.require_fresh(key, generation=generation)
            try:
                if planned_run.coalesced:
                    current = self._run_coalesced(run_stages, model, current)
                else:
                    current = run_stages[0].run(model, current)
            except Exception as error:
                names = ", ".join(repr(stage.name) for stage in run_stages)
                raise RuntimeError(f"Pipeline planned run [{names}] failed: {error}") from error
            if not isinstance(current, TensorDictBase):
                names = ", ".join(repr(stage.name) for stage in run_stages)
                raise TypeError(f"Planned run [{names}] returned {type(current).__name__}, not a TensorDict")
            for stage in run_stages:
                missing = [key for key in stage.provided_keys if key not in self._artifact_keys(current)]
                if missing:
                    raise ValueError(f"Stage {stage.name!r} did not provide declared artifact keys: {missing!r}")
                for key in stage.provided_keys:
                    registry.claim(key, stage.name, generation=generation)
                    registry.require_fresh(key, generation=generation)
                stage_results.append(StageResult(stage.name, stage.provided_keys, stage.effects))
                provenance.append(
                    make_provenance(
                        stage=stage.name,
                        method=stage.method_id,
                        configuration=(stage_configurations or {}).get(stage.name),
                        model_id=model_id,
                        seed=seed,
                        parents=stage.required_keys,
                        model=model,
                    )
                )
        return PipelineResult(current, tuple(stage_results), tuple(provenance), plan)
