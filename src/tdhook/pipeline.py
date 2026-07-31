"""A small, linear API for TensorDict interpretability workflows.

Pipelines make the data exchanged between independently executed methods
explicit.  They deliberately do not schedule a graph or convert artifacts:
each stage receives the same :class:`~tensordict.TensorDict` and declares the
keys it reads and writes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from torch import nn
from tensordict import TensorDictBase

from tdhook._types import UnraveledKey
from tdhook.artifacts import ArtifactContract, ArtifactProvenance, make_provenance
from tdhook.contexts import HookingContextFactory


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
    ) -> None:
        if not name:
            raise ValueError("A stage must have a non-empty name")
        self.name = name
        if artifact_contract is not None and (tuple(required_keys) or tuple(provided_keys)):
            raise ValueError("Use either artifact_contract or storage keys, not both")
        self.artifact_contract = artifact_contract
        self.required_keys = _keys(artifact_contract.required_keys if artifact_contract else required_keys)
        self.provided_keys = _keys(artifact_contract.provided_keys if artifact_contract else provided_keys)
        self.effects = frozenset(effects)
        self.incompatible_effects = frozenset(incompatible_effects)

    @abstractmethod
    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        """Execute the stage and return its TensorDict artifacts."""


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
    ) -> None:
        super().__init__(
            name,
            required_keys=required_keys,
            provided_keys=provided_keys,
            effects=("model_execution", *effects),
            incompatible_effects=incompatible_effects,
            artifact_contract=artifact_contract,
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
    ) -> None:
        super().__init__(
            name,
            required_keys=required_keys,
            provided_keys=provided_keys,
            effects=effects,
            incompatible_effects=incompatible_effects,
            artifact_contract=artifact_contract,
        )
        self.transform = transform

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        result = self.transform(artifacts)
        if not isinstance(result, TensorDictBase):
            raise TypeError(f"Transform stage {self.name!r} must return a TensorDict, got {type(result).__name__}")
        return result


class Pipeline:
    """Validate and execute an ordered sequence of stages."""

    def __init__(self, stages: Sequence[Stage], *, reserved_keys: Iterable[PipelineKey] = RESERVED_KEYS) -> None:
        self.stages = tuple(stages)
        self.reserved_keys = frozenset(_keys(reserved_keys))
        self._validate_static()

    def _validate_static(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline stage names must be unique")
        produced: dict[PipelineKey, str] = {}
        effects: dict[str, str] = {}
        incompatible_effects: dict[str, str] = {}
        for stage in self.stages:
            incompatible = stage.incompatible_effects.intersection(effects)
            if incompatible:
                effect = sorted(incompatible)[0]
                raise ValueError(
                    f"Stage {stage.name!r} is incompatible with effect {effect!r} from {effects[effect]!r}"
                )
            reverse_incompatible = stage.effects.intersection(incompatible_effects)
            if reverse_incompatible:
                effect = sorted(reverse_incompatible)[0]
                raise ValueError(
                    f"Stage {stage.name!r} conflicts with incompatible effect {effect!r} from "
                    f"{incompatible_effects[effect]!r}"
                )
            effects.update({effect: stage.name for effect in stage.effects})
            incompatible_effects.update({effect: stage.name for effect in stage.incompatible_effects})
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
                key for key in stage.provided_keys if any(_keys_conflict(key, existing) for existing in available)
            ]
            if collisions:
                raise ValueError(f"Stage {stage.name!r} writes existing artifact keys: {collisions!r}")
            available.update(stage.provided_keys)

    def run(
        self,
        model: nn.Module,
        artifacts: TensorDictBase,
        *,
        model_id: str | None = None,
        seed: int | None = None,
        stage_configurations: Mapping[str, Mapping[str, object]] | None = None,
    ) -> PipelineResult:
        self.validate(artifacts)
        stage_results: list[StageResult] = []
        provenance: list[ArtifactProvenance] = []
        current = artifacts
        for stage in self.stages:
            missing = [key for key in stage.required_keys if key not in self._artifact_keys(current)]
            if missing:
                raise ValueError(f"Stage {stage.name!r} requires missing artifact keys: {missing!r}")
            try:
                current = stage.run(model, current)
            except Exception as error:
                raise RuntimeError(f"Pipeline stage {stage.name!r} failed: {error}") from error
            if not isinstance(current, TensorDictBase):
                raise TypeError(f"Stage {stage.name!r} returned {type(current).__name__}, not a TensorDict")
            missing = [key for key in stage.provided_keys if key not in self._artifact_keys(current)]
            if missing:
                raise ValueError(f"Stage {stage.name!r} did not provide declared artifact keys: {missing!r}")
            stage_results.append(StageResult(stage.name, stage.provided_keys, stage.effects))
            provenance.append(
                make_provenance(
                    stage=stage.name,
                    method=type(stage).__name__,
                    configuration=(stage_configurations or {}).get(stage.name),
                    model_id=model_id,
                    seed=seed,
                    parents=stage.required_keys,
                    model=model,
                )
            )
        return PipelineResult(current, tuple(stage_results), tuple(provenance))
