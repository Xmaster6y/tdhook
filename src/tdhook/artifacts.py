"""Stable, migration-friendly contracts for pipeline artifacts.

This module deliberately describes *what* a method exchanges separately from
the TensorDict key where a current implementation stores it.  New pipeline
code should use the public namespaces below; adapters keep legacy methods
usable while their internals are migrated incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable, Mapping
from warnings import warn

from torch import nn
from tensordict import TensorDict, TensorDictBase

from tdhook._types import UnraveledKey


ArtifactKey = UnraveledKey
PUBLIC_NAMESPACES = frozenset(
    {
        "inputs",
        "outputs",
        "activations",
        "gradients",
        "attributions",
        "probes",
        "interventions",
        "metrics",
    }
)
PRIVATE_NAMESPACE = "_private"


def _path(key: ArtifactKey) -> tuple[str, ...]:
    if isinstance(key, str):
        return (key,)
    if not isinstance(key, tuple) or not key or not all(isinstance(part, str) and part for part in key):
        raise TypeError(f"Artifact keys must be strings or non-empty tuples of strings, got {key!r}")
    return key


def is_private_key(key: ArtifactKey) -> bool:
    """Return whether *key* is owned by a method implementation."""
    return _path(key)[0] == PRIVATE_NAMESPACE


def validate_artifact_key(key: ArtifactKey, *, public: bool = True) -> ArtifactKey:
    """Validate a stable public key, or a private implementation key.

    Public artifacts always start with a documented namespace.  Private
    values must be nested below ``("_private", method, ...)`` so that they
    cannot accidentally become a cross-stage dependency.
    """
    path = _path(key)
    if public and path[0] not in PUBLIC_NAMESPACES:
        raise ValueError(f"Public artifact key {key!r} must start with one of {sorted(PUBLIC_NAMESPACES)!r}")
    if not public and (path[0] != PRIVATE_NAMESPACE or len(path) < 2):
        raise ValueError("Private artifact keys must be nested below ('_private', <method>, ...)")
    return key


def _named_keys(keys: Mapping[str, ArtifactKey], *, role: str) -> dict[str, ArtifactKey]:
    result = dict(keys)
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{role} artifact names must be non-empty strings")
    for key in result.values():
        validate_artifact_key(key)
    if len(set(result.values())) != len(result):
        raise ValueError(f"{role} artifact keys must be unique")
    return result


@dataclass(frozen=True)
class ArtifactContract:
    """Named public requirements and products of one pipeline stage.

    Names are method-facing (for example ``"source"`` or ``"scores"``),
    while keys are storage-facing.  This lets a stage depend on a contract
    without exposing a legacy cache layout as its API.
    """

    requires: Mapping[str, ArtifactKey] = field(default_factory=dict)
    provides: Mapping[str, ArtifactKey] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", _named_keys(self.requires, role="Required"))
        object.__setattr__(self, "provides", _named_keys(self.provides, role="Provided"))

    @property
    def required_keys(self) -> tuple[ArtifactKey, ...]:
        return tuple(self.requires.values())

    @property
    def provided_keys(self) -> tuple[ArtifactKey, ...]:
        return tuple(self.provides.values())


@dataclass(frozen=True)
class ArtifactAdapter:
    """Map a public contract onto the keys used by an existing method.

    ``storage`` maps each contract name to the method's current TensorDict
    key. :meth:`prepare` copies public requirements into that legacy storage;
    :meth:`finalize` copies declared products back to the public contract.
    These methods provide the runtime bridge used by ``AdapterStage`` while
    leaving standalone methods unchanged.
    """

    method: str
    contract: ArtifactContract
    storage: Mapping[str, ArtifactKey]

    def __post_init__(self) -> None:
        names = set(self.contract.requires) | set(self.contract.provides)
        if not self.method:
            raise ValueError("An artifact adapter needs a method identifier")
        if set(self.storage) != names:
            raise ValueError("Adapter storage keys must exactly match its contract names")
        for key in self.storage.values():
            _path(key)

    def prepare(self, artifacts: TensorDictBase, storage: TensorDictBase | None = None) -> TensorDictBase:
        """Populate legacy storage with this adapter's public requirements."""
        storage = TensorDict() if storage is None else storage
        for name, public_key in self.contract.requires.items():
            storage.set(self.storage[name], artifacts.get(public_key))
        return storage

    def finalize(self, artifacts: TensorDictBase, storage: TensorDictBase) -> TensorDictBase:
        """Publish declared legacy products under their stable public keys."""
        for name, public_key in self.contract.provides.items():
            legacy_key = self.storage[name]
            if legacy_key not in storage.keys(include_nested=True, leaves_only=True):
                raise ValueError(f"Legacy method {self.method!r} did not provide {legacy_key!r}")
            value = storage.get(legacy_key)
            artifacts.set(public_key, value)
        return artifacts


def activation_caching_adapter(cache_key: ArtifactKey = "cache") -> ArtifactAdapter:
    return ArtifactAdapter(
        "activation-caching",
        ArtifactContract(provides={"activations": ("activations", "cache")}),
        {"activations": cache_key},
    )


def probing_adapter(result_key: ArtifactKey = "probes") -> ArtifactAdapter:
    return ArtifactAdapter(
        "probing", ArtifactContract(provides={"results": ("probes", "results")}), {"results": result_key}
    )


def attribution_adapter(result_key: ArtifactKey = "attr") -> ArtifactAdapter:
    return ArtifactAdapter(
        "attribution",
        ArtifactContract(provides={"attributions": ("attributions", "values")}),
        {"attributions": result_key},
    )


def weight_adapter(output_key: ArtifactKey = "output", *, cache_key: ArtifactKey | None = None) -> ArtifactAdapter:
    """Adapt a weight intervention pass's real model output.

    ``cache_key`` is retained as a deprecated alias for callers using the
    original helper signature.
    """
    if cache_key is not None:
        if output_key != "output":
            raise TypeError("weight_adapter() cannot receive both output_key and cache_key")
        warn("cache_key is deprecated; use output_key instead", DeprecationWarning, stacklevel=2)
        output_key = cache_key
    return ArtifactAdapter(
        "weight-adapters",
        ArtifactContract(provides={"output": ("outputs", "model")}),
        {"output": output_key},
    )


@dataclass(frozen=True)
class ArtifactProvenance:
    """Lightweight, in-memory provenance returned with :class:`PipelineResult`."""

    stage: str
    method: str
    configuration: Mapping[str, object]
    package_version: str
    model_id: str | None
    device: str | None
    dtype: str | None
    seed: int | None
    parents: tuple[ArtifactKey, ...] = ()


def make_provenance(
    *,
    stage: str,
    method: str,
    configuration: Mapping[str, object] | None = None,
    model_id: str | None = None,
    seed: int | None = None,
    parents: Iterable[ArtifactKey] = (),
    model: nn.Module | None = None,
) -> ArtifactProvenance:
    """Build serialisation-safe metadata without retaining a model reference."""
    try:
        package_version = version("tdhook")
    except PackageNotFoundError:
        package_version = "unknown"
    tensor = None
    if model is not None:
        tensor = next(iter(model.parameters()), None)
        if tensor is None:
            tensor = next(iter(model.buffers()), None)
    return ArtifactProvenance(
        stage=stage,
        method=method,
        configuration=dict(configuration or {}),
        package_version=package_version,
        model_id=model_id,
        device=None if tensor is None else str(tensor.device),
        dtype=None if tensor is None else str(tensor.dtype),
        seed=seed,
        parents=tuple(parents),
    )


class ArtifactRegistry:
    """Track ownership within one artifact exchange and reject stale reuse."""

    def __init__(self) -> None:
        self._generation = 0
        self._records: dict[ArtifactKey, tuple[str, int]] = {}

    def begin_generation(self) -> int:
        self._generation += 1
        return self._generation

    def claim(self, key: ArtifactKey, owner: str, *, generation: int | None = None) -> None:
        generation = self._generation if generation is None else generation
        existing = self._records.get(key)
        if existing is not None and existing[0] != owner:
            raise ValueError(f"Artifact key {key!r} is already owned by {existing[0]!r}")
        self._records[key] = (owner, generation)

    def require_fresh(self, key: ArtifactKey, *, generation: int | None = None) -> None:
        generation = self._generation if generation is None else generation
        if key not in self._records:
            raise ValueError(f"Artifact key {key!r} has not been registered")
        owner, recorded_generation = self._records[key]
        if recorded_generation != generation:
            raise ValueError(
                f"Artifact key {key!r} from {owner!r} is stale (generation {recorded_generation}, expected {generation})"
            )
