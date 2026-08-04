"""Declarative execution contracts shared by methods and workflows.

The contracts in this module contain no model, hook, or runtime state.  They
describe the TensorDict keys an operation owns and the observable effects that
an execution planner must account for before touching a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

from tensordict.utils import NestedKey


class Access(StrEnum):
    """Access performed on one category of model state."""

    NONE = "none"
    READ = "read"
    WRITE = "write"


class RunKind(StrEnum):
    """Whether an operation executes a model or only transforms a TensorDict."""

    MODEL = "model"
    TRANSFORM = "transform"


class GradientMode(StrEnum):
    """Gradient requirement for a model execution."""

    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True)
class EffectSpec:
    """Structured model and method-state effects used for safe planning.

    These fields describe effects beyond normal TensorDict key publication.
    A write conflicts conservatively with either a read or write in the same
    domain.  More specialised ordered composition can be added later without
    weakening this safe default.
    """

    activations: Access = Access.NONE
    gradients: Access = Access.NONE
    parameters: Access = Access.NONE
    state: Access = Access.NONE

    def __post_init__(self) -> None:
        for domain in ("activations", "gradients", "parameters", "state"):
            if not isinstance(getattr(self, domain), Access):
                raise TypeError(f"{domain} access must be an Access value")

    def conflict_domains(self, other: "EffectSpec") -> tuple[str, ...]:
        """Return effect domains that cannot safely share one execution."""

        conflicts = []
        for domain in ("activations", "gradients", "parameters", "state"):
            first = getattr(self, domain)
            second = getattr(other, domain)
            if Access.WRITE in (first, second) and Access.NONE not in (first, second):
                conflicts.append(domain)
        return tuple(conflicts)


def _path(key: NestedKey) -> tuple[str, ...]:
    if isinstance(key, str):
        if not key:
            raise ValueError("TensorDict keys must not be empty")
        return (key,)
    if not isinstance(key, tuple) or not key or not all(isinstance(part, str) and part for part in key):
        raise TypeError(f"TensorDict keys must be a string or non-empty tuple of strings, got {key!r}")
    return key


def _normalise_keys(keys: Iterable[NestedKey], *, role: str) -> tuple[NestedKey, ...]:
    result = tuple(keys)
    paths = tuple(_path(key) for key in result)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{role} contains duplicate TensorDict keys: {result!r}")
    return result


def _keys_overlap(first: NestedKey, second: NestedKey) -> bool:
    first_path, second_path = _path(first), _path(second)
    return first_path[: len(second_path)] == second_path or second_path[: len(first_path)] == first_path


@dataclass(frozen=True)
class KeyContract:
    """TensorDict inputs, outputs, and explicitly authorised replacements."""

    in_keys: tuple[NestedKey, ...] = ()
    out_keys: tuple[NestedKey, ...] = ()
    overwrite_keys: frozenset[NestedKey] = frozenset()

    def __post_init__(self) -> None:
        in_keys = _normalise_keys(self.in_keys, role="in_keys")
        out_keys = _normalise_keys(self.out_keys, role="out_keys")
        overwrite_keys = frozenset(_normalise_keys(self.overwrite_keys, role="overwrite_keys"))
        object.__setattr__(self, "in_keys", in_keys)
        object.__setattr__(self, "out_keys", out_keys)
        object.__setattr__(self, "overwrite_keys", overwrite_keys)

        unknown = overwrite_keys.difference(out_keys)
        if unknown:
            raise ValueError(f"overwrite_keys must also be declared in out_keys: {tuple(unknown)!r}")

        for index, first in enumerate(out_keys):
            for second in out_keys[index + 1 :]:
                if _keys_overlap(first, second):
                    raise ValueError(f"out_keys contains overlapping TensorDict paths: {first!r} and {second!r}")

        implicit_overwrites = {
            output
            for output in out_keys
            if any(_keys_overlap(output, input_) for input_ in in_keys) and output not in overwrite_keys
        }
        if implicit_overwrites:
            raise ValueError(
                f"outputs overlapping inputs must be declared in overwrite_keys: {tuple(implicit_overwrites)!r}"
            )

    def conflict_keys(self, other: "KeyContract") -> tuple[NestedKey, ...]:
        """Return outputs that conflict with another same-run contract."""

        conflicts = []
        for output in self.out_keys:
            if any(_keys_overlap(output, key) for key in (*other.in_keys, *other.out_keys)):
                conflicts.append(output)
        for output in other.out_keys:
            if any(_keys_overlap(output, key) for key in self.in_keys) and output not in conflicts:
                conflicts.append(output)
        return tuple(conflicts)


@dataclass(frozen=True)
class MethodSpec:
    """Complete declarative contract for one configured operation."""

    identifier: str
    keys: KeyContract = field(default_factory=KeyContract)
    run_kind: RunKind = RunKind.MODEL
    model_passes: int = 1
    gradient_mode: GradientMode = GradientMode.OPTIONAL
    effects: EffectSpec = field(default_factory=EffectSpec)
    coexecution_key: str | None = None
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("MethodSpec identifier must not be empty")
        if not isinstance(self.keys, KeyContract):
            raise TypeError("keys must be a KeyContract")
        if not isinstance(self.run_kind, RunKind):
            raise TypeError("run_kind must be a RunKind")
        if not isinstance(self.gradient_mode, GradientMode):
            raise TypeError("gradient_mode must be a GradientMode")
        if not isinstance(self.effects, EffectSpec):
            raise TypeError("effects must be an EffectSpec")
        if not isinstance(self.constraints, tuple) or not all(
            isinstance(constraint, str) and constraint.strip() for constraint in self.constraints
        ):
            raise TypeError("constraints must be a tuple of non-empty strings")
        if isinstance(self.model_passes, bool) or not isinstance(self.model_passes, int):
            raise TypeError("model_passes must be an integer")
        if self.run_kind is RunKind.MODEL and self.model_passes <= 0:
            raise ValueError("model operations require at least one model pass")
        if self.run_kind is RunKind.TRANSFORM and self.model_passes != 0:
            raise ValueError("TensorDict transforms must declare zero model passes")
        if self.run_kind is RunKind.TRANSFORM and self.gradient_mode is not GradientMode.DISABLED:
            raise ValueError("TensorDict transforms must disable model gradients")
        if self.run_kind is RunKind.TRANSFORM and any(
            access is not Access.NONE
            for access in (self.effects.activations, self.effects.gradients, self.effects.parameters)
        ):
            raise ValueError("TensorDict transforms cannot declare model effects")
        if self.gradient_mode is GradientMode.REQUIRED and self.effects.gradients is Access.NONE:
            raise ValueError("gradient-required methods must declare a gradient effect")
        if self.coexecution_key is not None:
            if not self.coexecution_key.strip():
                raise ValueError("coexecution_key must be non-empty when provided")
            if self.run_kind is not RunKind.MODEL or self.model_passes != 1:
                raise ValueError("only single-pass model operations may declare a coexecution_key")

    def coexecution_incompatibility(self, other: "MethodSpec") -> str | None:
        """Explain why two methods cannot conservatively share one model run."""

        if not self.coexecution_key or self.coexecution_key != other.coexecution_key:
            return "methods do not declare the same non-empty coexecution key"
        if self.run_kind is not RunKind.MODEL or other.run_kind is not RunKind.MODEL:
            return "only model operations can share a model run"
        if self.model_passes != 1 or other.model_passes != 1:
            return "only single-pass methods can share a model run"
        if self.gradient_mode is not other.gradient_mode:
            return "methods require different gradient modes"
        if self.constraints != other.constraints:
            return "methods declare different device, batch, or runtime constraints"
        effect_conflicts = self.effects.conflict_domains(other.effects)
        if effect_conflicts:
            return f"methods have conflicting effects: {', '.join(effect_conflicts)}"
        key_conflicts = self.keys.conflict_keys(other.keys)
        if key_conflicts:
            return f"methods have conflicting TensorDict keys: {key_conflicts!r}"
        return None


__all__ = [
    "Access",
    "EffectSpec",
    "GradientMode",
    "KeyContract",
    "MethodSpec",
    "RunKind",
]
