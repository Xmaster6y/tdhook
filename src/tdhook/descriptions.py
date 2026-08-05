"""Stable, JSON-compatible descriptions of configured workflow steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum

from tdhook._types import UnraveledKey
from tdhook.execution import ExecutionSpec
from tdhook.targets import Target


class FrozenDict(dict[str, object]):
    """A JSON-serializable mapping that cannot be mutated after construction."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Configured step descriptions are immutable")

    __delitem__ = __setitem__ = clear = pop = popitem = setdefault = update = _immutable


@dataclass(frozen=True)
class ConfiguredStepDescription:
    """An immutable description of one configured method or bound workflow step."""

    method_type: str
    parameters: FrozenDict
    execution: FrozenDict
    in_keys: tuple[UnraveledKey, ...] = ()
    out_keys: tuple[UnraveledKey, ...] = ()

    def with_keys(
        self, in_keys: Sequence[UnraveledKey], out_keys: Sequence[UnraveledKey]
    ) -> "ConfiguredStepDescription":
        """Return this description with its bound TensorDict interface."""

        return ConfiguredStepDescription(
            self.method_type, self.parameters, self.execution, tuple(in_keys), tuple(out_keys)
        )

    def to_dict(self) -> FrozenDict:
        """Return a JSON-compatible immutable representation."""

        return FrozenDict(
            method_type=self.method_type,
            parameters=self.parameters,
            execution=self.execution,
            in_keys=_key_list(self.in_keys),
            out_keys=_key_list(self.out_keys),
        )


def configured_step_description(
    method: object,
    parameters: Mapping[str, object],
    execution_spec: ExecutionSpec,
    *,
    callback_identifiers: Mapping[Callable[..., object], str] | None = None,
) -> ConfiguredStepDescription:
    """Build a stable description from a method's configured constructor values."""

    identifiers = callback_identifiers or {}
    return ConfiguredStepDescription(
        method_type=f"{type(method).__module__}.{type(method).__qualname__}",
        parameters=_freeze(parameters, identifiers),
        execution=_freeze(asdict(execution_spec), identifiers),
    )


def _freeze(value: object, identifiers: Mapping[Callable[..., object], str]) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Target):
        return _freeze(value.to_dict(), identifiers)
    if callable(value):
        try:
            identifier = identifiers[value]
        except KeyError as exc:
            name = getattr(value, "__qualname__", type(value).__qualname__)
            raise TypeError(f"Callable {name!r} requires an explicit stable identifier") from exc
        if not isinstance(identifier, str) or not identifier:
            raise TypeError("Callable identifiers must be non-empty strings")
        return FrozenDict(identifier=identifier)
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze(asdict(value), identifiers)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Configured mapping keys must be strings")
        return FrozenDict((key, _freeze(item, identifiers)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, identifiers) for item in value)
    raise TypeError(f"Configured value {type(value).__qualname__} is not JSON-compatible")


def _key_list(keys: Sequence[UnraveledKey]) -> tuple[object, ...]:
    return tuple(list(key) if isinstance(key, tuple) else key for key in keys)


__all__ = ["ConfiguredStepDescription", "FrozenDict", "configured_step_description"]
