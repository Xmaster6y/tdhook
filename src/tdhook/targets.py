"""Serializable, reusable selectors for model activations, gradients, and parameters."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import copy
from dataclasses import asdict, dataclass
import json
from typing import Literal

from tensordict import TensorDictBase
from torch import Tensor, nn

from tdhook.paths import resolve_submodule_path


TargetKind = Literal["activation", "gradient", "parameter"]
OutputPathComponent = int | str


@dataclass(frozen=True)
class Target:
    """A serializable selection within a module's output, gradient, or parameter.

    ``feature_axis`` identifies the axis containing units, channels, rows, or
    columns.  For example, use ``-1`` for MLP output units, ``1`` for CNN
    output channels, and ``0``/``1`` for parameter rows/columns respectively.
    ``occurrence`` optionally selects one zero-based activation or gradient
    observation per root-model execution when a module is called repeatedly.
    """

    module_path: str
    kind: TargetKind
    feature_axis: int
    indices: tuple[int, ...]
    parameter: str | None = None
    output_path: tuple[OutputPathComponent, ...] = ()
    occurrence: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"activation", "gradient", "parameter"}:
            raise ValueError(f"Invalid target kind: {self.kind!r}")
        if type(self.feature_axis) is not int:
            raise TypeError("feature_axis must be an integer")
        if not self.indices:
            raise ValueError("indices must contain at least one selection")
        if any(type(index) is not int for index in self.indices):
            raise TypeError("indices must be integers")
        if self.kind == "parameter" and not self.parameter:
            raise ValueError("parameter targets require a parameter name")
        if self.kind != "parameter" and self.parameter is not None:
            raise ValueError("parameter is only valid for parameter targets")
        if any(not isinstance(component, (int, str)) or isinstance(component, bool) for component in self.output_path):
            raise TypeError("output_path components must be integer slots or string mapping keys")
        if self.kind == "parameter" and self.output_path:
            raise ValueError("output_path is only valid for activation and gradient targets")
        if self.occurrence is not None:
            if type(self.occurrence) is not int:
                raise TypeError("occurrence must be an integer or None")
            if self.occurrence < 0:
                raise ValueError("occurrence must be non-negative")
            if self.kind == "parameter":
                raise ValueError("occurrence is only valid for activation and gradient targets")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this target."""
        data = asdict(self)
        data["indices"] = list(self.indices)
        data["output_path"] = list(self.output_path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Target":
        """Build a target from :meth:`to_dict` output."""
        values = dict(data)
        try:
            values["indices"] = tuple(values["indices"])  # type: ignore[arg-type]
        except KeyError as exc:
            raise ValueError("Target data is missing indices") from exc
        values["output_path"] = tuple(values.get("output_path", ()))  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def to_json(self) -> str:
        """Serialize this target to JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "Target":
        """Deserialize a target produced by :meth:`to_json`."""
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Target JSON is invalid") from exc
        if not isinstance(data, dict):
            raise ValueError("Target JSON must contain an object")
        return cls.from_dict(data)

    def validate(self, model: nn.Module) -> nn.Module:
        """Validate this target against ``model`` and return its selected module."""
        try:
            module = resolve_submodule_path(model, self.module_path)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Target path '{self.module_path}' does not resolve to a module") from exc
        if not isinstance(module, nn.Module):
            raise ValueError(f"Target path '{self.module_path}' does not resolve to a module")
        if self.kind == "parameter":
            try:
                parameter = module.get_parameter(self.parameter)  # type: ignore[arg-type]
            except AttributeError as exc:
                raise ValueError(f"Module '{self.module_path}' has no parameter '{self.parameter}'") from exc
            self._selection(parameter)
        return module

    def _selection(self, tensor: Tensor) -> tuple[object, ...]:
        axis = self.feature_axis if self.feature_axis >= 0 else tensor.ndim + self.feature_axis
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(f"feature_axis {self.feature_axis} is out of bounds for a {tensor.ndim}-D tensor")
        for index in self.indices:
            if index < -tensor.shape[axis] or index >= tensor.shape[axis]:
                raise ValueError(
                    f"index {index} is out of bounds for axis {self.feature_axis} with size {tensor.shape[axis]}"
                )
        return tuple(slice(None) if dim != axis else list(self.indices) for dim in range(tensor.ndim))

    def _select(self, tensor: Tensor) -> Tensor:
        return tensor[self._selection(tensor)]

    def _assign(self, tensor: Tensor, value: Tensor | float | int) -> None:
        tensor[self._selection(tensor)] = value

    def select_output(self, value: object) -> Tensor:
        """Select this target from a structured activation or gradient value."""

        return self._select(self._output_tensor(value, self.output_path))

    def replace_output(self, value: object, replacement: Tensor | float | int) -> object:
        """Return ``value`` with only this target replaced, preserving its structure."""

        tensor = self._output_tensor(value, self.output_path).clone()
        self._assign(tensor, replacement)
        return self._replace_output_tensor(value, self.output_path, tensor)

    @staticmethod
    def _resolved_output_path(
        value: object,
        path: tuple[OutputPathComponent, ...],
    ) -> tuple[OutputPathComponent, ...]:
        if path or isinstance(value, Tensor):
            return path
        if isinstance(value, (tuple, list)) and len(value) == 1:
            return (0,)
        raise ValueError("Structured hook values require Target.output_path to select a tensor leaf")

    @classmethod
    def _output_tensor(
        cls,
        value: object,
        path: tuple[OutputPathComponent, ...] = (),
    ) -> Tensor:
        selected = value
        for component in cls._resolved_output_path(value, path):
            if isinstance(component, int) and isinstance(selected, (tuple, list)):
                try:
                    selected = selected[component]
                except IndexError as exc:
                    raise ValueError(f"output_path slot {component} is out of range") from exc
            elif isinstance(component, str) and isinstance(selected, Mapping):
                try:
                    selected = selected[component]
                except KeyError as exc:
                    raise ValueError(f"output_path key {component!r} is missing") from exc
            else:
                raise ValueError(f"output_path component {component!r} does not match the hook value structure")
        if not isinstance(selected, Tensor):
            raise ValueError("Target.output_path must select a tensor leaf")
        return selected

    @classmethod
    def _replace_output_tensor(
        cls,
        value: object,
        path: tuple[OutputPathComponent, ...],
        replacement: Tensor,
    ) -> object:
        resolved = cls._resolved_output_path(value, path)
        if not resolved:
            return replacement
        component, *remainder = resolved
        if isinstance(component, int) and isinstance(value, tuple):
            items = list(value)
            items[component] = cls._replace_output_tensor(items[component], tuple(remainder), replacement)
            return tuple(items)
        if isinstance(component, int) and isinstance(value, list):
            items = list(value)
            items[component] = cls._replace_output_tensor(items[component], tuple(remainder), replacement)
            return items
        if isinstance(component, str) and isinstance(value, TensorDictBase):
            items = value.clone(recurse=False)
            items.set(component, cls._replace_output_tensor(value.get(component), tuple(remainder), replacement))
            return items
        if isinstance(component, str) and isinstance(value, Mapping):
            try:
                current = value[component]
            except KeyError as exc:
                raise ValueError(f"output_path key {component!r} is missing") from exc
            items = copy.copy(value)
            if not isinstance(items, MutableMapping):
                raise ValueError(
                    f"mapping type {type(value).__name__} does not support structure-preserving replacement"
                )
            items[component] = cls._replace_output_tensor(current, tuple(remainder), replacement)
            return items
        raise ValueError(f"output_path component {component!r} does not match the hook value structure")


__all__ = ["OutputPathComponent", "Target", "TargetKind"]
