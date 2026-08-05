"""Serializable, reusable selectors for model activations, gradients, and parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal

from torch import Tensor, nn

from tdhook.paths import resolve_submodule_path


TargetKind = Literal["activation", "gradient", "parameter"]


@dataclass(frozen=True)
class Target:
    """A serializable selection within a module's output, gradient, or parameter.

    ``feature_axis`` identifies the axis containing units, channels, rows, or
    columns.  For example, use ``-1`` for MLP output units, ``1`` for CNN
    output channels, and ``0``/``1`` for parameter rows/columns respectively.
    """

    module_path: str
    kind: TargetKind
    feature_axis: int
    indices: tuple[int, ...]
    parameter: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"activation", "gradient", "parameter"}:
            raise ValueError(f"Invalid target kind: {self.kind!r}")
        if not self.indices:
            raise ValueError("indices must contain at least one selection")
        if any(not isinstance(index, int) for index in self.indices):
            raise TypeError("indices must be integers")
        if self.kind == "parameter" and not self.parameter:
            raise ValueError("parameter targets require a parameter name")
        if self.kind != "parameter" and self.parameter is not None:
            raise ValueError("parameter is only valid for parameter targets")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this target."""
        data = asdict(self)
        data["indices"] = list(self.indices)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Target":
        """Build a target from :meth:`to_dict` output."""
        values = dict(data)
        try:
            values["indices"] = tuple(values["indices"])  # type: ignore[arg-type]
        except KeyError as exc:
            raise ValueError("Target data is missing indices") from exc
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


__all__ = ["Target", "TargetKind"]
