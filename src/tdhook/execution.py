"""Method-owned execution requirements not represented by TensorDict modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GradientMode(StrEnum):
    """Autograd requirement for a method's model executions."""

    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True)
class ExecutionSpec:
    """Static execution requirements for one configured method.

    TensorDict modules already expose their data contract through ``in_keys``
    and ``out_keys``.  This record deliberately contains only requirements
    TensorDict cannot express: the number of model passes and whether those
    passes require autograd.
    """

    model_passes: int = 1
    gradient_mode: GradientMode = GradientMode.OPTIONAL

    def __post_init__(self) -> None:
        if isinstance(self.model_passes, bool) or not isinstance(self.model_passes, int):
            raise TypeError("model_passes must be an integer")
        if self.model_passes <= 0:
            raise ValueError("method execution requires at least one model pass")
        if not isinstance(self.gradient_mode, GradientMode):
            raise TypeError("gradient_mode must be a GradientMode")


__all__ = ["ExecutionSpec", "GradientMode"]
