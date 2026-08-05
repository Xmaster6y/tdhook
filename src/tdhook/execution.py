"""Method-owned execution requirements not represented by TensorDict modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GradientMode(StrEnum):
    """Autograd requirement for a method's model executions."""

    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


class AutogradLifetime(StrEnum):
    """When a method's autograd-dependent resources may be released."""

    CALL = "call"
    BACKWARD = "backward"


@dataclass(frozen=True)
class ExecutionSpec:
    """Static execution requirements for one configured method.

    TensorDict modules already expose their data contract through ``in_keys``
    and ``out_keys``.  This record deliberately contains only requirements
    TensorDict cannot express: the number of model passes and whether those
    passes require autograd and when autograd-dependent resources may be
    released. ``CALL`` methods complete their autograd work while they run;
    ``BACKWARD`` methods require their hooks and prepared state through a
    caller-driven backward pass.
    """

    model_passes: int = 1
    gradient_mode: GradientMode = GradientMode.OPTIONAL
    autograd_lifetime: AutogradLifetime = AutogradLifetime.CALL

    def __post_init__(self) -> None:
        if isinstance(self.model_passes, bool) or not isinstance(self.model_passes, int):
            raise TypeError("model_passes must be an integer")
        if self.model_passes <= 0:
            raise ValueError("method execution requires at least one model pass")
        if not isinstance(self.gradient_mode, GradientMode):
            raise TypeError("gradient_mode must be a GradientMode")
        if not isinstance(self.autograd_lifetime, AutogradLifetime):
            raise TypeError("autograd_lifetime must be an AutogradLifetime")
        if self.autograd_lifetime is AutogradLifetime.BACKWARD and self.gradient_mode is not GradientMode.REQUIRED:
            raise ValueError("deferred backward requires gradient_mode=REQUIRED")


__all__ = ["AutogradLifetime", "ExecutionSpec", "GradientMode"]
