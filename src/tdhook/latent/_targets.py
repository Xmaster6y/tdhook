"""Shared target normalization for prepared activation methods."""

from tdhook.targets import Target


def activation_target(value: str | Target, *, argument: str) -> tuple[str, Target | None]:
    """Normalize a whole-output module path or an activation selection."""

    if isinstance(value, str):
        return value, None
    if not isinstance(value, Target):
        raise TypeError(f"{argument} entries must be module paths or Targets")
    if value.kind != "activation":
        raise ValueError("prepared activation methods require activation Targets")
    return value.module_path, value


__all__ = ["activation_target"]
