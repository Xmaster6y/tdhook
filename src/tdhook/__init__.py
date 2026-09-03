from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover

__all__ = [
    # Core
    "contexts",
    "execution",
    "hooks",
    "interventions",
    "paths",
    "runtime",
    "session",
    "targets",
    "workflow",
    "metrics",
    "modules",
    # Methods
    "latent",
    "attribution",
    "weights",
    "concepts",
    "dimension",
]
