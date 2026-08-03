from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover

__all__ = [
    # Core
    "contexts",
    "hooks",
    "metrics",
    "modules",
    # Methods
    "latent",
    "attribution",
    "auto",
    "weights",
    "pipeline",
    "artifacts",
    "stages",
    "concepts",
    "dimension",
]
