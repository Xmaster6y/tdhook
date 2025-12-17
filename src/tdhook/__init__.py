from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    # Core
    "contexts",
    "hooks",
    "metrics",
    "module",
    # Methods
    "latent",
    "attribution",
    "auto",
    "weights",
]
