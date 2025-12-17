"""
tdhook
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tdhook")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    # Core
    "artifacts",
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
