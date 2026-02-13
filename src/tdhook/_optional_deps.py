import functools
import importlib.util


def _ensure_sklearn() -> None:
    """Raise ImportError if scikit-learn is not installed."""
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("scikit-learn is required for this feature. Install with: pip install scikit-learn")


def requires_sklearn(func):
    """Decorator: raise ImportError if sklearn is missing when the decorated function is called."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ensure_sklearn()
        return func(*args, **kwargs)

    return wrapper
