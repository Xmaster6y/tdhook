from tensordict.utils import NestedKey


def is_nested_key(value: object) -> bool:
    """Return whether ``value`` is a valid runtime TensorDict nested key."""

    return isinstance(value, str) or (
        isinstance(value, tuple) and bool(value) and all(isinstance(part, str) for part in value)
    )


def append_key_suffix(key: NestedKey, suffix: str) -> NestedKey:
    """Append a suffix to the leaf of a native TensorDict key."""

    if not is_nested_key(key):
        raise TypeError("key must be a string or non-empty tuple of strings")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    if isinstance(key, str):
        return f"{key}{suffix}"
    return (*key[:-1], f"{key[-1]}{suffix}")


def join_keys(prefix: NestedKey, key: NestedKey) -> tuple[str, ...]:
    """Join two native TensorDict keys without nesting tuple objects."""

    if not is_nested_key(prefix) or not is_nested_key(key):
        raise TypeError("prefix and key must be strings or non-empty tuples of strings")
    prefix_parts = (prefix,) if isinstance(prefix, str) else prefix
    key_parts = (key,) if isinstance(key, str) else key
    return (*prefix_parts, *key_parts)
