from tensordict.utils import NestedKey


# Internal compatibility alias. TensorDict owns the public key model.
UnraveledKey = NestedKey


def append_key_suffix(key: NestedKey, suffix: str) -> NestedKey:
    """Append a suffix to the leaf of a native TensorDict key."""

    if not isinstance(key, NestedKey):
        raise TypeError("key must be a string or non-empty tuple of strings")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    if isinstance(key, str):
        return f"{key}{suffix}"
    return (*key[:-1], f"{key[-1]}{suffix}")


def join_keys(prefix: NestedKey, key: NestedKey) -> tuple[str, ...]:
    """Join two native TensorDict keys without nesting tuple objects."""

    if not isinstance(prefix, NestedKey) or not isinstance(key, NestedKey):
        raise TypeError("prefix and key must be strings or non-empty tuples of strings")
    prefix_parts = (prefix,) if isinstance(prefix, str) else prefix
    key_parts = (key,) if isinstance(key, str) else key
    return (*prefix_parts, *key_parts)
