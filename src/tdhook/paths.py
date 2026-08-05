"""Declarative, side-effect-free model path resolution."""

from __future__ import annotations

from ast import literal_eval
import inspect
import re

from torch import nn


def resolve_submodule_path(root: nn.Module, path: str):
    """Resolve a declarative attribute-and-index path from ``root``.

    The grammar is ``attribute ("." attribute | index | escaped_attribute)*``.
    Attributes contain only letters, digits, and underscores; use
    ``<attribute name>`` for any other name (for example ``<block/0>``).
    Indexes are integer literals (including negative integers), quoted string
    literals, or integer slices such as ``[1:3]``. A path may also start with
    an index. Calls, operators, comprehensions, and arbitrary expressions are
    rejected while parsing, before any attribute or item lookup is attempted.
    """
    if not isinstance(path, str):
        raise TypeError("submodule paths must be strings")
    if not path:
        return root

    operations = _parse_submodule_path(path)
    current = root
    try:
        for operation, value in operations:
            current = (
                _resolve_submodule_attribute(current, value)
                if operation == "attribute"
                else _resolve_submodule_index(current, value)
            )
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid submodule path '{path}': {exc}") from exc
    return current


def _parse_submodule_path(path: str) -> list[tuple[str, str | int | slice]]:
    """Parse ``path`` without accessing the object it will be resolved from."""
    operations: list[tuple[str, str | int | slice]] = []
    position = 0
    needs_segment = True
    allows_index = True
    while position < len(path):
        char = path[position]
        if needs_segment:
            if char == "[":
                if not allows_index:
                    raise ValueError(f"Invalid submodule path '{path}': expected an attribute at {position}")
                position, index = _parse_submodule_index(path, position)
                operations.append(("index", index))
            elif char == "<":
                end = path.find(">", position + 1)
                if end < 0:
                    raise ValueError(f"Invalid submodule path '{path}': missing closing '>'")
                attribute = path[position + 1 : end]
                if not attribute or "<" in attribute:
                    raise ValueError(f"Invalid submodule path '{path}': invalid escaped attribute")
                operations.append(("attribute", attribute))
                position = end + 1
            else:
                match = re.match(r"[A-Za-z0-9_]+", path[position:])
                if match is None:
                    raise ValueError(f"Invalid submodule path '{path}': expected an attribute or index at {position}")
                operations.append(("attribute", match.group()))
                position += len(match.group())
            needs_segment = False
            allows_index = True
            continue
        if char == "[":
            position, index = _parse_submodule_index(path, position)
            operations.append(("index", index))
        elif char == ".":
            position += 1
            needs_segment = True
            allows_index = False
        elif char == "<":
            needs_segment = True
        else:
            raise ValueError(f"Invalid submodule path '{path}': unexpected character {char!r} at {position}")
    if needs_segment:
        raise ValueError(f"Invalid submodule path '{path}': path cannot end with '.'")
    return operations


def _parse_submodule_index(path: str, position: int) -> tuple[int, str | int | slice]:
    """Parse one literal index without evaluating arbitrary Python code."""
    end = _find_submodule_index_end(path, position)
    value = path[position + 1 : end]
    if re.fullmatch(r"-?\d+", value):
        return end + 1, int(value)
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        try:
            key = literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid submodule path '{path}': invalid string index") from exc
        if isinstance(key, str):
            return end + 1, key
    if ":" in value:
        parts = value.split(":")
        if len(parts) > 3 or any(part and re.fullmatch(r"-?\d+", part) is None for part in parts):
            raise ValueError(f"Invalid submodule path '{path}': slices contain only integer bounds")
        bounds = [int(part) if part else None for part in parts]
        return end + 1, slice(*bounds)
    raise ValueError(f"Invalid submodule path '{path}': indexes must be integers, slices, or quoted strings")


def _find_submodule_index_end(path: str, position: int) -> int:
    """Return the closing bracket for an index, respecting string literals."""
    quote = None
    end = position + 1
    while end < len(path):
        char = path[end]
        if quote is None:
            if char in "\"'":
                quote = char
            elif char == "]":
                return end
        elif char == "\\":
            end += 1
        elif char == quote:
            quote = None
        end += 1
    raise ValueError(f"Invalid submodule path '{path}': missing closing ']'")


def _resolve_submodule_attribute(current: object, attribute: str) -> object:
    """Read a registered module or plain instance attribute without descriptors."""
    modules = _instance_attributes(current).get("_modules")
    if isinstance(modules, dict) and attribute in modules:
        return modules[attribute]
    if isinstance(modules, dict) and isinstance(modules.get("module"), nn.Module):
        return _resolve_submodule_attribute(modules["module"], attribute)
    value = inspect.getattr_static(current, attribute)
    if inspect.getattr_static(type(value), "__get__", None) is not None:
        raise AttributeError(f"'{type(current).__name__}' has a descriptor attribute '{attribute}'")
    return value


def _resolve_submodule_index(current: object, index: str | int | slice) -> object:
    """Index only registered module containers and builtin collections."""
    if isinstance(current, nn.ModuleDict):
        if not isinstance(index, str):
            raise TypeError("ModuleDict indexes must be strings")
        modules = _instance_attributes(current).get("_modules")
        assert isinstance(modules, dict)
        return modules[index]
    if isinstance(current, (nn.ModuleList, nn.Sequential)):
        if isinstance(index, str):
            raise TypeError(f"{type(current).__name__} indexes must be integers or slices")
        modules = _instance_attributes(current).get("_modules")
        assert isinstance(modules, dict)
        return list(modules.values())[index]
    if isinstance(current, nn.Module):
        raise TypeError(f"'{type(current).__name__}' is not a positional module container")
    if type(current) in (dict, list, tuple):
        return current[index]
    raise TypeError(f"'{type(current).__name__}' is not a supported indexable container")


def _instance_attributes(value: object) -> dict[str, object]:
    """Return an instance dictionary without invoking custom attribute access."""
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except AttributeError:
        return {}
    return attributes if isinstance(attributes, dict) else {}


def submodule_path_to_name(path: str) -> str:
    """Convert a submodule path to a name."""
    if re.search(r"(\[\-)|\:|\(|\)", path):
        return path
    path = re.sub(r"[\"\']", "", path)
    path = re.sub(r"[<>\[\]]", ".", path)
    path = re.sub(r"\.+", ".", path)
    return path.strip(".")


__all__ = ["resolve_submodule_path", "submodule_path_to_name"]
