import pytest

from tdhook._types import append_key_suffix, is_nested_key, join_keys


def test_runtime_nested_key_validation_and_composition():
    assert is_nested_key("value")
    assert is_nested_key(("nested", "value"))
    assert not is_nested_key(())
    assert not is_nested_key(("nested", 1))
    assert append_key_suffix(("metrics", "value"), "_x") == ("metrics", "value_x")
    assert join_keys(("metrics", "group"), "value") == ("metrics", "group", "value")


def test_key_helpers_reject_invalid_runtime_values():
    with pytest.raises(TypeError, match="key must"):
        append_key_suffix((), "_x")
    with pytest.raises(TypeError, match="suffix"):
        append_key_suffix("value", object())
    with pytest.raises(TypeError, match="prefix and key"):
        join_keys((), "value")
