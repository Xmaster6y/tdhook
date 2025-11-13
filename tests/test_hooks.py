"""
Tests for the hooks functionality.
"""

import torch
from tensordict import TensorDict
import pytest
import gc

from tdhook.hooks import (
    register_hook_to_module,
    MultiHookManager,
    MultiHookHandle,
    HookFactory,
    EarlyStoppingException,
    MutableWeakRef,
    CacheProxy,
    resolve_submodule_path,
    submodule_path_to_name,
)


class TestHookRegistration:
    """Test hook registration functionality."""

    def test_register_forward_hook(self, default_test_model):
        """Test registering a forward hook."""

        def forward_hook(module, args, output):
            return output + 1

        input = torch.randn(2, 10)
        original_output = default_test_model(input)
        handle = register_hook_to_module(default_test_model.linear1, forward_hook, direction="fwd")
        assert handle is not None
        model_output = default_test_model(input)
        assert not torch.allclose(model_output, original_output)
        handle.remove()
        output = default_test_model(input)
        assert torch.allclose(output, original_output)

    def test_multi_hook_manager(self, default_test_model):
        """Test MultiHookManager."""

        def hook_factory(name: str):
            def hook(module, args, output):
                return output + 1

            return hook

        input = torch.randn(2, 10)
        original_output = default_test_model(input)

        manager = MultiHookManager(pattern=r"linear\d+")
        handle = manager.register_hook(default_test_model, hook_factory, direction="fwd")
        assert isinstance(handle, MultiHookHandle)
        mod_output = default_test_model(input)
        assert not torch.allclose(mod_output, original_output)
        handle.remove()
        output = default_test_model(input)
        assert torch.allclose(output, original_output)


class TestHookFactory:
    """Test hook factory functionality."""

    def test_make_caching_hook(self, default_test_model):
        """Test making a caching hook."""
        cache = TensorDict()
        hook = HookFactory.make_caching_hook("key", cache)
        assert hook is not None
        hook(default_test_model, None, torch.tensor(1))
        assert cache["key"] == 1

    def test_make_setting_hook(self, default_test_model):
        """Test making a setting hook."""

        def callback(value, **_):
            return value + 1

        hook = HookFactory.make_setting_hook(1, callback=callback)
        assert hook is not None
        output = hook(default_test_model, None, 1)
        assert output == 2

    def test_make_stopping_hook(self, default_test_model):
        """Test making a stopping hook."""

        hook = HookFactory.make_stopping_hook("key")
        assert hook is not None
        with pytest.raises(EarlyStoppingException):
            hook(default_test_model, None, 1)

    @pytest.mark.parametrize("callback", [None, lambda value, **_: None])
    def test_setting_hook_none_does_not_change_value(self, default_test_model, callback):
        """Test making a setting hook."""

        hook = HookFactory.make_setting_hook(None, callback=callback)
        output = hook(default_test_model, None, 1)
        assert output is None


class TestMultiHookHandle:
    """Tests specific to MultiHookHandle behaviors."""

    def test_add_handles_and_remove_all(self, default_test_model):
        # import moved to module level

        input = torch.randn(2, 10)
        original_output = default_test_model(input)

        def hook_factory(_name: str):
            def hook(module, args, output):
                return output + 1

            return hook

        manager1 = MultiHookManager(pattern=r"linear1$")
        handle1 = manager1.register_hook(default_test_model, hook_factory, direction="fwd")

        manager2 = MultiHookManager(pattern=r"linear2$|linear3$")
        handle2 = manager2.register_hook(default_test_model, hook_factory, direction="fwd")

        combined = handle1 + handle2
        assert isinstance(combined, MultiHookHandle)

        # Hooks active -> output should differ
        changed_output = default_test_model(input)
        assert not torch.allclose(changed_output, original_output)

        # Removing combined should remove all underlying hooks
        combined.remove()
        restored_output = default_test_model(input)
        assert torch.allclose(restored_output, original_output)

        # Type safety on addition

        with pytest.raises(TypeError):
            _ = combined + 123  # not a MultiHookHandle

    def test_context_manager_removes_on_exit(self, default_test_model):
        input = torch.randn(2, 10)
        original_output = default_test_model(input)

        def hook_factory(_name: str):
            def hook(module, args, output):
                return output + 1

            return hook

        manager = MultiHookManager(pattern=r"linear1$")
        handle = manager.register_hook(default_test_model, hook_factory, direction="fwd")

        with handle:
            changed_output = default_test_model(input)
            assert not torch.allclose(changed_output, original_output)

        # After exiting context, hooks should be removed
        restored_output = default_test_model(input)
        assert torch.allclose(restored_output, original_output)

    def test_empty_handle_remove_noop(self):
        # Should not raise
        handle = MultiHookHandle()
        handle.remove()


class TestMultiHookManagerPattern:
    def test_pattern_setter_changes_selection(self, default_test_model):
        input = torch.randn(2, 10)
        original_output = default_test_model(input)

        def hook_factory(_name: str):
            def hook(module, args, output):
                return output + 1

            return hook

        manager = MultiHookManager(pattern=r"linear1$")
        assert manager.pattern == r"linear1$"

        handle1 = manager.register_hook(default_test_model, hook_factory, direction="fwd")
        out1 = default_test_model(input)
        assert not torch.allclose(out1, original_output)
        handle1.remove()
        out1_restored = default_test_model(input)
        assert torch.allclose(out1_restored, original_output)

        manager.pattern = r"linear2$"
        assert manager.pattern == r"linear2$"

        handle2 = manager.register_hook(default_test_model, hook_factory, direction="fwd")
        out2 = default_test_model(input)
        assert not torch.allclose(out2, original_output)
        handle2.remove()
        out2_restored = default_test_model(input)
        assert torch.allclose(out2_restored, original_output)


class TestHookRegistrationKwargsAndBackward:
    """Covers hooks registration with kwargs and backward directions."""

    def test_register_forward_hook_with_kwargs(self, default_test_model):
        """Forward hook with with_kwargs=True can modify output."""

        def forward_hook(module, args, kwargs, output):
            return output + 1

        x = torch.randn(2, 10)
        original = default_test_model(x)
        handle = register_hook_to_module(default_test_model.linear1, forward_hook, direction="fwd_kwargs")
        out = default_test_model(x)
        assert not torch.allclose(out, original)
        handle.remove()
        out2 = default_test_model(x)
        assert torch.allclose(out2, original)

    def test_register_forward_pre_hook_with_kwargs(self, default_test_model):
        """Forward pre hook with kwargs can modify inputs."""

        def pre_hook(module, args, kwargs):
            return (args[0] + 1,), kwargs

        x = torch.randn(2, 10)
        original = default_test_model(x)
        handle = register_hook_to_module(default_test_model.linear1, pre_hook, direction="fwd_pre_kwargs")
        out = default_test_model(x)
        assert not torch.allclose(out, original)
        handle.remove()
        out2 = default_test_model(x)
        assert torch.allclose(out2, original)

    def test_register_backward_and_backward_pre_hooks(self, default_test_model):
        """Backward and backward pre hooks are invoked during autograd."""

        calls = []

        def bwd_hook(module, grad_input, grad_output):
            calls.append("bwd")
            return grad_input

        def bwd_pre_hook(module, grad_output):
            calls.append("bwd_pre")
            return grad_output

        handle_bwd = register_hook_to_module(default_test_model.linear2, bwd_hook, direction="bwd")
        handle_bwd_pre = register_hook_to_module(default_test_model.linear2, bwd_pre_hook, direction="bwd_pre")

        x = torch.randn(4, 10, requires_grad=True)
        y = default_test_model(x).sum()
        y.backward()

        assert "bwd" in calls and "bwd_pre" in calls
        handle_bwd.remove()
        handle_bwd_pre.remove()


class TestHookSignatureValidation:
    """Input validation for hook signatures and callbacks."""

    def test_invalid_hook_signature_raises(self, default_test_model):
        """Incorrect forward hook signature should raise at registration."""

        def bad_hook(module, output):
            return output

        with pytest.raises(ValueError):
            register_hook_to_module(default_test_model.linear1, bad_hook, direction="fwd")

    def test_callback_missing_params_raises(self):
        """Callback missing required named params is rejected."""

        def bad_cb(module):
            return 0

        cache = TensorDict()

        with pytest.raises(ValueError):
            HookFactory.make_caching_hook("k", cache, callback=bad_cb, direction="fwd")

    def test_caching_hook_fwd_pre_kwargs_value_index(self):
        """Caching hook for fwd_pre_kwargs can use callback to store a non-tuple value."""

        cache = TensorDict()
        hook = HookFactory.make_caching_hook(
            "k",
            cache,
            direction="fwd_pre_kwargs",
            callback=lambda args, **_: args[0],
        )
        module = object()
        my_tensor = torch.tensor(1)
        args = (my_tensor,)
        kwargs = {"unused": 1}
        hook(module, args, kwargs)
        assert cache["k"] is my_tensor


class TestHookEdgeCases:
    """Targeted edge cases for hooks internals."""

    def test_register_invalid_direction_raises(self, default_test_model):
        """Registering with an invalid direction fails early."""

        def some_hook(module, args, output):
            return output

        with pytest.raises(ValueError):
            register_hook_to_module(default_test_model.linear1, some_hook, direction="nope")

    def test_varargs_too_many_positional_params_raises(self, default_test_model):
        """Varargs hooks with too many fixed params are rejected."""

        def bad_varargs_hook(module, a, b, c, d, *args):
            return None

        with pytest.raises(ValueError):
            register_hook_to_module(default_test_model.linear1, bad_varargs_hook, direction="fwd")

    def test_multihookmanager_default_pattern_matches_nothing(self, default_test_model):
        """Default manager pattern matches no modules."""

        def hook_factory(name: str):
            def hook(module, args, output):
                return output + 1

            return hook

        x = torch.randn(2, 10)
        original = default_test_model(x)
        manager = MultiHookManager()
        handle = manager.register_hook(default_test_model, hook_factory, direction="fwd")
        out = default_test_model(x)
        assert torch.allclose(out, original)
        handle.remove()

    def test_cacheproxy_dead_reference_raises(self):
        """Resolving a CacheProxy with a dead cache reference raises."""

        def make_proxy():
            cache = TensorDict()
            return CacheProxy("k", cache)

        proxy = make_proxy()
        gc.collect()

        with pytest.raises(ValueError):
            proxy.resolve()

    def test_callback_positional_only_params_rejected(self):
        """Callbacks with positional-only parameters are not allowed."""

        def cb(module, /, args=None, output=None, value=None):
            return value

        with pytest.raises(ValueError):
            HookFactory.make_setting_hook(1, callback=cb, direction="fwd")

    def test_make_caching_hook_invalid_direction_raises(self):
        """Invalid direction for caching hook raises."""

        with pytest.raises(ValueError):
            HookFactory.make_caching_hook("k", TensorDict(), direction="nope")

    def test_make_setting_hook_cacheproxy_resolve_branch(self):
        """Setting hook resolves CacheProxy and can return a proxy via callback."""

        cache = TensorDict({"k": 123})
        proxy = CacheProxy("k", cache)

        def cb(value, **_):
            return value + 1

        hook = HookFactory.make_setting_hook(proxy, callback=cb, direction="fwd")
        result = hook(object(), None, torch.tensor(-1))
        assert result == 124

    def test_make_setting_hook_type_mismatch_raises(self):
        """Setting hook raises when callback changes the value type."""

        def cb(**_):
            return 1.0

        hook = HookFactory.make_setting_hook(1, callback=cb, direction="fwd")
        with pytest.raises(RuntimeError):
            hook(object(), None, None)

    def test_caching_hook_non_tensor_value_raises(self):
        """Caching hook raises when callback returns a non-tensor value."""

        cache = TensorDict()
        hook = HookFactory.make_caching_hook(
            "k",
            cache,
            direction="fwd_pre_kwargs",
            callback=lambda args, **_: args[0],
        )
        module = object()
        args = ("A",)
        kwargs = {"unused": 1}
        with pytest.raises(RuntimeError):
            hook(module, args, kwargs)

    def test_caching_hook_dead_cache_reference_raises(self):
        """Caching hook raises when underlying cache has been garbage collected."""

        cache = TensorDict()
        cache_ref = MutableWeakRef(cache)

        # Remove strong reference to allow garbage collection
        del cache
        gc.collect()

        hook = HookFactory.make_caching_hook("k", cache_ref, direction="fwd")

        # Expect ValueError due to dead weak reference resolution inside hook
        with pytest.raises(ValueError):
            hook(object(), None, torch.tensor(1))


class TestResolveSubmodulePath:
    """Test resolve_submodule_path functionality with simple dummy objects."""

    def test_empty_key_returns_root(self):
        """Empty key should return the root object."""

        class DummyRoot:
            pass

        root = DummyRoot()
        assert resolve_submodule_path(root, "") is root

    def test_simple_attribute_access(self):
        """Test simple attribute access."""

        class DummyRoot:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = "value2"

        root = DummyRoot()
        assert resolve_submodule_path(root, "attr1") == "value1"
        assert resolve_submodule_path(root, "attr2") == "value2"

    def test_nested_attribute_access(self):
        """Test nested attribute access."""

        class DummyChild:
            def __init__(self):
                self.nested_attr = "nested_value"

        class DummyRoot:
            def __init__(self):
                self.child = DummyChild()

        root = DummyRoot()
        assert resolve_submodule_path(root, "child.nested_attr") == "nested_value"

    def test_list_indexing(self):
        """Test list indexing patterns."""

        class DummyRoot:
            def __init__(self):
                self.items = ["first", "second", "third"]

        root = DummyRoot()

        # Test positive indexing
        assert resolve_submodule_path(root, "items[0]") == "first"
        assert resolve_submodule_path(root, "items[1]") == "second"

        # Test negative indexing
        assert resolve_submodule_path(root, "items[-1]") == "third"
        assert resolve_submodule_path(root, "items[-2]") == "second"

        # Test slice indexing
        assert resolve_submodule_path(root, "items[1:3]") == ["second", "third"]

    def test_dict_indexing(self):
        """Test dictionary indexing patterns."""

        class DummyRoot:
            def __init__(self):
                self.data = {"first": "value1", "second": "value2", "third": "value3"}

        root = DummyRoot()

        # Test string key indexing
        assert resolve_submodule_path(root, "data['first']") == "value1"
        assert resolve_submodule_path(root, "data['second']") == "value2"

    def test_custom_attributes_with_angles(self):
        """Test custom attributes using angles syntax."""

        class DummyChild:
            def __init__(self):
                pass

        class DummyRoot:
            def __init__(self):
                self.child = DummyChild()

        root = DummyRoot()
        setattr(root, "block0/module", "custom_value1")
        setattr(root, "block1/module", ["custom_value2"])
        setattr(root, "block2/module", DummyChild())
        setattr(getattr(root, "block2/module"), "subblock0/module", ["custom_value3"])
        setattr(root.child, "block3/module", {"custom_value4": "custom_value5"})

        # Test custom attribute access
        assert resolve_submodule_path(root, "<block0/module>") == "custom_value1"
        assert resolve_submodule_path(root, "<block1/module>[0]") == "custom_value2"
        assert resolve_submodule_path(root, "<block2/module>.<subblock0/module>[0]") == "custom_value3"
        assert resolve_submodule_path(root, "<block2/module><subblock0/module>[0]") == "custom_value3"
        assert resolve_submodule_path(root, "child.<block3/module>['custom_value4']") == "custom_value5"

    def test_number_attribute_access(self):
        """Test number attribute access."""

        class DummyChild(torch.nn.Module):
            def __init__(self):
                self.layers = ["a", "b", "c"]

        class DummyRoot:
            def __init__(self):
                self.m1 = torch.nn.ModuleList(
                    [
                        torch.nn.ReLU(),
                        torch.nn.ReLU(),
                        DummyChild(),
                    ]
                )

        root = DummyRoot()
        assert resolve_submodule_path(root, "m1.0") is root.m1[0]
        assert resolve_submodule_path(root, "m1.2.layers[1]") is root.m1[2].layers[1]

    def test_mixed_attribute_and_indexing(self):
        """Test mixed attribute access and indexing."""

        class DummyChild:
            def __init__(self):
                self.items = ["a", "b", "c"]

        class DummyRoot:
            def __init__(self):
                self.layers = [DummyChild(), DummyChild()]
                self.name = "root"

        root = DummyRoot()

        # Test mixed patterns
        assert resolve_submodule_path(root, "layers[0].items[1]") == "b"
        assert resolve_submodule_path(root, "name") == "root"

    def test_function_call(self):
        """Test function call."""

        class DummyRoot:
            def __init__(self):
                self.fn = lambda x: x + 1

        root = DummyRoot()
        assert resolve_submodule_path(root, "fn(0)") == 1

    def test_invalid_paths_raise_value_error(self):
        """Test that invalid paths raise ValueError."""

        class DummyRoot:
            def __init__(self):
                self.valid_attr = "value"

        root = DummyRoot()

        # Test non-existent attribute
        with pytest.raises(ValueError):
            resolve_submodule_path(root, "nonexistent")

        # Test invalid indexing
        with pytest.raises(ValueError):
            resolve_submodule_path(root, "valid_attr[999]")

        # Test malformed custom attribute (missing closing colon)
        with pytest.raises(ValueError):
            resolve_submodule_path(root, ":block0/module")

        # Test malformed custom attribute (missing opening colon)
        with pytest.raises(ValueError):
            resolve_submodule_path(root, "block0/module:")


class TestSubmodulePathToName:
    """Test submodule_path_to_name functionality."""

    def test_simple_attribute_access(self):
        """Test simple attribute access."""
        assert submodule_path_to_name("") == ""
        assert submodule_path_to_name("attr1") == "attr1"
        assert submodule_path_to_name(".attr2") == "attr2"

    def test_nested_attribute_access(self):
        """Test nested attribute access."""
        assert submodule_path_to_name("child.nested_attr") == "child.nested_attr"

    def test_list_indexing(self):
        """Test list indexing patterns."""
        assert submodule_path_to_name("items[0]") == "items.0"
        assert submodule_path_to_name("items[1]") == "items.1"

    def test_dict_indexing(self):
        """Test dictionary indexing patterns."""
        assert submodule_path_to_name("data['first']") == "data.first"
        assert submodule_path_to_name('data["second"]') == "data.second"

    def test_custom_attributes_with_angles(self):
        """Test custom attributes using angles syntax."""
        assert submodule_path_to_name("<block0/module>") == "block0/module"
        assert submodule_path_to_name("<block1/module>[0]") == "block1/module.0"
        assert submodule_path_to_name("<block2/module>.<subblock0/module>[0]") == "block2/module.subblock0/module.0"
        assert submodule_path_to_name("<block2/module><subblock0/module>[0]") == "block2/module.subblock0/module.0"
        assert submodule_path_to_name("child.<block3/module>['custom_value4']") == "child.block3/module.custom_value4"

    def test_mixed_attribute_and_indexing(self):
        """Test mixed attribute access and indexing."""
        assert submodule_path_to_name("layers[0].items[1]") == "layers.0.items.1"
        assert submodule_path_to_name("layers.1name") == "layers.1name"

    def test_paths_returned_as_is(self):
        """Test paths that should be returned as-is."""
        assert submodule_path_to_name("[-1]") == "[-1]"
        assert submodule_path_to_name("[1:3]") == "[1:3]"
        assert submodule_path_to_name(":something") == ":something"
        assert submodule_path_to_name("(arg)") == "(arg)"
        assert submodule_path_to_name(")something") == ")something"
