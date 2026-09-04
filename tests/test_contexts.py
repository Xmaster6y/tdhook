"""
Tests for method context.
"""

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from typing import List

import pytest

from tdhook.contexts import HookingContext, HookingContextFactory
from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle
from tdhook.runtime import HookProgramBuilder, HookSpec


class Context1(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        with HookProgramBuilder() as builder:
            builder.register_path(
                module.hook_root,
                lambda module, args, output: output + 1,
                HookSpec("", "replace", "fwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class Context2(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        with HookProgramBuilder() as builder:
            builder.register_path(
                module.hook_root,
                lambda module, args, output: output * 2,
                HookSpec("", "replace", "fwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class PrepFlagFactory(HookingContextFactory):
    def __init__(self, flag_name: str = "prep_flag"):
        super().__init__()
        self.flag_name = flag_name

    def _prepare_module(
        self,
        module: TensorDictModule,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModule:
        return module

    def _hook_module(self, module):
        setattr(module.hook_root, self.flag_name, 1)
        return MultiHookHandle()

    def _restore_module(
        self,
        module: TensorDictModule,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey],
        extra_relative_path: str,
    ) -> TensorDictModule:
        delattr(module, self.flag_name)
        return module


class RestoreFailureFactory(HookingContextFactory):
    def _restore_module(self, module, in_keys, out_keys, extra_relative_path):
        raise RuntimeError("restoration failed")


class RemoveFailureHandle:
    def __init__(self, should_fail, removed):
        self.should_fail = should_fail
        self.removed = removed

    def remove(self):
        self.removed.append(self.should_fail)
        if self.should_fail:
            raise RuntimeError("removal failed")


class RestoreAfterRemovalFailureFactory(PrepFlagFactory):
    def _hook_module(self, module):
        setattr(module.hook_root, self.flag_name, 1)
        return MultiHookHandle([RemoveFailureHandle(True, [])])


class ProgramFailureHandle:
    def __init__(self, removed):
        self.removed = removed

    @property
    def program(self):
        raise RuntimeError("program inspection failed")

    def remove(self):
        self.removed.append(True)


class ProgramFailureFactory(HookingContextFactory):
    def __init__(self, removed):
        super().__init__()
        self.removed = removed

    def _hook_module(self, module):
        return ProgramFailureHandle(self.removed)


class TestBaseContext:
    """Basic single-context behavior."""

    def test_context1(self, default_test_model):
        """Applies +1 hook via Context1."""
        input = torch.randn(2, 3, 10)
        original_output = default_test_model(input)
        with Context1().prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": input}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert torch.allclose(data["output"], original_output + 1)

    def test_context2(self, default_test_model):
        """Applies *2 hook via Context2."""
        input = torch.randn(2, 3, 10)
        original_output = default_test_model(input)
        with Context2().prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": input}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert torch.allclose(data["output"], original_output * 2)


class TestHookingContextLifecycle:
    def test_internal_binding_root_requires_a_tensordict_module_and_relative_path(self):
        with pytest.raises(TypeError, match="TensorDict module and relative path"):
            HookingContext(
                HookingContextFactory(), torch.nn.Identity(), hook_root=TensorDictModule(torch.nn.Identity(), [], [])
            )

    def test_hook_failure_cleanup_requires_an_active_bound_program(self, default_test_model):
        context = HookingContextFactory().prepare(default_test_model)

        with pytest.raises(RuntimeError, match="only available inside"):
            context.on_hook_failure(lambda: None)

        with context:
            with pytest.raises(TypeError, match="BoundHookProgram"):
                context.on_hook_failure(lambda: None)

    def test_cannot_enter_twice(self, default_test_model):
        """Raises when entering the same context twice."""
        ctx = Context1().prepare(default_test_model)
        with ctx:
            with pytest.raises(RuntimeError):
                ctx.__enter__()

    def test_bound_module_cannot_run_outside_context(self, default_test_model):
        """HookedModule cannot be called outside of its context."""
        x = torch.randn(2, 3, 10)
        original_output = default_test_model(x)
        ctx = Context1().prepare(default_test_model)
        with ctx as hm:
            data = TensorDict({"input": x}, batch_size=[2, 3])
            hm(data)
            assert torch.allclose(data["output"], original_output + 1)
        data2 = TensorDict({"input": x}, batch_size=[2, 3])
        with pytest.raises(RuntimeError):
            hm(data2)

    def test_disable_hooks_temporarily(self, default_test_model):
        """Temporarily disabling hooks restores raw behavior."""
        x = torch.randn(2, 3, 10)
        original_output = default_test_model(x)
        ctx = Context1().prepare(default_test_model)
        with ctx as hm:
            data = TensorDict({"input": x}, batch_size=[2, 3])
            hm(data)
            assert torch.allclose(data["output"], original_output + 1)

            with ctx.disable_hooks():
                data_disabled = TensorDict({"input": x}, batch_size=[2, 3])
                hm(data_disabled)
                assert torch.allclose(data_disabled["output"], original_output)

            data_again = TensorDict({"input": x}, batch_size=[2, 3])
            hm(data_again)
            assert torch.allclose(data_again["output"], original_output + 1)

    def test_disable_context_yields_raw_module(self, default_test_model):
        """Disabling context yields the raw underlying module."""
        x = torch.randn(2, 3, 10)
        original_output = default_test_model(x)
        ctx = Context1().prepare(default_test_model)
        with ctx as hm:
            with ctx.disable() as raw_module:
                raw_out = raw_module(x)
                assert torch.allclose(raw_out, original_output)

            data_after = TensorDict({"input": x}, batch_size=[2, 3])
            hm(data_after)
            assert torch.allclose(data_after["output"], original_output + 1)

    def test_disable_hooks_outside_context_raises(self, default_test_model):
        """disable_hooks() outside context raises."""
        ctx = Context1().prepare(default_test_model)
        with pytest.raises(RuntimeError):
            with ctx.disable_hooks():
                pass

    def test_disable_context_outside_context_raises(self, default_test_model):
        """disable() outside context raises."""
        ctx = Context1().prepare(default_test_model)
        with pytest.raises(RuntimeError):
            with ctx.disable():
                pass

    def test_normal_exit_restores_after_hook_removal_failure(self, default_test_model):
        context = RestoreAfterRemovalFailureFactory().prepare(default_test_model)

        with pytest.raises(RuntimeError, match="removal failed"):
            with context:
                assert hasattr(context._module, "prep_flag")

        assert not hasattr(context._module, "prep_flag")

    def test_normal_exit_reports_restore_failure(self, default_test_model):
        context = RestoreFailureFactory().prepare(default_test_model)

        with pytest.raises(RuntimeError, match="restoration failed"):
            with context:
                pass

        assert not context._in_context
        assert context._hooked_module is None

    def test_normal_exit_reports_pre_context_stack_failure(self, default_test_model):
        class FailingStack:
            def __exit__(self, *args):
                raise RuntimeError("stack cleanup failed")

        context = HookingContextFactory().prepare(default_test_model)
        with pytest.raises(RuntimeError, match="stack cleanup failed"):
            with context:
                context._stack = FailingStack()

        assert not context._in_context
        assert context._stack is None

    def test_entry_failure_after_hook_installation_removes_the_handle(self, default_test_model):
        removed = []
        context = ProgramFailureFactory(removed).prepare(default_test_model)

        with pytest.raises(RuntimeError, match="program inspection failed"):
            with context:
                pass

        assert removed == [True]

    def test_direct_execution_state_is_scoped_to_an_active_binding(self, default_test_model):
        context = HookingContextFactory().prepare(default_test_model)
        with pytest.raises(RuntimeError, match="only available inside"):
            _ = context.executes_model_directly
        with context:
            assert context.executes_model_directly is True


class TestTensorDictModuleContext:
    def test_prepare_and_restore_td_module_calls_wrapped_prepare_restore(self, default_test_model):
        """Prepare/restore of a TensorDictModule uses factory hooks on wrapped module."""
        td_mod = TensorDictModule(module=default_test_model, in_keys=["input"], out_keys=["output"])
        assert not hasattr(td_mod, "prep_flag")

        ctx = PrepFlagFactory().prepare(td_mod)
        assert not hasattr(td_mod, "prep_flag")
        with ctx as hm:
            assert isinstance(hm, HookedModule)
            assert getattr(td_mod, "prep_flag") == 1
        assert not hasattr(td_mod, "prep_flag")

    def test_in_out_keys_default_from_td_module(self, default_test_model):
        """HookingContext defaults in/out keys from the TensorDictModule."""
        td_mod = TensorDictModule(module=default_test_model, in_keys=["foo"], out_keys=["bar"])
        with HookingContextFactory().prepare(td_mod) as hm:
            assert hm.in_keys == ["foo"]
            assert hm.out_keys == ["bar"]
            x = torch.randn(2, 3, 10)
            data = TensorDict({"foo": x}, batch_size=[2, 3])
            hm(data)
            assert "bar" in data and data["bar"].shape == (2, 3, 5)

    def test_prepare_validates_selected_tensordict_module_keys(self, default_test_model):
        td_mod = TensorDictModule(module=default_test_model, in_keys=["input"], out_keys=["output"])
        invalid = (
            ({"in_keys": [object()]}, "in_keys must be TensorDict nested keys"),
            ({"in_keys": ["missing"]}, "not in module.in_keys"),
            ({"out_keys": [object()]}, "out_keys must be TensorDict nested keys"),
            ({"out_keys": ["missing"]}, "not in module.out_keys"),
        )
        for kwargs, message in invalid:
            with pytest.raises(ValueError, match=message):
                HookingContextFactory().prepare(td_mod, **kwargs)
