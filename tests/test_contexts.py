"""
Tests for the contexts functionality.
"""

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import List

import pytest

from tdhook.contexts import HookingContextFactory, CompositeHookingContextFactory
from tdhook.modules import HookedModule
from tdhook.hooks import MultiHookHandle
from tdhook._types import UnraveledKey


class Context1(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handle = module.register_submodule_hook(
            key="",
            hook=lambda module, args, output: output + 1,
            direction="fwd",
        )
        return handle


class Context2(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handle = module.register_submodule_hook(
            key="",
            hook=lambda module, args, output: output * 2,
            direction="fwd",
        )
        return handle


class PrepFlagFactory(HookingContextFactory):
    def __init__(self, flag_name: str = "prep_flag"):
        super().__init__()
        self.flag_name = flag_name

    def _prepare_module(
        self,
        module: TensorDictModule,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModule:
        setattr(module, self.flag_name, 1)
        return module

    def _restore_module(
        self,
        module: TensorDictModule,
        in_keys: List[UnraveledKey],
        out_keys: List[UnraveledKey],
        extra_relative_path: str,
    ) -> TensorDictModule:
        delattr(module, self.flag_name)
        return module


class BadSpawnFactory(HookingContextFactory):
    def _spawn_hooked_module(self, prep_module, in_keys, out_keys, hooking_context):
        return super()._spawn_hooked_module(prep_module, in_keys, out_keys, hooking_context)


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


class TestCompositeContext:
    """Composition of multiple contexts."""

    def test_composite_context(self, default_test_model):
        """Composes Context1 then Context2."""
        input = torch.randn(2, 3, 10)
        original_output = default_test_model(input)
        context = CompositeHookingContextFactory(Context1(), Context2())
        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": input}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert torch.allclose(data["output"], (original_output + 1) * 2)


class TestHookingContextLifecycle:
    def test_cannot_enter_twice(self, default_test_model):
        """Raises when entering the same context twice."""
        ctx = Context1().prepare(default_test_model)
        with ctx:
            with pytest.raises(RuntimeError):
                ctx.__enter__()

    def test_hooked_module_cannot_run_outside_context(self, default_test_model):
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

            with hm.disable_context_hooks():
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
            with hm.disable_context() as raw_module:
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


class TestCompositeContextDisable:
    def test_disable_hooks_in_composite(self, default_test_model):
        """Disabling hooks in a composite restores raw behavior temporarily."""
        x = torch.randn(2, 3, 10)
        original_output = default_test_model(x)
        composite = CompositeHookingContextFactory(Context1(), Context2())
        with composite.prepare(default_test_model) as hm:
            data = TensorDict({"input": x}, batch_size=[2, 3])
            hm(data)
            assert torch.allclose(data["output"], (original_output + 1) * 2)

            with hm.disable_context_hooks():
                data_disabled = TensorDict({"input": x}, batch_size=[2, 3])
                hm(data_disabled)
                assert torch.allclose(data_disabled["output"], original_output)

            data_after = TensorDict({"input": x}, batch_size=[2, 3])
            hm(data_after)
            assert torch.allclose(data_after["output"], (original_output + 1) * 2)


class TestTensorDictModuleContext:
    def test_prepare_and_restore_td_module_calls_wrapped_prepare_restore(self, default_test_model):
        """Prepare/restore of a TensorDictModule uses factory hooks on wrapped module."""
        td_mod = TensorDictModule(module=default_test_model, in_keys=["input"], out_keys=["output"])
        assert not hasattr(td_mod, "prep_flag")

        ctx = PrepFlagFactory().prepare(td_mod)
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


class TestCompositeTensorDictModule:
    def test_composite_prepare_restore_td_module_order(self, default_test_model):
        """Composite applies prepare for each context and restores in reverse order."""
        td_mod = TensorDictModule(module=default_test_model, in_keys=["input"], out_keys=["output"])
        c1 = PrepFlagFactory("flag1")
        c2 = PrepFlagFactory("flag2")
        composite = CompositeHookingContextFactory(c1, c2)
        assert not hasattr(td_mod, "flag1")
        assert not hasattr(td_mod, "flag2")
        with composite.prepare(td_mod):
            assert getattr(td_mod, "flag1") == 1
            assert getattr(td_mod, "flag2") == 1
        assert not hasattr(td_mod, "flag1")
        assert not hasattr(td_mod, "flag2")

    def test_composite_raises_on_bad_spawn_override(self, default_test_model):
        """Composite rejects contexts overriding _spawn_hooked_module."""
        with pytest.raises(ValueError):
            CompositeHookingContextFactory(BadSpawnFactory())
