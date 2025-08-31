"""
Tests for the module functionality.
"""

import torch
from tensordict import TensorDict
import pytest

from nnsight import NNsight

from tdhook.modules import ModuleCallWithCache, IntermediateKeysCleaner
from tdhook.modules import HookedModule


class TestHookedModule:
    """Test the HookedModule class."""

    def test_hooked_module_creation(self, default_test_model):
        """Test creating a HookedModule from a regular model."""
        hooked_module = HookedModule.from_module(default_test_model, in_keys=["input"], out_keys=["output"])
        td_output = hooked_module(TensorDict({"input": torch.randn(2, 3, 10)}, batch_size=[2, 3]))
        assert td_output["output"].shape == (2, 3, 5)
        assert torch.allclose(td_output["output"], default_test_model(td_output["input"]))

    def test_hooked_module_run(self, default_test_model):
        """Test creating a HookedModuleRun."""
        hooked_module = HookedModule.from_module(default_test_model, in_keys=["input"], out_keys=["output"])
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with hooked_module.run(data):
            pass

        assert data["output"].shape == (2, 5)

    def test_cache_proxy(self, default_test_model):
        """Test the CacheProxy class."""
        hooked_module = HookedModule.from_module(default_test_model, in_keys=["input"], out_keys=["output"])
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with hooked_module.run(data) as run:
            proxy = run.get("linear3")
            assert proxy.resolve() is None

        assert torch.allclose(proxy.resolve(), data["output"])


class TestHookedModuleGetSet:
    """Test the HookedModule get/set functionality."""

    def test_get(self, get_model):
        """Test getting a value."""
        nnsight_model = NNsight(get_model())
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10)

        with nnsight_model.trace(input_data):
            nnsight_state = nnsight_model.linear2.output.save()

        save_cache = TensorDict()
        with hooked_module.run(TensorDict({"input": input_data}), cache=save_cache) as run:
            run_proxy = run.get("linear2")
            save_proxy = run.save("linear2")

        cache = TensorDict()
        handle, proxy = hooked_module.get(cache, "linear2")
        hooked_module(TensorDict({"input": input_data}))
        handle.remove()

        assert run_proxy.resolve() is run.cache["linear2_output"]
        assert save_proxy.resolve() is save_cache["run.linear2_output"]
        assert save_proxy.resolve() is run_proxy.resolve()
        assert proxy.resolve() is cache["linear2_output"]

        assert torch.allclose(run_proxy.resolve(), nnsight_state)
        assert torch.allclose(proxy.resolve(), nnsight_state)

    def test_set(self, get_model):
        """Test setting a value."""

        nnsight_model = NNsight(get_model())
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10)
        intervention_data = torch.randn(2, 20)

        original_output = nnsight_model(input_data)

        with nnsight_model.trace(input_data):
            nnsight_model.linear2.output[:] = intervention_data
            nnsight_output = nnsight_model.output.save()

        shuttle_ctx = TensorDict({"input": input_data})
        with hooked_module.run(shuttle_ctx) as run:
            run.set("linear2", intervention_data)

        handle = hooked_module.set("linear2", intervention_data)
        shuttle_out_ctx = TensorDict({"input": input_data})
        hooked_module(shuttle_out_ctx)
        handle.remove()

        shuttle_after = hooked_module(TensorDict({"input": input_data}))

        assert torch.allclose(shuttle_ctx["output"], nnsight_output)
        assert torch.allclose(shuttle_out_ctx["output"], nnsight_output)
        assert torch.allclose(shuttle_after["output"], original_output)

    def test_get_input(self, get_model):
        """Test getting and saving input (pre-forward) values."""
        nnsight_model = NNsight(get_model())
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10)

        with nnsight_model.trace(input_data):
            nnsight_input = nnsight_model.linear2.input.save()

        save_cache = TensorDict()
        with hooked_module.run(TensorDict({"input": input_data}), cache=save_cache) as run:
            run_proxy = run.get_input("linear2", callback=lambda **kwargs: kwargs["args"][0])
            save_proxy = run.save_input("linear2", callback=lambda **kwargs: kwargs["args"][0])

        cache = TensorDict()
        handle, proxy = hooked_module.get(
            cache, "linear2", direction="fwd_pre", callback=lambda **kwargs: kwargs["args"][0]
        )
        hooked_module(TensorDict({"input": input_data}))
        handle.remove()

        assert torch.allclose(run_proxy.resolve(), nnsight_input)
        assert torch.allclose(proxy.resolve(), nnsight_input)
        assert run_proxy.resolve() is run.cache["linear2_input"]
        assert save_proxy.resolve() is save_cache["run.linear2_input"]

    def test_set_input(self, get_model):
        """Test setting input via pre-forward hook."""
        nnsight_model = NNsight(get_model())
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10)
        intervention_input = torch.randn(2, 20)

        original_output = nnsight_model(input_data)

        with nnsight_model.trace(input_data):
            nnsight_model.linear2.input[:] = intervention_input
            nnsight_output = nnsight_model.output.save()

        td_ctx = TensorDict({"input": input_data})
        with hooked_module.run(td_ctx) as run:
            # For pre hooks, value should be args tuple
            run.set_input("linear2", (intervention_input,))

        handle = hooked_module.set_input("linear2", (intervention_input,))
        td_ctx2 = TensorDict({"input": input_data})
        hooked_module(td_ctx2)
        handle.remove()

        td_after = hooked_module(TensorDict({"input": input_data}))

        assert torch.allclose(td_ctx["output"], nnsight_output)
        assert torch.allclose(td_ctx2["output"], nnsight_output)
        assert torch.allclose(td_after["output"], original_output)

    def test_get_grad(self, get_model):
        """Test getting and saving grad (backward) values."""
        nnsight_model = NNsight(get_model())
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10, requires_grad=True)

        with nnsight_model.trace(input_data):
            loss = nnsight_model.output.sum()
            loss.backward()
            nnsight_grad = nnsight_model.linear2.input.grad.save()

        save_cache = TensorDict()

        def run_with_backward(m, d):
            out = m(d)["output"]
            out.sum().backward()

        with hooked_module.run(
            TensorDict({"input": input_data.detach().clone().requires_grad_(True)}),
            cache=save_cache,
            grad_enabled=True,
            run_callback=run_with_backward,
        ) as run:
            run_proxy = run.get_grad("linear2", callback=lambda **kwargs: kwargs["grad_input"][0])
            save_proxy = run.save_grad("linear2", callback=lambda **kwargs: kwargs["grad_input"][0])

        cache = TensorDict()
        handle, proxy = hooked_module.get_grad(
            cache,
            "linear2",
            callback=lambda **kwargs: kwargs["grad_input"][0],
        )
        td = TensorDict({"input": input_data.detach().clone().requires_grad_(True)})
        out = hooked_module(td)["output"]
        out.sum().backward()
        handle.remove()

        # For backward hooks, wrapper and direct get use callbacks to return the grad_output tensor
        assert torch.allclose(run_proxy.resolve(), nnsight_grad)
        assert torch.allclose(proxy.resolve(), nnsight_grad)
        assert run_proxy.resolve() is run.cache["linear2_grad_input"]
        assert save_proxy.resolve() is save_cache["run.linear2_grad_input"]

    def test_set_grad(self, get_model):
        """Test setting backward gradients to zero stops upstream grads."""
        model = get_model()
        hooked_module = HookedModule.from_module(model, in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10, requires_grad=True)

        td = TensorDict({"input": input_data.clone().detach().requires_grad_(True)})
        out = hooked_module(td)["output"]
        loss = out.sum()
        loss.backward()
        baseline_grad = model.linear1.weight.grad.clone()
        model.zero_grad(set_to_none=True)

        zero_like = torch.zeros(2, 20)
        td2 = TensorDict({"input": input_data.clone().detach().requires_grad_(True)})

        def run_with_backward(m, d):
            out = m(d)["output"]
            out.sum().backward()

        with hooked_module.run(td2, grad_enabled=True, run_callback=run_with_backward) as run:
            run.set_grad("linear2", (zero_like,))
        grad_after_pre = model.linear1.weight.grad.clone()
        assert torch.allclose(grad_after_pre, torch.zeros_like(grad_after_pre))
        model.zero_grad(set_to_none=True)

        td3 = TensorDict({"input": input_data.clone().detach().requires_grad_(True)})
        with hooked_module.run(td3, grad_enabled=True, run_callback=run_with_backward) as run:
            run.set_grad("linear2", (zero_like,))
        grad_after_full = model.linear1.weight.grad.clone()
        assert torch.allclose(grad_after_full, torch.zeros_like(grad_after_full))
        assert not torch.allclose(baseline_grad, torch.zeros_like(baseline_grad))

    def test_get_grad_output(self, get_model):
        """Test getting and saving grad_input (backward pre-hook) values."""
        nnsight_model = NNsight(get_model())
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10, requires_grad=True)

        with nnsight_model.trace(input_data):
            loss = nnsight_model.output.sum()
            loss.backward()
            nnsight_grad_in = nnsight_model.linear2.output.grad.save()

        save_cache = TensorDict()

        def run_with_backward(m, d):
            out = m(d)["output"]
            out.sum().backward()

        with hooked_module.run(
            TensorDict({"input": input_data.detach().clone().requires_grad_(True)}),
            cache=save_cache,
            grad_enabled=True,
            run_callback=run_with_backward,
        ) as run:
            run_proxy = run.get_grad_output("linear2", callback=lambda **kwargs: kwargs["grad_output"][0])
            save_proxy = run.save_grad_output("linear2", callback=lambda **kwargs: kwargs["grad_output"][0])

        cache = TensorDict()
        handle, proxy = hooked_module.get_grad_output(
            cache,
            "linear2",
            callback=lambda **kwargs: kwargs["grad_output"][0],
        )
        td = TensorDict({"input": input_data.detach().clone().requires_grad_(True)})
        out = hooked_module(td)["output"]
        out.sum().backward()
        handle.remove()

        assert torch.allclose(run_proxy.resolve(), nnsight_grad_in)
        assert torch.allclose(proxy.resolve(), nnsight_grad_in)
        assert run_proxy.resolve() is run.cache["linear2_grad_output"]
        assert save_proxy.resolve() is save_cache["run.linear2_grad_output"]

    def test_set_grad_output(self, get_model):
        model = get_model()
        hooked_module = HookedModule.from_module(model, in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10, requires_grad=True)

        # Baseline: compute grad for linear1.weight
        td = TensorDict({"input": input_data.clone().detach().requires_grad_(True)})
        out = hooked_module(td)["output"]
        loss = out.sum()
        loss.backward()
        baseline_grad = model.linear1.weight.grad.clone()
        model.zero_grad(set_to_none=True)

        # Using bwd_pre: set grad_output at linear2 to zero
        zero_like = torch.zeros(2, 20)
        td2 = TensorDict({"input": input_data.clone().detach().requires_grad_(True)})

        def run_with_backward(m, d):
            out = m(d)["output"]
            out.sum().backward()

        with hooked_module.run(td2, grad_enabled=True, run_callback=run_with_backward) as run:
            run.set_grad_output("linear2", (zero_like,))
        grad_after_pre = model.linear1.weight.grad.clone()
        assert torch.allclose(grad_after_pre, torch.zeros_like(grad_after_pre))
        model.zero_grad(set_to_none=True)

        # Using module-level wrapper: set grad_output at linear2 to zero
        handle = hooked_module.set_grad_output("linear2", (zero_like,))
        td3 = TensorDict({"input": input_data.clone().detach().requires_grad_(True)})
        out = hooked_module(td3)["output"]
        out.sum().backward()
        handle.remove()
        grad_after_handle = model.linear1.weight.grad.clone()
        assert torch.allclose(grad_after_handle, torch.zeros_like(grad_after_handle))
        # Ensure baseline had non-zero grads
        assert not torch.allclose(baseline_grad, torch.zeros_like(baseline_grad))


class TestStopAndErrors:
    def test_stop_prevents_later_layers(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        input_data = torch.randn(2, 10)

        called = False

        def hook(module, args):
            nonlocal called
            called = True

        handle_pre = hooked_module.module.linear3.register_forward_pre_hook(hook)

        with hooked_module.run(TensorDict({"input": input_data})) as run:
            run.stop("linear2")
        handle_pre.remove()

        assert not called

    def test_run_methods_outside_context_raise(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        td = TensorDict({"input": torch.randn(2, 10)})
        run = hooked_module.run(td)

        with pytest.raises(RuntimeError):
            run.get("linear2")
        with pytest.raises(RuntimeError):
            run.save("linear2")
        with pytest.raises(RuntimeError):
            run.set("linear2", torch.randn(2, 20))

    def test_invalid_submodule_path_raises(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        cache = TensorDict()
        with pytest.raises(ValueError):
            hooked_module.get(cache, "module.missing")
        with pytest.raises(ValueError):
            hooked_module.set("module.missing", torch.randn(2, 20))

    def test_invalid_direction_raises(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        cache = TensorDict()
        with pytest.raises((KeyError, ValueError)):
            hooked_module.get(cache, "linear2", direction="invalid")
        with pytest.raises((KeyError, ValueError)):
            hooked_module.set("linear2", torch.randn(2, 20), direction="invalid")

    def test_callback_signature_mismatch_raises(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        cache = TensorDict()
        # fwd_pre expects (module, args); missing 'args' -> error
        with pytest.raises(ValueError):
            hooked_module.get(cache, "linear2", direction="fwd_pre", callback=lambda module: None)
        # bwd expects (module, grad_input, grad_output); missing 'grad_input'/'grad_output' -> error
        with pytest.raises(ValueError):
            hooked_module.get(cache, "linear2", direction="bwd", callback=lambda module: None)

    def test_tuple_caching_without_callback_raises(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        td = TensorDict({"input": torch.randn(2, 10)})
        cache = TensorDict()
        handle, _ = hooked_module.get(cache, "linear2", direction="fwd_pre")
        with pytest.raises((ValueError, RuntimeError)):
            hooked_module(td)
        handle.remove()


class TestAdditionalCoverage:
    def test_run_cache_setter_and_exception_path(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        td = TensorDict({"input": torch.randn(2, 10)})
        outer_cache = TensorDict()
        # exercise cache setter (line 62)
        run = hooked_module.run(td, cache=outer_cache)
        run.cache = TensorDict()  # triggers setter
        # exercise generic exception re-raise path (lines 74-75)
        with pytest.raises(RuntimeError):
            with hooked_module.run(td, run_callback=lambda m, d: (_ for _ in ()).throw(RuntimeError("boom"))):
                pass

    def test_modulelist_warning(self):
        class WithList(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 10)])
                self.linear_out = torch.nn.Linear(10, 5)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.linear_out(x)

        m = WithList()
        hm = HookedModule.from_module(m, in_keys=["input"], out_keys=["output"])
        with pytest.warns(UserWarning):
            handle = hm.register_submodule_hook("layers", lambda module, args, output: output, direction="fwd")
            handle.remove()

    def test_module_wrappers_shortcuts(self, get_model):
        hooked_module = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        x = torch.randn(2, 10, requires_grad=True)
        cache = TensorDict()
        h_in, _ = hooked_module.get_input(cache, "linear2", callback=lambda **kw: kw["args"][0])
        h_setg = hooked_module.set_grad("linear2", (torch.zeros(2, 20),))
        out = hooked_module(TensorDict({"input": x}))["output"]
        out.sum().backward()
        h_in.remove()
        h_setg.remove()

    def test_set_get_grad_output_wrappers(self, get_model):
        model = get_model()
        hooked_module = HookedModule.from_module(model, in_keys=["input"], out_keys=["output"])
        x = torch.randn(2, 10, requires_grad=True)
        h_set = hooked_module.set_grad_output("linear2", (torch.zeros(2, 20),))
        cache = TensorDict()
        h_get, _ = hooked_module.get_grad_output(cache, "linear2", callback=lambda **kw: kw["grad_output"][0])
        out = hooked_module(TensorDict({"input": x}))["output"]
        out.sum().backward()
        h_set.remove()
        h_get.remove()

    def test_forward_guard_and_context_managers(self, get_model):
        # forward() guard (line 302)
        dummy_ctx = type("C", (), {"_in_context": False})()
        hm_guard = HookedModule.from_module(
            get_model(), in_keys=["input"], out_keys=["output"], hooking_context=dummy_ctx
        )
        with pytest.raises(RuntimeError):
            hm_guard(TensorDict({"input": torch.randn(1, 10)}))

        # disable_context_hooks without context (307-310) and disable_context without context (315)
        hm = HookedModule.from_module(get_model(), in_keys=["input"], out_keys=["output"])
        with pytest.raises(RuntimeError):
            with hm.disable_context_hooks():
                pass
        with pytest.raises(RuntimeError):
            with hm.disable_context():
                pass

        # When context exists, both managers work
        from tdhook.contexts import HookingContextFactory

        ctx = HookingContextFactory().prepare(get_model())
        with ctx as ctx_hm:
            # disable hooks context manager
            with ctx_hm.disable_context_hooks():
                _ = ctx_hm(TensorDict({"input": torch.randn(1, 10)}))
            # disable full context manager yields raw module
            with ctx_hm.disable_context() as raw_module:
                assert callable(getattr(raw_module, "forward"))


class TestModuleCallWithCache:
    """Test the ModuleCallWithCache class."""

    def test_module_with_cache_creation(self, default_test_model):
        """Test creating a ModuleCallWithCache with basic functionality."""

        td_module = HookedModule.from_module(module=default_test_model, in_keys=["input"], out_keys=["output"])

        caching_module = ModuleCallWithCache(
            td_module=td_module, cache_key="cache", stored_keys=["linear2"], out_key="outs"
        )

        td_module.get(
            cache=caching_module.cache_ref,
            module_key="linear2",
            callback=lambda **kwargs: kwargs["output"].requires_grad_(True),
        )

        input_data = torch.randn(2, 10)
        input_td = TensorDict({"input": input_data}, batch_size=[2])

        output_td = caching_module(input_td)

        assert "cache" in output_td
        assert "linear2_output" in output_td["cache"]
        assert output_td["cache"]["linear2_output"].shape == (2, 20)

        expected_output = default_test_model(input_data)
        assert torch.allclose(output_td[("outs", "output")], expected_output)


class TestIntermediateKeysCleaner:
    def test_intermediate_keys_cleaner(self):
        """Test the IntermediateKeysCleaner class."""
        td = TensorDict({"a": torch.randn(2, 10), "b": torch.randn(2, 20), "c": torch.randn(2, 30)}, batch_size=[2])
        cleaner = IntermediateKeysCleaner(intermediate_keys=["b"])
        output_td = cleaner(td)
        assert "a" in output_td
        assert "b" not in output_td
        assert "c" in output_td
