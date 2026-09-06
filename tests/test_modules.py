import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MutableWeakRef
from tdhook.modules import (
    FunctionModule,
    HookedModule,
    IntermediateKeysCleaner,
    ModuleCall,
    ModuleCallWithCache,
    PGDModule,
    flatten_select_reshape_call,
)


def test_function_module_and_intermediate_cleaner_are_native_operators():
    module = FunctionModule(lambda data: data.set("output", data["input"] + 1), ["input"], ["output"])
    data = module(TensorDict({"input": torch.ones(2, 3)}, batch_size=[2]))
    cleaned = IntermediateKeysCleaner(["input"])(data)

    assert torch.equal(cleaned["output"], torch.full((2, 3), 2.0))
    assert "input" not in cleaned
    assert "FunctionModule" in repr(module)
    assert "IntermediateKeysCleaner" in repr(IntermediateKeysCleaner(["input"]))


@pytest.mark.parametrize("batch_size", [(), (3,), (2, 3)])
@pytest.mark.parametrize("select", [True, False])
@pytest.mark.parametrize("reshape", [True, False])
def test_flatten_select_reshape_call(batch_size, select, reshape):
    def transform(data):
        assert data["input"].ndim == 2
        return data.set("output", data["input"])

    module = FunctionModule(transform, ["input"], ["output"])
    data = TensorDict({"input": torch.randn(*batch_size, 4)}, batch_size=batch_size)

    result = flatten_select_reshape_call(module, data, select=select, reshape=reshape)

    assert "output" in result
    assert ("input" not in result) is select


def test_module_call_routes_nested_inputs_and_outputs():
    model = TensorDictModule(lambda value: value + 1, in_keys=["value"], out_keys=["result"])
    operator = ModuleCall(model, in_key="source", out_key="prediction")
    data = TensorDict({"source": {"value": torch.ones(2, 3)}}, batch_size=[2])

    result = operator(data)

    assert torch.equal(result["prediction", "result"], torch.full((2, 3), 2.0))
    assert "ModuleCall" in repr(operator)


def test_module_call_with_cache_publishes_runtime_cache():
    cache_ref = MutableWeakRef(TensorDict())

    class CacheWriter(TensorDictModuleBase):
        in_keys = ["input"]
        out_keys = ["output"]

        def forward(self, data):
            cache_ref.resolve().set("hidden", data["input"] * 2)
            return data.set("output", data["input"] + 1)

    operator = ModuleCallWithCache(
        CacheWriter(),
        stored_keys=["hidden"],
        cache_key="cache",
        out_key="prediction",
        cache_ref=cache_ref,
    )
    data = TensorDict({"input": torch.ones(2, 3)}, batch_size=[2])

    result = operator(data)

    assert torch.equal(result["cache", "hidden"], torch.full((2, 3), 2.0))
    assert torch.equal(result["prediction", "output"], torch.full((2, 3), 2.0))
    assert operator.cache_ref is cache_ref
    assert "ModuleCallWithCache" in repr(operator)


def test_pgd_module_updates_and_clamps_working_values():
    class GradientModule(TensorDictModuleBase):
        in_keys = ["value"]
        out_keys = ["value", "_grad"]

        def forward(self, data):
            data.set("_grad", TensorDict({"value": torch.ones_like(data["value"])}, batch_size=data.batch_size))
            return data

    module = PGDModule(GradientModule(), alpha=0.5, n_steps=2, min_value=-0.75, working_key=None, use_sign=False)
    data = TensorDict({"value": torch.zeros(2, 3)}, batch_size=[2])

    result = module(data)

    assert torch.equal(result["value"], torch.full((2, 3), -0.75))
    assert "PGDModule" in repr(module)


def test_bound_module_is_context_owned_and_finalizes_results(default_test_model):
    context = HookingContextFactory().prepare(default_test_model)
    with context as prepared:
        assert isinstance(prepared, HookedModule)
        assert "HookedModule" in repr(prepared)
        result = prepared(TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))
        assert result["output"].shape == (2, 5)

    with pytest.raises(RuntimeError, match="called in context"):
        prepared(TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))


def test_method_outputs_can_be_selected_and_reset_without_changing_model_outputs():
    class PublishingModule(HookedModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.out_keys = [*self.out_keys, ("metrics", "sum")]

        def finalize_tensordict(self, data):
            return data.set(("metrics", "sum"), data["output"].sum(-1))

    class PublishingMethod(HookingContextFactory):
        _hooked_module_class = PublishingModule

    model = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=["output"])
    with PublishingMethod().prepare(model) as prepared:
        assert prepared.out_keys_source == ["output", ("metrics", "sum")]
        prepared.select_out_keys(("metrics", "sum"))
        selected = prepared(TensorDict({"input": torch.ones(2, 3)}, batch_size=[2]))
        assert prepared.out_keys == [("metrics", "sum")]
        assert "output" not in selected
        torch.testing.assert_close(selected["metrics", "sum"], torch.full((2,), 3.0))

        prepared.reset_out_keys()
        assert prepared.out_keys == ["output", ("metrics", "sum")]
        restored = prepared(TensorDict({"input": torch.ones(2, 3)}, batch_size=[2]))
        torch.testing.assert_close(restored["output"], torch.ones(2, 3))

    assert model.out_keys == ["output"]
    assert model.out_keys_source == ["output"]
