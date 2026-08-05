import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from tdhook.contexts import HookingContextFactory
from tdhook.latent import ActivationCaching
from tdhook.modules import HookedModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook.workflow import Workflow


class CountingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return self.model(value)


class CaptureOutput(HookingContextFactory):
    def __init__(self):
        super().__init__()
        self.values = []

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        def capture(_module, _args, output):
            self.values.append(output.detach())

        with HookProgramBuilder() as builder:
            builder.register_path(
                module,
                capture,
                HookSpec("", "capture", "fwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class ReplaceOutput(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        def replace(_module, _args, output):
            return output + 1

        with HookProgramBuilder() as builder:
            builder.register_path(
                module,
                replace,
                HookSpec("", "replace", "fwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


def test_workflow_composes_a_method_and_native_tensordict_operator(default_test_model):
    model = TensorDictModule(
        default_test_model,
        in_keys=[("source", "input")],
        out_keys=["prediction"],
    )
    summarise = TensorDictModule(
        lambda prediction: prediction.mean(-1),
        in_keys=["prediction"],
        out_keys=[("summary", "mean")],
    )
    workflow = Workflow(HookingContextFactory(), summarise)
    data = TensorDict({"source": {"input": torch.ones(2, 10)}}, batch_size=[2])

    plan = workflow.plan(model, data)
    result = workflow(model, data)

    assert plan.model_passes == 1
    assert [execution.kind for execution in plan.executions] == ["method", "operator"]
    assert plan.executions[0].in_keys == (("source", "input"),)
    assert plan.executions[1].out_keys == (("summary", "mean"),)
    assert result["prediction"].shape == (2, 5)
    assert result["summary", "mean"].shape == (2,)


def test_workflow_coexecutes_real_activation_methods_and_publishes_each_cache(default_test_model):
    model = CountingModel(default_test_model)
    first = ActivationCaching("model.linear1", cache_key=("activations", "first"))
    second = ActivationCaching("model.linear2", cache_key=("activations", "second"))
    workflow = Workflow(first, second)
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    plan = workflow.plan(model, data)
    result = workflow(model, data)

    assert plan.model_passes == 1
    assert plan.executions[0].coexecuted
    assert plan.compatibility[0].compatible
    assert plan.compatibility[0].reason == "bound read-only capture programs are compatible"
    assert model.calls == 1
    assert plan.executions[0].out_keys == (
        "output",
        ("activations", "first"),
        ("activations", "second"),
    )
    assert result["activations", "first"]["model.linear1"].shape == (2, 20)
    assert result["activations", "second"]["model.linear2"].shape == (2, 20)


def test_workflow_splits_mutating_programs_and_explains_why(default_test_model):
    model = CountingModel(default_test_model)
    workflow = Workflow(CaptureOutput(), ReplaceOutput())
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    plan = workflow.plan(model, data)
    workflow(model, data)

    assert plan.model_passes == 2
    assert not plan.compatibility[0].compatible
    assert "read-only capture" in plan.compatibility[0].reason
    assert model.calls == 2


def test_workflow_rejects_missing_native_dependencies_before_model_execution(default_test_model):
    model = CountingModel(default_test_model)
    operator = TensorDictModule(lambda missing: missing + 1, in_keys=["missing"], out_keys=["prepared"])
    workflow = Workflow(operator, CaptureOutput())
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    try:
        workflow(model, data)
    except ValueError as error:
        assert "missing TensorDict keys" in str(error)
        assert "missing" in str(error)
    else:
        raise AssertionError("workflow accepted a missing TensorDict dependency")

    assert model.calls == 0


def test_workflow_binding_restores_hooks_when_later_validation_fails(default_test_model):
    model = CountingModel(default_test_model)
    capture = CaptureOutput()
    missing = TensorDictModule(lambda value: value, in_keys=["absent"], out_keys=["unused"])
    workflow = Workflow(capture, missing)
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    try:
        workflow.plan(model, data)
    except ValueError:
        pass
    else:
        raise AssertionError("workflow accepted a missing TensorDict dependency")

    model(torch.ones(2, 10))
    assert capture.values == []


def test_workflow_planning_does_not_clear_method_execution_state(default_test_model):
    cache = TensorDict({"existing": torch.ones(2)}, batch_size=[2])
    workflow = Workflow(ActivationCaching("linear1", cache=cache, clear_cache=True))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    workflow.plan(default_test_model, data)

    assert "existing" in cache
