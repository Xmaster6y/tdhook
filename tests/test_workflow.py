import pytest
import torch
from contextlib import ExitStack
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch import nn

from tests.composition_conformance import assert_conformance
from tdhook.contexts import HookingContext, HookingContextFactory
from tdhook.execution import AutogradLifetime, ExecutionSpec, GradientMode
from tdhook.latent import ActivationCaching, Probing
from tdhook.modules import HookedModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook.targets import Target
from tdhook.workflow import Workflow, WorkflowUpdate, _DeferredAutogradCleanup


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


class BackwardCapture(HookingContextFactory):
    def __init__(self, *, fail=False):
        super().__init__()
        self.values = []
        self.fail = fail

    @property
    def execution_spec(self):
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED, autograd_lifetime=AutogradLifetime.BACKWARD)

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        def capture(_module, _grad_input, grad_output):
            self.values.append(grad_output[0].detach())
            if self.fail:
                raise RuntimeError("backward hook failed")

        with HookProgramBuilder() as builder:
            builder.register_path(
                module,
                capture,
                HookSpec("", "capture", "bwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class EmptyProgram(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        with HookProgramBuilder() as builder:
            return builder.build()


class InvalidOutput(TensorDictModuleBase):
    in_keys = ["input"]
    out_keys = ["output"]

    def forward(self, data):
        return "not a tensordict"


class PublishingModule(HookedModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_keys = [*self.out_keys, "published"]

    def finalize_tensordict(self, data):
        return data.set("published", torch.ones(*data.batch_size))


class PublishingMethod(HookingContextFactory):
    _hooked_module_class = PublishingModule


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

    assert_conformance(
        "test_workflow_coexecutes_real_activation_methods_and_publishes_each_cache", plan, status="supported"
    )

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


def test_workflow_plans_and_executes_a_targeted_activation_capture(default_test_model):
    target = Target("linear2", "activation", -1, (0, 2))
    workflow = Workflow(ActivationCaching(target, cache_key=("activations", "selected")))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    plan = workflow.plan(default_test_model, data)
    result = workflow(default_test_model, data)

    assert plan.model_passes == 1
    assert plan.executions[0].out_keys == ("output", ("activations", "selected"))
    assert result["activations", "selected", "linear2"].shape == (2, 2)


def test_method_publication_contract_applies_to_standalone_and_workflow_execution(default_test_model):
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    method = PublishingMethod()

    with method.prepare(default_test_model) as prepared:
        standalone = prepared(data.clone())
    workflow_result = Workflow(method)(default_test_model, data.clone())

    assert torch.equal(standalone["published"], torch.ones(2))
    assert torch.equal(workflow_result["published"], torch.ones(2))


def test_workflow_rejects_overlapping_method_owned_outputs_without_explicit_update(default_test_model):
    first = ActivationCaching("linear1", cache_key=("activations", "shared"))
    second = ActivationCaching("linear2", cache_key=("activations", "shared"))
    workflow = Workflow(first, second)
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="WorkflowUpdate"):
        workflow.plan(default_test_model, data)


def test_workflow_update_explicitly_allows_owned_output_replacement(default_test_model):
    first = ActivationCaching("linear1", cache_key=("activations", "shared"))
    second = ActivationCaching("linear2", cache_key=("activations", "shared"))
    workflow = Workflow(first, WorkflowUpdate(second))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    plan = workflow.plan(default_test_model, data)
    result = workflow(default_test_model, data)
    assert plan.model_passes == 2
    assert "output namespaces overlap" in plan.compatibility[0].reason
    assert "linear2" in result["activations", "shared"]
    assert "linear1" not in result["activations", "shared"]


def test_workflow_rejects_ancestor_output_collisions_before_execution(default_test_model):
    first = TensorDictModule(lambda value: value, in_keys=["input"], out_keys=[("results", "first")])
    second = TensorDictModule(lambda value: value, in_keys=["input"], out_keys=["results"])
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="overlaps earlier workflow-owned outputs"):
        Workflow(first, second).plan(default_test_model, data)


def test_workflow_rejects_undeclared_externally_driven_backward_capture(default_test_model):
    workflow = Workflow(ActivationCaching("linear1", directions=["bwd"]))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="backward hooks"):
        workflow.plan(default_test_model, data)


def test_workflow_keeps_deferred_backward_hooks_until_backward_completes(default_test_model):
    method = BackwardCapture()
    result = Workflow(method)(default_test_model, TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))

    assert any(module._backward_hooks for module in default_test_model.modules())
    result["output"].sum().backward()

    assert method.values
    assert all(not module._backward_hooks for module in default_test_model.modules())


def test_workflow_cleans_deferred_backward_hooks_when_backward_fails(default_test_model):
    method = BackwardCapture(fail=True)
    result = Workflow(method)(default_test_model, TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))

    with pytest.raises(RuntimeError, match="backward hook failed"):
        result["output"].sum().backward()

    assert all(not module._backward_hooks for module in default_test_model.modules())


def test_workflow_rejects_deferred_backward_without_an_autograd_output():
    method = BackwardCapture()
    model = nn.Identity()
    with pytest.raises(RuntimeError, match="autograd-enabled model output"):
        Workflow(method)(model, TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))

    assert not model._backward_hooks


def test_deferred_cleanup_preserves_first_cleanup_error_and_is_idempotent():
    class FailingHandle:
        def remove(self):
            raise RuntimeError("handle cleanup failed")

    def fail_stack_cleanup():
        raise RuntimeError("stack cleanup failed")

    stack = ExitStack()
    stack.callback(fail_stack_cleanup)
    cleanup = _DeferredAutogradCleanup(stack)
    cleanup._handles = [FailingHandle()]

    with pytest.raises(RuntimeError, match="handle cleanup failed"):
        cleanup.close()
    cleanup.close()


def test_probing_inspection_is_inert_and_declares_additional_keys(default_test_model):
    created = []

    class Probe:
        def step(self, data, **kwargs):
            return None

    def factory(name, direction):
        created.append((name, direction))
        return Probe()

    method = Probing("linear1", factory, additional_keys=["labels", "step_type"])
    workflow = Workflow(method)
    data = TensorDict(
        {"input": torch.ones(2, 10), "labels": torch.ones(2), "step_type": "fit"},
        batch_size=[2],
    )

    plan = workflow.plan(default_test_model, data)

    assert created == []
    assert plan.executions[0].in_keys == ("input", "labels", "step_type")
    with pytest.raises(ValueError, match="step_type"):
        workflow.plan(default_test_model, data.exclude("step_type"))

    workflow(default_test_model, data)
    assert created == [("linear1", "fwd")]


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


def test_workflow_rechecks_a_declared_namespace_before_the_consumer_runs():
    consumed = []

    def consume(value):
        consumed.append(value)
        return value

    producer = TensorDictModule(lambda value: value, in_keys=["input"], out_keys=["namespace"])
    consumer = TensorDictModule(consume, in_keys=[("namespace", "leaf")], out_keys=["result"])
    workflow = Workflow(producer, consumer)
    data = TensorDict({"input": torch.ones(2)}, batch_size=[2])

    workflow.plan(nn.Identity(), data)
    try:
        workflow(nn.Identity(), data)
    except ValueError as error:
        assert "namespace" in str(error) and "leaf" in str(error)
    else:
        raise AssertionError("workflow accepted a missing runtime namespace leaf")

    assert consumed == []


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


def test_workflow_validates_public_boundary_types(default_test_model):
    try:
        Workflow(object())
    except TypeError as error:
        assert "configured method or TensorDictModuleBase" in str(error)
    else:
        raise AssertionError("workflow accepted an invalid step")

    workflow = Workflow()
    for operation in (workflow.plan, workflow.run):
        try:
            operation(default_test_model, object())
        except TypeError as error:
            assert "must be a TensorDict" in str(error)
        else:
            raise AssertionError("workflow accepted invalid data")
    try:
        workflow.plan(object(), TensorDict())
    except TypeError as error:
        assert "torch.nn.Module" in str(error)
    else:
        raise AssertionError("workflow accepted an invalid model")


def test_workflow_rejects_invalid_method_protocol_results(default_test_model):
    class InvalidSpec(HookingContextFactory):
        @property
        def execution_spec(self):
            return object()

    class InvalidContext(HookingContextFactory):
        def prepare(self, model):
            return object()

    class NonModuleContext(HookingContext):
        def _enter(self, *, for_inspection=False):
            self._in_context = True
            return object()

    class InvalidPrepared(HookingContextFactory):
        _hooking_context_class = NonModuleContext

    class MissingBindingContext(HookingContext):
        def _enter(self, *, for_inspection=False):
            self._in_context = True
            return TensorDictModule(lambda value: value, in_keys=["input"], out_keys=["output"])

    class MissingBinding(HookingContextFactory):
        _hooking_context_class = MissingBindingContext

    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    cases = (
        (InvalidSpec(), "ExecutionSpec"),
        (InvalidContext(), "HookingContext"),
        (InvalidPrepared(), "TensorDictModuleBase"),
        (MissingBinding(), "MethodBinding"),
    )
    for method, message in cases:
        try:
            Workflow(method).plan(default_test_model, data)
        except TypeError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"workflow accepted {type(method).__name__}")


def test_workflow_explains_each_unproven_coexecution_case(default_test_model):
    class TwoPassCapture(CaptureOutput):
        @property
        def execution_spec(self):
            return ExecutionSpec(model_passes=2)

    class GradientCapture(CaptureOutput):
        @property
        def execution_spec(self):
            return ExecutionSpec(gradient_mode=GradientMode.REQUIRED)

    class DeferredGradientCapture(GradientCapture):
        @property
        def execution_spec(self):
            return ExecutionSpec(gradient_mode=GradientMode.REQUIRED, autograd_lifetime=AutogradLifetime.BACKWARD)

    class IndirectContext(HookingContext):
        @property
        def executes_model_directly(self):
            return False

    class IndirectCapture(CaptureOutput):
        _hooking_context_class = IndirectContext

    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    cases = (
        ((CaptureOutput(), TwoPassCapture()), "exactly one model pass"),
        ((CaptureOutput(), GradientCapture()), "different autograd modes"),
        ((GradientCapture(), DeferredGradientCapture()), "different autograd lifetimes"),
        ((CaptureOutput(), HookingContextFactory()), "did not expose"),
        ((CaptureOutput(), EmptyProgram()), "empty hook program"),
        ((CaptureOutput(), IndirectCapture()), "transforms model execution"),
    )
    for methods, message in cases:
        plan = Workflow(*methods).plan(default_test_model, data)
        assert not plan.compatibility[0].compatible
        assert message in plan.compatibility[0].reason


def test_workflow_rejects_non_tensordict_step_results(default_test_model):
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    class InvalidMethod(HookingContextFactory):
        def _prepare_module(self, module, in_keys, out_keys, extra_relative_path):
            return InvalidOutput()

    for step, message in ((InvalidOutput(), "operator"), (InvalidMethod(), "method execution")):
        try:
            Workflow(step)(default_test_model, data)
        except TypeError as error:
            assert message in str(error)
        else:
            raise AssertionError("workflow accepted a non-TensorDict result")


def test_workflow_splits_methods_bound_to_different_model_signatures():
    model = TensorDictModule(
        lambda left, right: (left, right),
        in_keys=["left", "right"],
        out_keys=["left_output", "right_output"],
    )

    class SignatureContext(HookingContext):
        def __init__(self, *args, selected_in, selected_out, **kwargs):
            super().__init__(*args, **kwargs)
            self._in_keys = [selected_in]
            self._out_keys = [selected_out]

    class SelectedSignature(CaptureOutput):
        _hooking_context_class = SignatureContext

        def __init__(self, in_key, out_key):
            super().__init__()
            self._hooking_context_kwargs = {"selected_in": in_key, "selected_out": out_key}

    data = TensorDict({"left": torch.ones(2), "right": torch.ones(2)}, batch_size=[2])
    plan = Workflow(
        SelectedSignature("left", "left_output"),
        SelectedSignature("right", "right_output"),
    ).plan(model, data)

    assert not plan.compatibility[0].compatible
    assert "different model TensorDict signatures" in plan.compatibility[0].reason


def test_workflow_rejects_method_facts_that_change_after_inspection(default_test_model):
    class FlakyMethod(HookingContextFactory):
        def __init__(self):
            super().__init__()
            self.bindings = 0

        def _prepare_module(self, module, in_keys, out_keys, extra_relative_path):
            self.bindings += 1
            if self.bindings == 1:
                return module
            return TensorDictModule(lambda value: value, in_keys=in_keys, out_keys=["changed"])

    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    try:
        Workflow(FlakyMethod())(default_test_model, data)
    except RuntimeError as error:
        assert "changed after planning" in str(error)
    else:
        raise AssertionError("workflow executed rebound facts that differed from inspection")
