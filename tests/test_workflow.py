import multiprocessing as mp
from contextlib import ExitStack

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch import nn

from tdhook.methods import BoundMethod, Method
from tdhook.execution import AutogradLifetime, ExecutionSpec, GradientMode
from tdhook.latent import ActivationCaching
from tdhook.modules import BoundModule
from tdhook.runtime import BoundHookProgram, HookProgramBuilder, HookSpec
from tdhook.targets import Target
from tdhook.workflow import Workflow, WorkflowHandoffError, WorkflowUpdate, _DeferredAutogradCleanup


class CountingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return self.model(value)


class CaptureOutput(Method):
    def __init__(self):
        super().__init__()
        self.values = []

    def _install_hooks(self, module: BoundModule) -> BoundHookProgram:
        def capture(_module, _args, output):
            self.values.append(output.detach())

        with HookProgramBuilder() as builder:
            builder.register_path(
                module.hook_root,
                capture,
                HookSpec("", "capture", "fwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class ReplaceOutput(Method):
    def _install_hooks(self, module: BoundModule) -> BoundHookProgram:
        def replace(_module, _args, output):
            return output + 1

        with HookProgramBuilder() as builder:
            builder.register_path(
                module.hook_root,
                replace,
                HookSpec("", "replace", "fwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class BackwardCapture(Method):
    def __init__(self, *, fail=False):
        super().__init__()
        self.values = []
        self.fail = fail

    @property
    def execution_spec(self):
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED, autograd_lifetime=AutogradLifetime.BACKWARD)

    def _install_hooks(self, module: BoundModule) -> BoundHookProgram:
        def capture(_module, _grad_input, grad_output):
            self.values.append(grad_output[0].detach())
            if self.fail:
                raise RuntimeError("backward hook failed")

        with HookProgramBuilder() as builder:
            builder.register_path(
                module.hook_root,
                capture,
                HookSpec("", "capture", "bwd"),
                relative_path=module.relative_path,
            )
            return builder.build()


class InvalidOutput(TensorDictModuleBase):
    in_keys = ["input"]
    out_keys = ["output"]

    def forward(self, data):
        return "not a tensordict"


class DeferredInvalidMethod(Method):
    def __init__(self):
        super().__init__()
        self.context = None

    @property
    def execution_spec(self):
        return ExecutionSpec(gradient_mode=GradientMode.REQUIRED, autograd_lifetime=AutogradLifetime.BACKWARD)

    def bind(self, *args, **kwargs):
        self.context = super().bind(*args, **kwargs)
        return self.context

    def _bind_module(self, module, in_keys, out_keys, extra_relative_path):
        return InvalidOutput()


class PublishingModule(BoundModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_keys = [*self.out_keys, "published"]

    def finalize_tensordict(self, data):
        return data.set("published", torch.ones(*data.batch_size))


class PublishingMethod(Method):
    _bound_module_class = PublishingModule


class HandoffMutation(TensorDictModuleBase):
    in_keys = ["input"]
    out_keys = ["output"]

    def __init__(self, mutation):
        super().__init__()
        self.mutation = mutation

    def forward(self, data):
        if self.mutation == "metadata":
            data.batch_size = []
        elif self.mutation == "keys":
            data.set("extra", torch.zeros(2))
        elif self.mutation == "non_tensor":
            data.set("output", "invalid")
        return data


def _double(value):
    return value * 2


def _run_handoff_workflow(data, results, release):
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=[("result", "doubled")])
    result = Workflow(operator)(nn.Identity(), data)
    results.put(result)
    release.wait(timeout=30)


@pytest.mark.parametrize("storage", ["shared", "consolidated"])
def test_workflow_preserves_native_handoff_storage_across_processes(storage):
    context = mp.get_context("spawn")
    data = TensorDict(
        {"input": torch.ones(2), "result": {"doubled": torch.zeros(2)}},
        batch_size=[2],
        device="cpu",
    )
    if storage == "shared":
        data.share_memory_()
    else:
        data = data.consolidate(metadata=True)

    results = context.Queue()
    release = context.Event()
    process = context.Process(target=_run_handoff_workflow, args=(data, results, release))
    process.start()
    try:
        result = results.get(timeout=30)
        assert result.batch_size == torch.Size([2])
        assert result.device == torch.device("cpu")
        assert set(result.keys(include_nested=True, leaves_only=True)) == {"input", ("result", "doubled")}
        assert result.is_shared() is (storage == "shared")
        assert result.is_consolidated() is (storage == "consolidated")
        assert torch.equal(result["result", "doubled"], torch.full((2,), 2.0))
        if storage == "shared":
            assert torch.equal(data["result", "doubled"], torch.full((2,), 2.0))
    finally:
        release.set()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
    assert process.exitcode == 0


def test_workflow_handoff_requires_preallocated_outputs():
    data = TensorDict({"input": torch.ones(2)}, batch_size=[2]).share_memory_()
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=[("result", "doubled")])

    with pytest.raises(WorkflowHandoffError, match="preallocated.*result.*doubled"):
        Workflow(operator)(nn.Identity(), data)


def test_workflow_handoff_rejects_incompatible_output_metadata():
    data = TensorDict(
        {"input": torch.ones(2), "output": torch.zeros(2, dtype=torch.int64)}, batch_size=[2]
    ).share_memory_()
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=["output"])

    with pytest.raises(WorkflowHandoffError, match="shape/dtype/device"):
        Workflow(operator)(nn.Identity(), data)


def test_workflow_handoff_rejects_autograd_inputs():
    data = TensorDict(
        {"input": torch.ones(2, requires_grad=True), "output": torch.zeros(2)}, batch_size=[2]
    ).share_memory_()
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=["output"])

    with pytest.raises(WorkflowHandoffError, match="requires gradients.*detach"):
        Workflow(operator)(nn.Identity(), data)


def test_workflow_handoff_rejects_non_tensor_artifacts():
    data = TensorDict({"label": "unsafe"}, batch_size=[]).consolidate()

    with pytest.raises(WorkflowHandoffError, match="label.*must contain a Tensor"):
        Workflow()(nn.Identity(), data)


def test_workflow_handoff_rejects_non_cpu_artifacts(monkeypatch):
    data = TensorDict({"input": torch.empty(2, device="meta")}, batch_size=[2])
    monkeypatch.setattr(TensorDict, "is_shared", lambda _self: True)

    with pytest.raises(WorkflowHandoffError, match="local process handoff requires CPU"):
        Workflow()(nn.Identity(), data)


def test_workflow_handoff_rejects_deferred_backward(default_test_model):
    data = TensorDict({"input": torch.ones(2, 10), "output": torch.zeros(2, 5)}, batch_size=[2]).share_memory_()

    with pytest.raises(WorkflowHandoffError, match="deferred backward"):
        Workflow(BackwardCapture())(default_test_model, data)


def test_workflow_handoff_requires_in_place_step_semantics():
    data = TensorDict({"input": torch.ones(2), "output": torch.zeros(2)}, batch_size=[2]).share_memory_()
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=["output"], inplace=False)

    with pytest.raises(WorkflowHandoffError, match="must mutate.*in place"):
        Workflow(operator)(nn.Identity(), data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("metadata", "changed artifact batch size or device metadata"),
        ("keys", "changed artifact keys"),
        ("non_tensor", "output.*must remain a Tensor"),
    ],
)
def test_workflow_handoff_rejects_unsafe_step_mutations(mutation, message):
    data = TensorDict({"input": torch.ones(2), "output": torch.zeros(2)}, batch_size=[2]).share_memory_()

    with pytest.raises(WorkflowHandoffError, match=message):
        Workflow(HandoffMutation(mutation))(nn.Identity(), data)


def test_workflow_handoff_ignores_discarded_operator_outputs():
    data = TensorDict({"input": torch.ones(2)}, batch_size=[2]).share_memory_()
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=["_"])

    result = Workflow(operator)(nn.Identity(), data)

    assert result is data
    assert result.is_shared()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("record", "without storage metadata"),
        ("storage", "storage metadata is unavailable"),
        ("nested", "metadata is missing output"),
        ("leaf", "metadata is missing output"),
    ],
)
def test_workflow_handoff_rejects_malformed_consolidated_metadata(corruption, message):
    data = TensorDict({"input": torch.ones(2), "result": {"doubled": torch.zeros(2)}}, batch_size=[2]).consolidate()
    if corruption == "record":
        data._consolidated = None
    elif corruption == "storage":
        data._consolidated = {"metadata": {}}
    elif corruption == "nested":
        data._consolidated["metadata"]["result"] = None
    else:
        del data._consolidated["metadata"]["result"]["leaves"]["doubled"]
    operator = TensorDictModule(_double, in_keys=["input"], out_keys=[("result", "doubled")])

    with pytest.raises(WorkflowHandoffError, match=message):
        Workflow(operator)(nn.Identity(), data)


def test_workflow_handoff_detects_final_storage_changes():
    data = TensorDict({"input": torch.ones(2), "output": torch.zeros(2)}, batch_size=[2]).share_memory_()

    def unlock_artifact(value):
        data.unlock_()
        return value * 2

    operator = TensorDictModule(unlock_artifact, in_keys=["input"], out_keys=["output"])

    with pytest.raises(WorkflowHandoffError, match="changed native handoff artifact storage"):
        Workflow(operator)(nn.Identity(), data)


def test_workflow_method_preserves_shared_artifact_storage(default_test_model):
    data = TensorDict(
        {"input": torch.ones(2, 10), "output": torch.zeros(2, 5)}, batch_size=[2], device="cpu"
    ).share_memory_()

    result = Workflow(CaptureOutput())(default_test_model, data)

    assert result is data
    assert result.is_shared()
    assert result.device == torch.device("cpu")
    assert result.batch_size == torch.Size([2])
    assert torch.count_nonzero(result["output"]) > 0


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
    workflow = Workflow(Method(), summarise)
    data = TensorDict({"source": {"input": torch.ones(2, 10)}}, batch_size=[2])

    result = workflow(model, data)

    assert result["prediction"].shape == (2, 5)
    assert result["summary", "mean"].shape == (2,)


def test_workflow_executes_methods_sequentially_and_publishes_each_cache(default_test_model):
    model = CountingModel(default_test_model)
    first = ActivationCaching("model.linear1", cache_key=("activations", "first"))
    second = ActivationCaching("model.linear2", cache_key=("activations", "second"))
    workflow = Workflow(first, second)
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    result = workflow(model, data)

    assert model.calls == 2
    assert result["activations", "first"]["model.linear1"].shape == (2, 20)
    assert result["activations", "second"]["model.linear2"].shape == (2, 20)


def test_workflow_executes_a_targeted_activation_capture(default_test_model):
    target = Target("linear2", "activation", -1, (0, 2))
    workflow = Workflow(ActivationCaching(target, cache_key=("activations", "selected")))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    result = workflow(default_test_model, data)

    assert result["activations", "selected", "linear2"].shape == (2, 2)


def test_method_publication_contract_applies_to_standalone_and_workflow_execution(default_test_model):
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    method = PublishingMethod()

    with method.bind(default_test_model) as prepared:
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
        workflow(default_test_model, data)


def test_workflow_update_explicitly_allows_owned_output_replacement(default_test_model):
    first = ActivationCaching("linear1", cache_key=("activations", "shared"))
    second = ActivationCaching("linear2", cache_key=("activations", "shared"))
    workflow = Workflow(first, WorkflowUpdate(second))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    result = workflow(default_test_model, data)
    assert "linear2" in result["activations", "shared"]
    assert "linear1" not in result["activations", "shared"]


def test_workflow_rejects_ancestor_output_collisions_before_execution(default_test_model):
    first = TensorDictModule(lambda value: value, in_keys=["input"], out_keys=[("results", "first")])
    second = TensorDictModule(lambda value: value, in_keys=["input"], out_keys=["results"])
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="overlaps earlier workflow-owned outputs"):
        Workflow(first, second)(default_test_model, data)


def test_workflow_rejects_undeclared_externally_driven_backward_capture(default_test_model):
    workflow = Workflow(ActivationCaching("linear1", directions=["bwd"]))
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="backward hooks"):
        workflow(default_test_model, data)


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


def test_workflow_rejects_deferred_backward_before_a_later_model_execution(default_test_model):
    workflow = Workflow(BackwardCapture(), CaptureOutput())
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(ValueError, match="cannot precede another model-executing method"):
        workflow(default_test_model, data)

    assert all(not module._backward_hooks for module in default_test_model.modules())


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
        workflow(model, data)
    except ValueError:
        pass
    else:
        raise AssertionError("workflow accepted a missing TensorDict dependency")

    assert model.calls == 0
    assert capture.values == []
    captured = len(capture.values)
    model(torch.ones(2, 10))
    assert len(capture.values) == captured


def test_workflow_dependency_preflight_does_not_clear_a_method_cache(default_test_model):
    cache = TensorDict({"sentinel": torch.ones(1)}, batch_size=[])
    method = ActivationCaching("linear1", cache=cache)

    with pytest.raises(ValueError, match="missing TensorDict keys"):
        Workflow(method)(default_test_model, TensorDict(batch_size=[]))

    assert list(cache.keys()) == ["sentinel"]
    torch.testing.assert_close(cache["sentinel"], torch.ones(1))


def test_workflow_validates_public_boundary_types(default_test_model):
    try:
        Workflow(object())
    except TypeError as error:
        assert "configured method or TensorDictModuleBase" in str(error)
    else:
        raise AssertionError("workflow accepted an invalid step")

    workflow = Workflow()
    try:
        workflow.run(default_test_model, object())
    except TypeError as error:
        assert "must be a TensorDict" in str(error)
    else:
        raise AssertionError("workflow accepted invalid data")
    try:
        workflow.run(object(), TensorDict())
    except TypeError as error:
        assert "torch.nn.Module" in str(error)
    else:
        raise AssertionError("workflow accepted an invalid model")


def test_workflow_method_does_not_require_optional_metadata(default_test_model):
    class ExecutableMethod:
        @property
        def execution_spec(self):
            return ExecutionSpec()

        def bind(self, model):
            return Method().bind(model)

    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    result = Workflow(ExecutableMethod())(default_test_model, data)

    assert result["output"].shape == (2, 5)


def test_workflow_rejects_invalid_method_protocol_results(default_test_model):
    class InvalidSpec(Method):
        @property
        def execution_spec(self):
            return object()

    class InvalidContext(Method):
        def bind(self, model):
            return object()

    class NonModuleContext(BoundMethod):
        def __enter__(self):
            self._in_context = True
            return object()

    class InvalidPrepared(Method):
        _binding_class = NonModuleContext

    class MissingBindingContext(BoundMethod):
        def __enter__(self):
            self._in_context = True
            return TensorDictModule(lambda value: value, in_keys=["input"], out_keys=["output"])

    class MissingBinding(Method):
        _binding_class = MissingBindingContext

    class NonModuleContractContext(BoundMethod):
        @property
        def module(self):
            return object()

    class NonModuleContract(Method):
        _binding_class = NonModuleContractContext

    class InvalidContractContext(BoundMethod):
        @property
        def module(self):
            return TensorDictModule(lambda value: value, in_keys=["input"], out_keys=["output"])

    class InvalidContract(Method):
        _binding_class = InvalidContractContext

    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    cases = (
        (InvalidSpec(), "ExecutionSpec"),
        (InvalidContext(), "BoundMethod"),
        (InvalidPrepared(), "TensorDictModuleBase"),
        (MissingBinding(), "invalid bound module"),
        (NonModuleContract(), "TensorDictModuleBase"),
        (InvalidContract(), "invalid bound module"),
    )
    for method, message in cases:
        try:
            Workflow(method)(default_test_model, data)
        except TypeError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"workflow accepted {type(method).__name__}")


def test_workflow_rejects_non_tensordict_step_results(default_test_model):
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    class InvalidMethod(Method):
        def _bind_module(self, module, in_keys, out_keys, extra_relative_path):
            return InvalidOutput()

    for step, message in ((InvalidOutput(), "operator"), (InvalidMethod(), "method execution")):
        try:
            Workflow(step)(default_test_model, data)
        except TypeError as error:
            assert message in str(error)
        else:
            raise AssertionError("workflow accepted a non-TensorDict result")


def test_deferred_method_non_tensordict_result_closes_its_context(default_test_model):
    method = DeferredInvalidMethod()
    data = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])

    with pytest.raises(TypeError, match="method execution"):
        Workflow(method)(default_test_model, data)

    assert method.context is not None
    assert not method.context._in_context
