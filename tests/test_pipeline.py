import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from tdhook.contexts import HookingContextFactory
from tdhook.artifacts import ArtifactContract
from tdhook.hooks import MultiHookHandle
from tdhook.modules import HookedModule
from tdhook.pipeline import MethodStage, Pipeline, Stage, TransformStage


class AddOne(HookingContextFactory):
    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        return module.register_submodule_hook("", lambda module, args, output: output + 1, direction="fwd")


def test_method_then_transform(default_test_model):
    artifacts = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
    pipeline = Pipeline(
        [
            MethodStage("predict", AddOne(), required_keys=["input"], provided_keys=["output"]),
            TransformStage(
                "summarise",
                lambda td: td.set("summary", td["output"].mean(-1)),
                required_keys=["output"],
                provided_keys=["summary"],
            ),
        ]
    )

    result = pipeline.run(default_test_model, artifacts)

    assert torch.allclose(result.artifacts["output"], default_test_model(torch.ones(2, 10)) + 1)
    assert result.artifacts["summary"].shape == (2,)
    assert [stage.name for stage in result.stages] == ["predict", "summarise"]


def test_transform_then_method_with_nested_key(default_test_model):
    artifacts = TensorDict({"source": {"x": torch.ones(2, 10)}}, batch_size=[2])
    pipeline = Pipeline(
        [
            TransformStage(
                "prepare",
                lambda td: td.set("input", td["source", "x"] * 2),
                required_keys=[("source", "x")],
                provided_keys=["input"],
            ),
            MethodStage("predict", HookingContextFactory(), required_keys=["input"], provided_keys=["output"]),
        ]
    )

    result = pipeline.run(default_test_model, artifacts)

    assert torch.allclose(result.artifacts["output"], default_test_model(torch.ones(2, 10) * 2))


def test_method_stage_keeps_artifact_contract_separate_from_model_signature(default_test_model):
    model = TensorDictModule(default_test_model, in_keys=["model_input"], out_keys=["prediction"])
    artifacts = TensorDict({"model_input": torch.ones(2, 10), "baseline": torch.zeros(2, 10)}, batch_size=[2])
    pipeline = Pipeline(
        [
            MethodStage(
                "predict",
                HookingContextFactory(),
                required_keys=["model_input", "baseline"],
                provided_keys=["prediction"],
                model_in_keys=["model_input"],
                model_out_keys=["prediction"],
            )
        ]
    )

    result = pipeline.run(model, artifacts)

    assert "baseline" in result.artifacts
    assert torch.allclose(result.artifacts["prediction"], default_test_model(artifacts["model_input"]))


def test_public_artifact_contract_and_provenance(default_test_model):
    contract = ArtifactContract(
        requires={"source": ("inputs", "model")}, provides={"prediction": ("outputs", "prediction")}
    )
    pipeline = Pipeline(
        [
            TransformStage(
                "predict",
                lambda td: td.set(("outputs", "prediction"), td[("inputs", "model")] + 1),
                artifact_contract=contract,
            )
        ]
    )

    result = pipeline.run(
        default_test_model,
        TensorDict({"inputs": {"model": torch.ones(2, 10)}}, batch_size=[2]),
        model_id="demo-model-v1",
        seed=7,
        stage_configurations={"predict": {"normalise": False}},
    )

    assert result.artifacts[("outputs", "prediction")].shape == (2, 10)
    assert result.provenance[0].model_id == "demo-model-v1"
    assert result.provenance[0].seed == 7
    assert result.provenance[0].parents == (("inputs", "model"),)
    assert result.provenance[0].configuration == {"normalise": False}


def test_invalid_dependencies_fail_before_model_execution(default_test_model):
    called = False

    def transform(td):
        nonlocal called
        called = True
        return td

    pipeline = Pipeline([TransformStage("needs_input", transform, required_keys=["missing"])])
    with pytest.raises(ValueError, match="needs_input.*missing"):
        pipeline.run(default_test_model, TensorDict({}, batch_size=[]))
    assert not called


def test_duplicate_outputs_and_reserved_keys_are_rejected():
    first = TransformStage("first", lambda td: td, provided_keys=["result"])
    second = TransformStage("second", lambda td: td, provided_keys=["result"])
    with pytest.raises(ValueError, match="duplicates output key.*result.*first"):
        Pipeline([first, second])
    with pytest.raises(ValueError, match="reserved pipeline key"):
        Pipeline([TransformStage("bad", lambda td: td, provided_keys=["_pipeline"])])
    with pytest.raises(ValueError, match="duplicates output key.*report.*first"):
        Pipeline(
            [
                TransformStage("first", lambda td: td, provided_keys=["report"]),
                TransformStage("second", lambda td: td, provided_keys=[("report", "mean")]),
            ]
        )
    with pytest.raises(ValueError, match="reader.*writes_model.*writer"):
        Pipeline(
            [
                TransformStage("writer", lambda td: td, effects=["writes_model"]),
                TransformStage("reader", lambda td: td, incompatible_effects=["writes_model"]),
            ]
        )


def test_stage_contract_validation_errors():
    with pytest.raises(ValueError, match="non-empty"):
        TransformStage("", lambda td: td)
    with pytest.raises(TypeError, match="Pipeline keys"):
        TransformStage("bad-key", lambda td: td, required_keys=[1])
    with pytest.raises(ValueError, match="duplicate keys"):
        TransformStage("duplicate-key", lambda td: td, provided_keys=["output", "output"])
    with pytest.raises(ValueError, match="stage names"):
        Pipeline([TransformStage("same", lambda td: td), TransformStage("same", lambda td: td)])
    with pytest.raises(ValueError, match="writer.*reads_model.*reader"):
        Pipeline(
            [
                TransformStage("reader", lambda td: td, incompatible_effects=["reads_model"]),
                TransformStage("writer", lambda td: td, effects=["reads_model"]),
            ]
        )


def test_pipeline_reports_input_output_and_transform_contract_errors(default_test_model):
    pipeline = Pipeline([TransformStage("transform", lambda td: "not a tensordict")])
    with pytest.raises(RuntimeError, match="transform.*must return a TensorDict"):
        pipeline.run(default_test_model, TensorDict({}, batch_size=[]))

    with pytest.raises(TypeError, match="artifacts must be a TensorDict"):
        pipeline.validate({})

    collision = Pipeline([TransformStage("collision", lambda td: td, provided_keys=["input"])])
    with pytest.raises(ValueError, match="collision.*existing artifact keys.*input"):
        collision.run(default_test_model, TensorDict({"input": torch.ones(1)}, batch_size=[1]))

    nested_collision = Pipeline([TransformStage("collision", lambda td: td, provided_keys=["report"])])
    with pytest.raises(ValueError, match="collision.*existing artifact keys.*report"):
        nested_collision.run(default_test_model, TensorDict({"report": {"mean": torch.ones(1)}}, batch_size=[1]))

    missing_output = Pipeline([TransformStage("missing-output", lambda td: td, provided_keys=["output"])])
    with pytest.raises(ValueError, match="missing-output.*did not provide.*output"):
        missing_output.run(default_test_model, TensorDict({}, batch_size=[]))


class InvalidResultStage(Stage):
    def run(self, model, artifacts):
        return "not a tensordict"


def test_pipeline_rejects_a_stage_that_returns_non_tensordict(default_test_model):
    pipeline = Pipeline([InvalidResultStage("invalid")])
    with pytest.raises(TypeError, match="invalid.*not a TensorDict"):
        pipeline.run(default_test_model, TensorDict({}, batch_size=[]))


def test_pipeline_rechecks_requirements_after_a_transform(default_test_model):
    pipeline = Pipeline(
        [
            TransformStage("remove-input", lambda td: td.del_("input"), required_keys=["input"]),
            TransformStage("needs-input", lambda td: td, required_keys=["input"]),
        ]
    )

    with pytest.raises(ValueError, match="needs-input.*missing artifact keys.*input"):
        pipeline.run(default_test_model, TensorDict({"input": torch.ones(1)}, batch_size=[1]))


class FailingPrepare(HookingContextFactory):
    def _prepare_module(self, module, in_keys, out_keys, extra_relative_path):
        module.pipeline_setup_flag = True
        return module

    def _restore_module(self, module, in_keys, out_keys, extra_relative_path):
        del module.pipeline_setup_flag
        return module

    def _hook_module(self, module):
        raise RuntimeError("hook setup failed")


class InterruptingPrepare(FailingPrepare):
    def _hook_module(self, module):
        raise KeyboardInterrupt("interrupted")


class FailingCleanup(FailingPrepare):
    def _restore_module(self, module, in_keys, out_keys, extra_relative_path):
        raise RuntimeError("restore failed")


def test_context_cleanup_on_method_setup_and_execution_failure(default_test_model):
    setup = Pipeline([MethodStage("setup", FailingPrepare(), required_keys=["input"], provided_keys=["output"])])
    with pytest.raises(RuntimeError, match="setup.*hook setup failed"):
        setup.run(default_test_model, TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))
    assert not hasattr(default_test_model, "pipeline_setup_flag")

    execute = Pipeline([TransformStage("explode", lambda td: (_ for _ in ()).throw(RuntimeError("boom")))])
    with pytest.raises(RuntimeError, match="explode.*boom"):
        execute.run(default_test_model, TensorDict({}, batch_size=[]))


def test_context_cleanup_handles_interrupts_and_cleanup_errors(default_test_model):
    interrupted = Pipeline(
        [MethodStage("interrupt", InterruptingPrepare(), required_keys=["input"], provided_keys=["output"])]
    )
    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        interrupted.run(default_test_model, TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))
    assert not hasattr(default_test_model, "pipeline_setup_flag")

    class Handle:
        removed = False

        def remove(self):
            self.removed = True

    class Stack:
        exited = False

        def __exit__(self, *args):
            self.exited = True

    context = AddOne().prepare(default_test_model)
    handle, stack = Handle(), Stack()
    context._handle = handle
    context._stack = stack
    context._in_context = True
    context._abort_enter(prepared=False)
    assert not context._in_context
    assert context._handle is None
    assert handle.removed
    assert stack.exited

    context = FailingCleanup().prepare(default_test_model)
    with pytest.raises(RuntimeError, match="restore failed"):
        context.__enter__()
    assert not context._in_context
    assert context._handle is None
    assert context._hooked_module is None
