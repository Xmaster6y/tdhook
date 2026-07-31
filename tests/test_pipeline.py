import pytest
import torch
from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookHandle
from tdhook.modules import HookedModule
from tdhook.pipeline import MethodStage, Pipeline, TransformStage


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
    with pytest.raises(ValueError, match="reader.*writes_model.*writer"):
        Pipeline(
            [
                TransformStage("writer", lambda td: td, effects=["writes_model"]),
                TransformStage("reader", lambda td: td, incompatible_effects=["writes_model"]),
            ]
        )


class FailingPrepare(HookingContextFactory):
    def _prepare_module(self, module, in_keys, out_keys, extra_relative_path):
        module.pipeline_setup_flag = True
        return module

    def _restore_module(self, module, in_keys, out_keys, extra_relative_path):
        del module.pipeline_setup_flag
        return module

    def _hook_module(self, module):
        raise RuntimeError("hook setup failed")


def test_context_cleanup_on_method_setup_and_execution_failure(default_test_model):
    setup = Pipeline([MethodStage("setup", FailingPrepare(), required_keys=["input"], provided_keys=["output"])])
    with pytest.raises(RuntimeError, match="setup.*hook setup failed"):
        setup.run(default_test_model, TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))
    assert not hasattr(default_test_model, "pipeline_setup_flag")

    execute = Pipeline([TransformStage("explode", lambda td: (_ for _ in ()).throw(RuntimeError("boom")))])
    with pytest.raises(RuntimeError, match="explode.*boom"):
        execute.run(default_test_model, TensorDict({}, batch_size=[]))
