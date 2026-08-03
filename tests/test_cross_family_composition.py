import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from tdhook.contexts import HookingContextFactory
from tdhook.latent import ActivationCaching, Probing
from tdhook.pipeline import MethodStage, Pipeline
from tdhook.stages import ActivationCachingStage, ProbingStage, WeightInterventionStage
from tdhook.weights import Adapters


CONFORMANCE_MATRIX = Path(__file__).parents[1] / "docs" / "source" / "_static" / "composition-conformance.csv"


def _hook_count(model):
    hook_attributes = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    return sum(len(getattr(module, attribute)) for module in model.modules() for attribute in hook_attributes)


def test_activation_capture_and_probing_follow_declared_conservative_split(default_test_model):
    class RecordingProbe:
        def __init__(self):
            self.values = []

        def step(self, data, **kwargs):
            self.values.append(data.detach().clone())

    probe = RecordingProbe()
    probe_results = object()
    cache_factory = ActivationCaching("linear1")
    pipeline = Pipeline(
        [
            ActivationCachingStage("capture", cache_factory),
            ProbingStage("probe", Probing("linear2", lambda *_: probe), probe_results),
        ]
    )
    inputs = torch.ones(2, 10)
    artifacts = TensorDict({"inputs": {"input": inputs}}, batch_size=[2])
    hooks_before = _hook_count(default_test_model)

    plan = pipeline.plan(artifacts)
    result = pipeline.run(default_test_model, artifacts, model_id="tiny-linear")

    assert [(run.stages, run.model_passes, run.coalesced) for run in plan.runs] == [
        (("capture",), 1, False),
        (("probe",), 1, False),
    ]
    assert plan.model_passes == 2
    assert result.artifacts[("activations", "cache")]["linear1"].shape == (2, 20)
    assert result.artifacts[("activations", "cache")]["linear1"].device == inputs.device
    assert result.artifacts[("probes", "results")] is probe_results
    assert len(probe.values) == 1 and probe.values[0].shape == (2, 20)
    assert [record.parents for record in result.provenance] == [
        (("inputs", "input"),),
        (("inputs", "input"),),
    ]
    assert cache_factory._hooking_context_kwargs["cache"] is None
    assert _hook_count(default_test_model) == hooks_before


def test_real_intervention_and_activation_read_split_without_state_leaks(default_test_model):
    inputs = torch.ones(2, 10)
    expected = default_test_model(inputs)
    state_before = {key: value.detach().clone() for key, value in default_test_model.state_dict().items()}
    hooks_before = _hook_count(default_test_model)
    pipeline = Pipeline(
        [
            WeightInterventionStage("intervene", Adapters({"identity": (nn.Identity(), "linear1", "linear1")})),
            ActivationCachingStage("read", ActivationCaching("linear2")),
        ]
    )
    artifacts = TensorDict({"inputs": {"input": inputs}}, batch_size=[2])

    plan = pipeline.plan(artifacts)
    result = pipeline.run(default_test_model, artifacts)

    assert [(run.stages, run.model_passes, run.coalesced) for run in plan.runs] == [
        (("intervene",), 1, False),
        (("read",), 1, False),
    ]
    assert plan.model_passes == 2
    torch.testing.assert_close(result.artifacts[("outputs", "model")], expected)
    assert result.artifacts[("activations", "cache")]["linear2"].shape == (2, 20)
    assert set(default_test_model.state_dict()) == set(state_before)
    for key, value in default_test_model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])
    assert _hook_count(default_test_model) == hooks_before
    assert not any("adapter" in name for name, _ in default_test_model.named_modules())


@pytest.mark.parametrize("wrap_tensordict", [False, True], ids=["pytorch", "tensordict-module"])
def test_composed_models_support_nested_multiple_input_output_keys(wrap_tensordict):
    class Pair(nn.Module):
        def forward(self, left, right):
            return left + right, left - right

    in_keys = [("inputs", "left"), ("inputs", "right")]
    out_keys = [("outputs", "sum"), ("outputs", "difference")]
    model = Pair()
    if wrap_tensordict:
        model = TensorDictModule(model, in_keys=in_keys, out_keys=out_keys)
    pipeline = Pipeline(
        [
            MethodStage(
                "pair",
                HookingContextFactory(),
                required_keys=in_keys,
                provided_keys=out_keys,
                model_in_keys=in_keys,
                model_out_keys=out_keys,
            )
        ]
    )
    left = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    right = torch.full((2, 3), 2.0)

    result = pipeline.run(
        model,
        TensorDict({"inputs": {"left": left, "right": right}}, batch_size=[2]),
    )

    assert result.plan.model_passes == 1
    torch.testing.assert_close(result.artifacts[("outputs", "sum")], left + right)
    torch.testing.assert_close(result.artifacts[("outputs", "difference")], left - right)
    assert _hook_count(model) == 0


def test_cross_family_failure_removes_hooks_and_restores_model(default_test_model):
    class ExplodingProbe:
        def step(self, data, **kwargs):
            raise ValueError("probe failed")

    state_before = {key: value.detach().clone() for key, value in default_test_model.state_dict().items()}
    hooks_before = _hook_count(default_test_model)
    pipeline = Pipeline(
        [
            WeightInterventionStage("intervene", Adapters({"identity": (nn.Identity(), "linear1", "linear1")})),
            ProbingStage("probe", Probing("linear2", lambda *_: ExplodingProbe()), object()),
        ]
    )
    artifacts = TensorDict({"inputs": {"input": torch.ones(2, 10)}}, batch_size=[2])

    plan = pipeline.plan(artifacts)
    assert [(run.stages, run.model_passes, run.coalesced) for run in plan.runs] == [
        (("intervene",), 1, False),
        (("probe",), 1, False),
    ]

    with pytest.raises(RuntimeError, match="probe.*probe failed"):
        pipeline.run(default_test_model, artifacts)

    for key, value in default_test_model.state_dict().items():
        torch.testing.assert_close(value, state_before[key])
    assert _hook_count(default_test_model) == hooks_before
    assert not any("adapter" in name for name, _ in default_test_model.named_modules())


def test_supported_conformance_rows_name_a_test_and_expected_plan():
    with CONFORMANCE_MATRIX.open(newline="") as matrix_file:
        rows = list(csv.DictReader(matrix_file))

    assert {row["status"] for row in rows} <= {"supported", "unsupported"}
    supported = [row for row in rows if row["status"] == "supported"]
    unsupported = [row for row in rows if row["status"] == "unsupported"]
    assert supported
    assert unsupported
    assert all(row["test_id"].startswith("test_") for row in supported)
    assert all(row["expected_plan"] and row["model_passes"].isdigit() for row in supported)
    source = Path(__file__).read_text()
    source += (Path(__file__).with_name("test_concepts.py")).read_text()
    source += (Path(__file__).with_name("test_dimension_pipeline.py")).read_text()
    source += (Path(__file__).with_name("test_pipeline.py")).read_text()
    source += (Path(__file__).with_name("test_stages.py")).read_text()
    for row in rows:
        assert f"def {row['test_id']}" in source, row["combination"]
    assert all("split" in row["expected_plan"] for row in unsupported)
