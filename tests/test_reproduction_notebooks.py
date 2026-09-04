"""Compile notebooks and execute their workflow statements on small inputs."""

import ast
import contextlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import NonTensorData, TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from tdhook.attribution import (
    LRP,
    AttentionContributor,
    CircuitLensArtifact,
    attention_contributions,
    cluster_circuit_artifacts,
)
from tdhook.attribution.lrp_helpers.rules import EpsilonPlus
from tdhook.concepts import ChannelConditionedLRP, ConceptSelection
from tdhook.dimension import (
    ActivationSamples,
    DimensionSummary,
    channel_conditioned_samples,
    spatial_conditioned_samples,
)
from tdhook.latent import ActivationCaching
from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator
from tdhook.weights import analyze_input_invariant_feature, select_projection_outliers
from tdhook.workflow import Workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "docs/source/notebooks/tutorials"
MULTISTEP_NOTEBOOKS = (
    "chess-dimension-estimation.ipynb",
    "concept-attribution.ipynb",
    "othello-research-reproduction.ipynb",
    "rome-research-reproduction.ipynb",
    "weight-circuit-research-reproduction.ipynb",
)


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _compilable_code(path: Path) -> str:
    return _python_source(_code(path))


def _python_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        lines.append(f"{line[: len(line) - len(stripped)]}pass" if stripped.startswith(("%", "!")) else line)
    return "\n".join(lines)


@pytest.mark.parametrize("filename", MULTISTEP_NOTEBOOKS)
def test_multistep_notebook_code_compiles(filename: str):
    path = NOTEBOOK_DIR / filename
    compile(_compilable_code(path), str(path), "exec")


def _workflow_statements(filename, first_function):
    """Execute a notebook section across cell boundaries through its result read."""
    final_result = {
        "project_feature": "output_items",
        "cluster_circuits": "clusters",
        "estimate_dimensions": "head_dims",
    }[first_function]
    tree = ast.parse(_compilable_code(NOTEBOOK_DIR / filename))
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.FunctionDef) and statement.name == first_function:
            body = []
            for node in tree.body[index:]:
                body.append(node)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == final_result for target in node.targets
                ):
                    return compile(ast.Module(body=body, type_ignores=[]), filename, "exec")
            raise AssertionError(f"Missing workflow result: {final_result}")
    raise AssertionError(f"Missing notebook function: {first_function}")


def _namespace():
    return dict(
        NonTensorData=NonTensorData,
        TensorDict=TensorDict,
        TensorDictModule=TensorDictModule,
        Workflow=Workflow,
        torch=torch,
        model=nn.Identity(),
    )


def _execute_named(filename, names, namespace):
    """Execute actual definitions and workflow constructors in notebook order."""
    tree = ast.parse(_compilable_code(NOTEBOOK_DIR / filename))
    body = []
    found = set()
    for node in tree.body:
        node_names = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            node_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if node_names & set(names):
            found |= node_names & set(names)
            body.append(node)
    assert found == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), namespace)


def test_concept_selection_artifact_controls_the_actual_explanation_workflow():
    torch.manual_seed(7)
    model = nn.Sequential()
    model.add_module("features", nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU()))
    model.add_module("flatten", nn.Flatten())
    model.add_module("classifier", nn.Linear(8, 2))
    namespace = _namespace()
    namespace.update(
        model=model,
        device="cpu",
        layer_number=0,
        LRP=LRP,
        CustomEpsilonPlus=EpsilonPlus,
        ConceptSelection=ConceptSelection,
        ChannelConditionedLRP=ChannelConditionedLRP,
    )
    _execute_named(
        "concept-attribution.ipynb",
        [
            "init_max_logit_targets",
            "collect_layer_relevances",
            "concept_selection_workflow",
            "class_lrp",
            "relevance_maps",
            "explanation_workflow",
        ],
        namespace,
    )
    data = TensorDict({"input": torch.randn(4, 3, 2, 2), "concept_labels": torch.tensor([1, 1, 0, 0])}, [4])
    calibrated = namespace["concept_selection_workflow"](model, data)
    selection = calibrated["metrics", "concept_selection"][0:1].clone()
    query = TensorDict({"input": data["input"][:1], ("metrics", "concept_selection"): selection}, [1])
    workflow = namespace["explanation_workflow"](0)
    first = workflow(model, query.clone())
    changed = query.clone()
    changed["metrics", "concept_selection", "channel"] = 1 - selection["channel"]
    second = workflow(model, changed)
    torch.testing.assert_close(first["views", "lrp"], second["views", "lrp"])
    assert not torch.allclose(first["views", "concept"], second["views", "concept"])
    with pytest.raises(ValueError, match="missing TensorDict keys"):
        workflow(model, query.exclude(("metrics", "concept_selection")))
    assert all(not module._forward_hooks and not module._backward_hooks for module in model.modules())


def test_circuit_workflow_builds_artifacts_from_captured_tensors():
    class CircuitModel(nn.Module):
        def __init__(self):
            super().__init__()
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.hook_pattern = nn.Identity()
            block.attn.hook_v = nn.Identity()
            block.hook_resid_mid = nn.Identity()
            self.blocks = nn.ModuleList([block])
            self.cfg = SimpleNamespace(d_model=2)

        def forward(self, input, stop_at_layer):
            assert stop_at_layer == 1
            n = input.shape[1]
            self.blocks[0].attn.hook_pattern(torch.ones(1, 1, n, n) / n)
            self.blocks[0].attn.hook_v(torch.ones(1, n, 1, 2))
            self.blocks[0].hook_resid_mid(input[..., None].float().expand(-1, -1, 2))

    model = CircuitModel()
    namespace = _namespace()
    namespace.update(
        model=model,
        re=re,
        ActivationCaching=ActivationCaching,
        CircuitLensArtifact=CircuitLensArtifact,
        attention_contributions=attention_contributions,
        select_projection_outliers=select_projection_outliers,
        feature_layer=0,
        feature_index=0,
        encoder=torch.ones(2),
        output_weight=torch.eye(2).unsqueeze(0),
        selected_transcoder=SimpleNamespace(encode=lambda value: value),
        pattern_name="module.blocks.0.attn.hook_pattern",
        value_name="module.blocks.0.attn.hook_v",
        feature_input_name="module.blocks.0.hook_resid_mid",
        cache_key=("activations", "circuit"),
    )
    namespace["names"] = tuple(namespace[name] for name in ("pattern_name", "value_name", "feature_input_name"))
    _execute_named("weight-circuit-research-reproduction.ipynb", ["analyze_sample", "circuit_workflow"], namespace)
    wrapped = TensorDictModule(model, {"input": "input", "stop_at_layer": "stop_at_layer"}, [])

    def run(value):
        return namespace["circuit_workflow"](
            wrapped,
            TensorDict(
                {
                    "input": torch.full((1, 4), value),
                    "stop_at_layer": NonTensorData(1),
                    "sample": NonTensorData({"target_position": 2}),
                },
                [],
            ),
        )

    first, second = run(1), run(3)
    assert first["artifacts", "circuit"].target_activation == 1
    assert second["artifacts", "circuit"].target_activation == 3
    assert first["activations", "observed"].item() == 1
    assert all(not module._forward_hooks for module in model.modules())


def test_rome_edit_workflow_consumes_the_declared_delta_and_restores_weights(monkeypatch):
    helper_path = NOTEBOOK_DIR / "rome_reproduction.py"
    spec = importlib.util.spec_from_file_location("rome_reproduction", helper_path)
    helper = importlib.util.module_from_spec(spec)
    # Dataclass decoration resolves this module through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, helper)
    spec.loader.exec_module(helper)
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(1, 1, bias=False))
    model.linear.weight.data.fill_(1)
    namespace = _namespace()
    namespace.update(
        model=model,
        tokenizer=None,
        hparams=None,
        contextlib=contextlib,
        io=io,
        execute_rome=lambda model, tokenizer, request, hparams: {
            "linear.weight": (torch.tensor([request["delta"]]), torch.ones(1))
        },
        temporary_rank_one_edit=helper.temporary_rank_one_edit,
        compute_rewrite_quality_counterfact=lambda model, *args: {"score": float(model(torch.ones(1, 1)).detach())},
    )
    _execute_named(
        "rome-research-reproduction.ipynb",
        ["derive_rank_one_edit", "evaluate_rank_one_edit", "edit_workflow"],
        namespace,
    )
    record = {"case_id": 1, "requested_rewrite": {"delta": 2.0}}
    result = namespace["edit_workflow"](model, TensorDict({"record": NonTensorData(record)}, []))
    assert result["metrics", "edited_case"]["post"]["score"] == 3.0
    assert model.linear.weight.item() == 1.0
    consumer = Workflow(namespace["edit_workflow"].steps[1])
    with pytest.raises(ValueError, match="missing TensorDict keys"):
        consumer(model, TensorDict({"record": NonTensorData(record)}, []))
    result = consumer(
        model,
        TensorDict(
            {
                "record": NonTensorData(record),
                ("edit", "deltas"): NonTensorData({"linear.weight": (torch.tensor([4.0]), torch.ones(1))}),
            },
            [],
        ),
    )
    assert result["metrics", "edited_case"]["post"]["score"] == 5.0
    assert model.linear.weight.item() == 1.0


def test_weight_notebook_transports_and_reads_non_tensor_artifacts():
    namespace = _namespace()
    model = namespace["model"]
    model.W_E = torch.zeros(128, 4)
    model.W_E[0, 0], model.W_E[1, 0] = 100, -100
    model.W_U = model.W_E.T.clone()
    namespace.update(
        analyze_input_invariant_feature=analyze_input_invariant_feature,
        feature_layer=0,
        feature_index=0,
        feature_encoders=[torch.eye(4)],
        feature_decoders=[torch.eye(4)],
        token_labels=[str(i) for i in range(128)],
    )
    exec(_workflow_statements("weight-circuit-research-reproduction.ipynb", "project_feature"), namespace)
    artifact = namespace["weight_artifact"]
    assert namespace["input_items"] == [*artifact.embedding_negative, *artifact.embedding_positive]
    assert namespace["output_items"] == list(artifact.output_positive)
    assert {item.index for item in namespace["input_items"]} == {0, 1}
    assert namespace["output_items"][0].index == 0


def test_othello_optimization_workflow_decodes_its_result_artifact():
    def optimize(model, inputs, target, objective):
        return SimpleNamespace(interventions=[SimpleNamespace(value=inputs)])

    namespace = _namespace()
    namespace.update(
        optimize_board_intervention=optimize,
        position_target=None,
        probe_board_objective=None,
        probe=torch.eye(3).reshape(1, 3, 1, 1, 3),
        probe_mode=0,
    )
    _execute_named(
        "othello-research-reproduction.ipynb",
        ["optimize_board", "decode_optimized_board", "optimization_workflow"],
        namespace,
    )
    workflow = namespace["optimization_workflow"]
    result = workflow(namespace["model"], TensorDict({"input": torch.tensor([[[0.0, 0.0, 1.0]]])}, []))
    assert result["views", "optimized_board"].item() == 2
    consumer = Workflow(workflow.steps[1])
    changed = SimpleNamespace(interventions=[SimpleNamespace(value=torch.tensor([[[1.0, 0.0, 0.0]]]))])
    result = consumer(namespace["model"], TensorDict({("interventions", "optimized"): NonTensorData(changed)}, []))
    assert result["views", "optimized_board"].item() == 0
    with pytest.raises(ValueError, match="missing TensorDict keys"):
        consumer(namespace["model"], TensorDict({}, []))


def test_circuit_notebook_transports_artifacts_and_computes_errors():
    namespace = _namespace()
    artifacts = [
        CircuitLensArtifact(0, 0, (1,), 1.0, (), (AttentionContributor(0, head, 0, 1, 1.0),), ()) for head in (0, 0, 1)
    ]
    namespace.update(
        cluster_circuit_artifacts=cluster_circuit_artifacts,
        circuit_artifacts=artifacts,
        observed_activations=[1.0, 2.0, 3.0],
        published_activations=[1.0, 1.0, 1.0],
        circuit_results=[
            TensorDict(
                {
                    ("artifacts", "circuit"): NonTensorData(artifact),
                    ("activations", "observed"): torch.tensor(float(i + 1)),
                },
                [],
            )
            for i, artifact in enumerate(artifacts)
        ],
        reference={"samples": [{"target_activation": 1.0}] * 3},
    )
    exec(_workflow_statements("weight-circuit-research-reproduction.ipynb", "cluster_circuits"), namespace)
    result = namespace["circuit_result"]
    assert result[("artifacts", "circuits")] == tuple(artifacts)
    torch.testing.assert_close(result[("metrics", "activation_error")], torch.tensor([0.0, 1.0, 2.0]))
    assert namespace["clusters"] == cluster_circuit_artifacts(
        artifacts,
        min_frequency=0.05,
        min_abs_score=0.0,
        eps=0.8,
        min_samples=2,
    )
    assert namespace["clusters"].labels == (0, 0, -1)


def test_chess_notebook_executes_dimension_workflow():
    namespace = _namespace()
    generator = torch.Generator().manual_seed(7)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, board):
            return self.layer(board)

    inputs = torch.randn(32, 2, 4, 4, generator=generator)
    namespace.update(
        model=TensorDictModule(Model(), ["board"], ["output"]),
        TwoNnDimensionEstimator=TwoNnDimensionEstimator,
        re=re,
        BACKBONE_PATTERN=r"module.layer",
        HEAD_PATTERN=r"no_match",
        ActivationCaching=ActivationCaching,
        ActivationSamples=ActivationSamples,
        DimensionSummary=DimensionSummary,
        channel_conditioned_samples=channel_conditioned_samples,
        spatial_conditioned_samples=spatial_conditioned_samples,
        td=TensorDict({"board": inputs}, [32]),
    )
    exec(_workflow_statements("chess-dimension-estimation.ipynb", "estimate_dimensions"), namespace)
    dimensions = namespace["backbone_dims"]["module.layer"]
    assert dimensions.shape == (2,)
    assert torch.isfinite(dimensions).all()
    assert (dimensions > 0).all()
    assert len(namespace["dimension_workflow"].steps) == 7
    result = namespace["dimension_result"]
    assert result["dimensions", "squares", "module.layer"].shape == (16,)
    torch.testing.assert_close(result["summaries", "channels", "module.layer"]["mean"], dimensions.mean())
    assert not namespace["model"].module.layer._forward_hooks


def test_concept_notebook_heatmap_preserves_tensordict_batch():
    tree = ast.parse(_compilable_code(NOTEBOOK_DIR / "concept-attribution.ipynb"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "relevance_maps")
    namespace = _namespace()
    exec(compile(ast.Module(body=[function], type_ignores=[]), "concept-attribution.ipynb", "exec"), namespace)
    relevance = torch.randn(1, 3, 4, 5)
    ordinary, conditioned = namespace["relevance_maps"](relevance, 2 * relevance)
    torch.testing.assert_close(ordinary, relevance.sum(1).abs())
    torch.testing.assert_close(conditioned, 2 * ordinary)
