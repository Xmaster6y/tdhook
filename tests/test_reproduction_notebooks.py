"""Compile notebooks and execute their workflow statements on small inputs."""

import ast
import json
from pathlib import Path

import pytest
import torch
from tensordict import NonTensorData, TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from tdhook.attribution import AttentionContributor, CircuitLensArtifact, cluster_circuit_artifacts
from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator
from tdhook.weights import analyze_input_invariant_feature
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
    """Select the actual cell statements from a function through its workflow call."""
    notebook = json.loads((NOTEBOOK_DIR / filename).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(_python_source("".join(cell["source"])))
        for index, statement in enumerate(tree.body):
            if isinstance(statement, ast.FunctionDef) and statement.name == first_function:
                body = []
                for node in tree.body[index:]:
                    if isinstance(node, ast.With):  # Plotting is independent of the workflow.
                        break
                    body.append(node)
                return compile(ast.Module(body=body, type_ignores=[]), filename, "exec")
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
    )
    exec(_workflow_statements("weight-circuit-research-reproduction.ipynb", "publish_circuit_batch"), namespace)
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
    namespace.update(
        estimator=TwoNnDimensionEstimator(), backbone_acts={"layer": torch.randn(32, 2, 4, 4, generator=generator)}
    )
    exec(_workflow_statements("chess-dimension-estimation.ipynb", "channel_intrinsic_dimension"), namespace)
    dimensions = namespace["backbone_dims"]["layer"]
    assert dimensions.shape == (2,)
    assert torch.isfinite(dimensions).all()
    assert (dimensions > 0).all()


def test_concept_notebook_heatmap_preserves_tensordict_batch():
    tree = ast.parse(_compilable_code(NOTEBOOK_DIR / "concept-attribution.ipynb"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_lrp_on_image"
    )
    reduction = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "reduce_map" for t in node.targets)
    )
    namespace = _namespace()
    exec(compile(ast.Module(body=[reduction], type_ignores=[]), "concept-attribution.ipynb", "exec"), namespace)
    relevance = torch.randn(1, 3, 4, 5)
    result = namespace["reduce_map"](TensorDict({("attr", "input"): relevance}, [1]))
    torch.testing.assert_close(result[("metrics", "relevance_map")], relevance.sum(1).abs())
