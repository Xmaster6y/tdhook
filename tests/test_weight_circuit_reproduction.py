"""Static contracts for the resource-intensive WeightLens/CircuitLens reproduction."""

import hashlib
import json
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "docs/source/notebooks/tutorials/weight-circuit-research-reproduction.ipynb"
REFERENCE = REPO_ROOT / "docs/source/notebooks/assets/gemma-2-2b-feature-24-reference.json"
EVIDENCE_SLICE_SHA256 = "b8bd6fda45dda1f6e18c8a283be1883af32dc5016ee9b2f859dc2b6ae01c62a4"


def test_reproduction_is_linked_and_declares_its_resource_boundary():
    tutorials = (REPO_ROOT / "docs/source/tutorials.rst").read_text()
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    assert ":link: notebooks/tutorials/weight-circuit-research-reproduction" in tutorials
    assert "notebooks/tutorials/weight-circuit-research-reproduction" in tutorials
    assert notebook.metadata["tdhook"] == {
        "ci": False,
        "estimated_download_gb": 20,
        "estimated_vram_gb": 16,
        "network": True,
        "runtime": "cuda",
    }


def test_reproduction_uses_public_tdhook_apis_and_all_code_cells_parse():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    for public_api in (
        "analyze_input_invariant_feature",
        "attention_contributions",
        "select_projection_outliers",
        "CircuitLensArtifact",
        "cluster_circuit_artifacts",
    ):
        assert public_api in code
    assert "--with jupyterlab" in "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert str(REFERENCE.relative_to(REPO_ROOT)) in code
    assert "AutoTokenizer.from_pretrained(model_snapshot)" in code
    assert "tokenizer=tokenizer" in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")


def test_reference_slice_records_exact_public_provenance_and_bounded_examples():
    reference = json.loads(REFERENCE.read_text())

    assert reference["feature"] == {"layer": 0, "index": 24}
    assert len(reference["samples"]) == 12
    evidence_bytes = json.dumps(
        {"weightlens": reference["weightlens"], "samples": reference["samples"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert reference["evidence_slice_sha256"] == EVIDENCE_SLICE_SHA256
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SLICE_SHA256
    assert reference["provenance"] == {
        "model": "google/gemma-2-2b",
        "model_revision": "c5ebcd40d208330abc697524c919956e692655cf",
        "transcoder_set": "mwhanna/gemma-scope-transcoders",
        "transcoder_revision": "bd5773156dea09893636c801df1237d0410307d2",
        "circuit_dataset": "egolimblevskaia/circuitlens-gemma-2-2b-transcoder-circuit-analysis",
        "circuit_dataset_revision": "fcd78fd98c3de6cc869d7df9e7d0f4a864ffea50",
        "circuit_file": "analysis_layer_0.jsonl",
        "circuit_blob_sha256": "76f8ba7dba35222d366835f82f5557480bace71f89dc60ec16a8f946a7e20faf",
        "weight_dataset": "egolimblevskaia/weightlens-gemma-2-2b-transcoder-descriptions",
        "weight_dataset_revision": "aa51f4ee40784ac7af168932e2505fe3e4bab833",
        "weight_file": "feature_analysis_layer_0.json",
        "weight_blob_sha256": "9b3c26c6fbf969bd82a996d561b52df6bbbe63f96d615684f332e23fefc5c935",
        "weightlens_revision": "93f820024034bb9b1829f7f09c1483ec3bc71f49",
        "circuitlens_revision": "d24b7e3a71ea1fbce0800056eb7333d1a282303d",
    }
    assert all(len(value) == 64 for key, value in reference["provenance"].items() if key.endswith("_sha256"))
    assert [item["token_id"] for item in reference["weightlens"]["embedding_positive"]] == [19538, 4818]
