import pytest
import torch
from torch import nn
from tensordict import TensorDict

import tdhook.artifacts as artifacts
from tdhook.artifacts import (
    ArtifactAdapter,
    ArtifactContract,
    ArtifactRegistry,
    activation_caching_adapter,
    attribution_adapter,
    probing_adapter,
    is_private_key,
    make_provenance,
    validate_artifact_key,
    weight_adapter,
)


def test_public_and_private_namespace_validation():
    assert validate_artifact_key(("inputs", "image")) == ("inputs", "image")
    assert validate_artifact_key(("_private", "saliency", "grads"), public=False) == (
        "_private",
        "saliency",
        "grads",
    )
    with pytest.raises(ValueError, match="Public artifact key"):
        validate_artifact_key("image")
    with pytest.raises(ValueError, match="Private artifact keys"):
        validate_artifact_key("scratch", public=False)
    with pytest.raises(TypeError, match="Artifact keys"):
        validate_artifact_key(("inputs", 1))
    assert is_private_key(("_private", "saliency"))
    assert not is_private_key(("inputs", "image"))


def test_contracts_and_existing_method_adapters_are_storage_independent():
    contract = ArtifactContract(requires={"source": ("inputs", "image")}, provides={"score": ("metrics", "score")})
    adapter = ArtifactAdapter("legacy-score", contract, {"source": "image", "score": "score"})
    assert adapter.contract.required_keys == (("inputs", "image"),)
    assert adapter.storage["score"] == "score"
    assert activation_caching_adapter().contract.provided_keys == (("activations", "cache"),)
    assert probing_adapter().contract.provided_keys == (("probes", "results"),)
    assert attribution_adapter().contract.provided_keys == (("attributions", "values"),)
    assert weight_adapter().contract.provided_keys == (("outputs", "model"),)

    with pytest.raises(ValueError, match="names must be non-empty"):
        ArtifactContract(requires={"": ("inputs", "image")})
    with pytest.raises(ValueError, match="keys must be unique"):
        ArtifactContract(provides={"one": ("metrics", "score"), "two": ("metrics", "score")})
    with pytest.raises(ValueError, match="method identifier"):
        ArtifactAdapter("", contract, {"source": "image", "score": "score"})
    with pytest.raises(ValueError, match="exactly match"):
        ArtifactAdapter("legacy-score", contract, {"source": "image"})


def test_adapter_copies_public_requirements_and_products_through_legacy_storage():
    adapter = ArtifactAdapter(
        "legacy-score",
        ArtifactContract(requires={"source": ("inputs", "image")}, provides={"score": ("metrics", "score")}),
        {"source": "image", "score": "score"},
    )
    artifacts_td = TensorDict({"inputs": {"image": torch.ones(2)}}, batch_size=[2])
    storage = adapter.prepare(artifacts_td)
    assert torch.equal(storage["image"], artifacts_td[("inputs", "image")])

    storage.set("score", torch.ones(2))
    result = adapter.finalize(artifacts_td, storage)
    assert torch.equal(result[("metrics", "score")], torch.ones(2))

    with pytest.raises(ValueError, match="did not provide"):
        adapter.finalize(artifacts_td, TensorDict())


def test_provenance_handles_missing_package_metadata_and_buffer_only_models(monkeypatch):
    class BufferOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1))

    def missing_package(_):
        raise artifacts.PackageNotFoundError

    monkeypatch.setattr(artifacts, "version", missing_package)
    provenance = make_provenance(stage="cache", method="ActivationCaching", model=BufferOnlyModel())

    assert provenance.package_version == "unknown"
    assert provenance.device == "cpu"
    assert provenance.dtype == "torch.float32"


def test_registry_rejects_conflicting_owners_and_stale_artifacts():
    registry = ArtifactRegistry()
    first = registry.begin_generation()
    registry.claim(("activations", "layer"), "cache", generation=first)
    registry.require_fresh(("activations", "layer"), generation=first)
    second = registry.begin_generation()
    with pytest.raises(ValueError, match="not been registered"):
        registry.require_fresh(("activations", "missing"), generation=second)
    with pytest.raises(ValueError, match="stale"):
        registry.require_fresh(("activations", "layer"), generation=second)
    with pytest.raises(ValueError, match="already owned"):
        registry.claim(("activations", "layer"), "other", generation=second)
