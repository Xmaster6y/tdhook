import pytest

from tdhook.artifacts import (
    ArtifactAdapter,
    ArtifactContract,
    ArtifactRegistry,
    activation_caching_adapter,
    attribution_adapter,
    probing_adapter,
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


def test_contracts_and_existing_method_adapters_are_storage_independent():
    contract = ArtifactContract(requires={"source": ("inputs", "image")}, provides={"score": ("metrics", "score")})
    adapter = ArtifactAdapter("legacy-score", contract, {"source": "image", "score": "score"})
    assert adapter.contract.required_keys == (("inputs", "image"),)
    assert adapter.storage["score"] == "score"
    assert activation_caching_adapter().contract.provided_keys == (("activations", "cache"),)
    assert probing_adapter().contract.provided_keys == (("probes", "results"),)
    assert attribution_adapter().contract.provided_keys == (("attributions", "values"),)
    assert weight_adapter().contract.provided_keys == (("interventions", "weights"),)


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
