import csv
from pathlib import Path

from torch import nn

from tdhook import attribution, latent, weights
from tdhook.attribution import IntegratedGradients, LRP
from tdhook.latent import dimension_estimation
from tdhook.latent import ActivationCaching, Probing
from tdhook.stages import (
    ActivationCachingStage,
    AttributionStage,
    DOCUMENTED_STAGE_CAPABILITIES,
    ProbingStage,
    WeightInterventionStage,
)
from tdhook.weights import Adapters


CAPABILITY_MATRIX = Path(__file__).parents[1] / "docs" / "source" / "_static" / "composition-capabilities.csv"
VALID_STATES = {"supported", "unsupported", "untested", "not-applicable"}


def _capability_rows():
    with CAPABILITY_MATRIX.open(newline="") as matrix_file:
        return list(csv.DictReader(matrix_file))


def test_capability_matrix_covers_every_public_method_and_operator():
    public_symbols = (
        set(attribution.__all__) | set(latent.__all__) | set(dimension_estimation.__all__) | set(weights.__all__)
    )

    rows = _capability_rows()

    matrix_symbols = {row["symbol"] for row in rows}
    assert len(matrix_symbols) == len(rows), "capability matrix contains duplicate symbols"
    assert matrix_symbols == public_symbols


def test_capability_matrix_has_preflight_fields_and_explicit_states():
    required_fields = {
        "execution",
        "hooks",
        "effects_and_ordering",
        "model_passes",
        "required_keys",
        "produced_keys",
        "model_mutation",
        "specialisation",
        "device_batch_gradient",
    }

    rows = _capability_rows()

    for row in rows:
        assert all(row[field].strip() for field in required_fields), row["symbol"]
        assert row["composed_model"] in VALID_STATES, row["symbol"]
        assert row["same_run"] in VALID_STATES, row["symbol"]
        assert row["multi_stage"] in VALID_STATES, row["symbol"]


def test_callback_and_module_key_capabilities_are_not_undercounted():
    rows = {row["symbol"]: row for row in _capability_rows()}

    assert rows["GradCAM"]["produced_keys"] == "attribution_key/modules_to_attribute"
    assert "potentially unbounded" in rows["TaskVectors"]["model_passes"]
    assert "both evaluation callbacks" in rows["TaskVectors"]["device_batch_gradient"]


def test_documented_built_in_stage_rows_have_executable_capabilities():
    rows = {row["symbol"]: row for row in _capability_rows()}
    stages = {
        "ActivationCaching": ActivationCachingStage("cache", ActivationCaching("0")),
        "IntegratedGradients": AttributionStage("ig", IntegratedGradients(n_steps=2)),
        "LRP": AttributionStage("lrp", LRP(warn_on_missing_rule=False)),
        "Probing": ProbingStage("probe", Probing("0", lambda *_: object()), object()),
        "Adapters": WeightInterventionStage("adapter", Adapters({"identity": (nn.Identity(), "0", "0")})),
    }
    supported_hook_stages = {
        row["symbol"]
        for row in rows.values()
        if row["kind"] in {"hook method", "weight method"} and row["multi_stage"] == "supported"
    }

    assert set(DOCUMENTED_STAGE_CAPABILITIES) == set(stages) == supported_hook_stages
    for symbol, stage in stages.items():
        row = rows[symbol]
        assert row["multi_stage"] == "supported"
        assert (row["same_run"] == "supported") is bool(stage.coexecution_key)
        for key in stage.required_keys:
            path = key if isinstance(key, str) else "/".join(key)
            assert path in row["required_keys"]
        for key in stage.provided_keys:
            path = key if isinstance(key, str) else "/".join(key)
            assert path in row["produced_keys"]
