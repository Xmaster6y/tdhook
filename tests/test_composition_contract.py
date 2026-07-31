import csv
from pathlib import Path

from tdhook import attribution, latent, weights
from tdhook.latent import dimension_estimation
from tdhook.stages import DOCUMENTED_STAGE_CAPABILITIES, capability_for_stage


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
    rows = {row["symbol"] for row in _capability_rows()}
    assert set(DOCUMENTED_STAGE_CAPABILITIES).issubset(rows)
    for method in DOCUMENTED_STAGE_CAPABILITIES.values():
        capability = capability_for_stage(method)
        assert capability.required_keys
        assert capability.provided_keys
