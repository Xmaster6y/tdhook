import csv
from pathlib import Path

from tdhook import attribution, latent, weights
from tdhook.latent import dimension_estimation


CAPABILITY_MATRIX = Path(__file__).parents[1] / "docs" / "source" / "_static" / "composition-capabilities.csv"
VALID_STATES = {"supported", "unsupported", "untested", "not-applicable"}


def test_capability_matrix_covers_every_public_method_and_operator():
    public_symbols = (
        set(attribution.__all__) | set(latent.__all__) | set(dimension_estimation.__all__) | set(weights.__all__)
    )

    with CAPABILITY_MATRIX.open(newline="") as matrix_file:
        rows = list(csv.DictReader(matrix_file))

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

    with CAPABILITY_MATRIX.open(newline="") as matrix_file:
        rows = list(csv.DictReader(matrix_file))

    for row in rows:
        assert all(row[field].strip() for field in required_fields), row["symbol"]
        assert row["composed_model"] in VALID_STATES, row["symbol"]
        assert row["same_run"] in VALID_STATES, row["symbol"]
        assert row["multi_stage"] in VALID_STATES, row["symbol"]
