import csv
from pathlib import Path

from tdhook.pipeline import ExecutionPlan


CONFORMANCE_MATRIX = Path(__file__).parents[1] / "docs" / "source" / "_static" / "composition-conformance.csv"


def conformance_rows():
    with CONFORMANCE_MATRIX.open(newline="") as matrix_file:
        return list(csv.DictReader(matrix_file))


def serialize_plan(plan: ExecutionPlan) -> str:
    return "; ".join(
        f"{'+'.join(run.stages)}:{run.kind}:{run.model_passes}:{'coalesced' if run.coalesced else 'separate'}"
        for run in plan.runs
    )


def assert_conformance(test_id: str, plan: ExecutionPlan, *, status: str) -> None:
    matches = [row for row in conformance_rows() if row["test_id"] == test_id]
    assert len(matches) == 1, f"Expected one conformance row for {test_id!r}"
    row = matches[0]
    assert row["status"] == status
    assert int(row["model_passes"]) == plan.model_passes
    assert row["expected_plan"] == serialize_plan(plan)
