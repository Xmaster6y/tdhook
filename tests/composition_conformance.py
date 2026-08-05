import csv
from pathlib import Path

from tdhook.workflow import WorkflowPlan


CONFORMANCE_MATRIX = Path(__file__).parents[1] / "docs" / "source" / "_static" / "composition-conformance.csv"


def conformance_rows():
    with CONFORMANCE_MATRIX.open(newline="") as matrix_file:
        return list(csv.DictReader(matrix_file))


def serialize_plan(plan: WorkflowPlan) -> str:
    return "; ".join(
        f"{'+'.join(execution.steps)}:{execution.kind}:{execution.model_passes}:"
        f"{'coalesced' if execution.coexecuted else 'separate'}"
        for execution in plan.executions
    )


def assert_conformance(test_id: str, plan: WorkflowPlan, *, status: str) -> None:
    matches = [row for row in conformance_rows() if row["test_id"] == test_id]
    assert len(matches) == 1, f"Expected one conformance row for {test_id!r}"
    row = matches[0]
    assert row["status"] == status
    assert int(row["model_passes"]) == plan.model_passes
    assert row["expected_plan"] == serialize_plan(plan)
