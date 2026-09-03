import json

from benchmarks.run import CONFIGS, SCHEMA_VERSION, main, run_suite


def test_smoke_benchmark_checks_agreement_and_emits_stable_schema():
    result = run_suite("smoke", "cpu")

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["suite"] == "tdhook-maintained"
    assert result["mode"] == "smoke"
    assert result["configuration"] == {
        "batch_size": 4,
        "width": 32,
        "depth": 3,
        "selected_features": 8,
        "warmup": 1,
        "repeats": 3,
    }
    assert result["environment"]["hardware"]["device"] == "cpu"
    assert isinstance(result["environment"]["dirty"], bool)
    assert {item["name"] for item in result["benchmarks"]} == {
        "activation_capture",
        "activation_intervention",
        "input_saliency",
    }
    for benchmark in result["benchmarks"]:
        assert benchmark["correctness"]["passed"] is True
        assert set(benchmark["implementations"]) == {"tdhook", "reference"}
        for implementation in benchmark["implementations"].values():
            assert len(implementation["timing"]["samples"]) == CONFIGS["smoke"].repeats
            assert implementation["timing"]["unit"] == "ns"
            assert implementation["memory"]["unit"] == "byte"

    json.dumps(result, sort_keys=True)


def test_cli_writes_machine_readable_report(tmp_path):
    output = tmp_path / "benchmark.json"

    assert main(["--mode", "smoke", "--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["schema_version"] == SCHEMA_VERSION
    assert all(item["correctness"]["passed"] for item in report["benchmarks"])
