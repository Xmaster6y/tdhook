# Maintained benchmark suite

This suite measures representative attribution, activation-capture, and
activation-intervention tasks against Captum or native PyTorch hooks. It covers
the current TDHook API; it does not reproduce or extend the historical v0.1
paper benchmark.

Install the project and benchmark reference dependency, then run the cheap
local profile:

```bash
uv sync --extra benchmark
uv run python -m benchmarks.run --mode smoke --device cpu --output benchmark-smoke.json
```

For a less noisy measurement, run the fixed full profile on the device being
reported:

```bash
uv run python -m benchmarks.run --mode full --device cuda --output benchmark-full.json
```

Use `--device cpu` on systems without CUDA. `--device auto` selects CUDA when
available and otherwise selects CPU.

## Result contract

The output is JSON with a versioned schema. It records the TDHook commit and an
explicit dirty-checkout marker, package versions, Python/platform details,
hardware device, fixed benchmark configuration, correctness tolerances, raw
nanosecond samples, summary timing, and peak memory. Timings are collected only
after TDHook and reference outputs agree. A mismatch aborts the run without
publishing a result file. Treat a report with `environment.dirty` set to `true`
as a measurement of local source changes rather than of the recorded commit
alone.

CUDA memory is the change in PyTorch's peak allocated tensor memory. CPU memory
uses `tracemalloc` and therefore reports peak Python-tracked allocations, not
native tensor storage. Compare results only when the mode, configuration,
device, and memory method match.
