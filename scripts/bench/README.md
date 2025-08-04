# TDHook Runtime Benchmark

Comprehensive benchmark measuring performance impact of batch size, model width, and model height on TDHook vs NNsight.

## Usage

```bash
uv run -m scripts.bench.runtime
```

## Output

- Console: Real-time progress and summary
- `./results/bench/runtime/benchmark_results.json`: Detailed timing data
- `./results/bench/runtime/benchmark_results.png`: Performance curves visualization

## Parameters Tested

- **Batch sizes**: [1, 8, 64, 256, 1024, 4096, 10_000, 20_000] (fixed: width=10, height=12)
- **Model widths**: [8, 16, 32, 64, 128, 256, 512] (fixed: height=12, batch_size=1)
- **Model heights**: [4, 8, 12, 16, 20, 24, 32] (fixed: width=10, batch_size=1)

Each configuration tested with 3 fixed seeds (42, 123, 456) and averaged.
