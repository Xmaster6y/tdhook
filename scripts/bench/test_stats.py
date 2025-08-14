"""
Test stats.

Run with:

```
uv run --group scripts -m scripts.bench.test_stats
```
"""

from typing import Dict
import json
from pathlib import Path
from loguru import logger
from importlib import import_module

from .utils import Measurer

SEEDS = [42, 123, 456]
TASKS = {
    "gpt2_cache": ["tdhook", "nnsight", "transformer_lens"],
    "integrated_gradients": ["tdhook", "captum_add", "captum"],
    "lrp": ["tdhook", "zennit"],
    "mlp_intervene": ["tdhook", "nnsight"],
}


def run_default_task(task, script_name: str, measurer: Measurer):
    """Run a task."""
    default_parameters = task.default_parameters
    measurer.measure_script(script_name, default_parameters)


def save_results(results: Dict):
    """Save results to JSON file."""
    results_dir = Path("./results/bench")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "test-results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    """Run all benchmarks."""
    logger.info("Starting test stats...")

    measurer = Measurer()
    results = {}

    for task_name, scripts in TASKS.items():
        results[task_name] = {}
        for script in scripts:
            logger.info(f"Running task `{task_name}` with script `{script}`")
            script_name = f"scripts/bench/tasks/{task_name}/_{script}.py"
            task = import_module(f"scripts.bench.tasks.{task_name}")
            results[task_name][script] = run_default_task(task, script_name, measurer)

    save_results(results)

    logger.success("Test stats completed!")
    logger.info("Results saved to './results/bench/test-results.json'")


if __name__ == "__main__":
    main()
