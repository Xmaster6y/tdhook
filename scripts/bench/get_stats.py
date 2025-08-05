"""
Get tasks results stats.

Run with:

```
uv run --group scripts -m scripts.bench.get_stats
```
"""

from typing import Dict
import json
from pathlib import Path
from loguru import logger
from importlib import import_module

from .utils import Measurer

# Fixed seeds for reproducibility
SEEDS = [42, 123, 456]
TASKS = {
    "mlp_intervene": ["nnsight", "tdhook"],
}
PARAMETERS = {
    "width": [2, 10],  # [2, 10, 50, 100, 500], # [10, 100, 1000, 10_000, 100_000],
    "height": [2, 10],
    "batch_size": [2, 10],
}
DEFAULT_PARAMETERS = {
    "width": 10,
    "height": 10,
    "batch_size": 10,
}


def run_task(task, script_name: str, measurer: Measurer):
    """Run a task."""
    all_parameters = {**PARAMETERS, "variation": task.variations}
    default_parameters = {**DEFAULT_PARAMETERS, "variation": task.default_variation}
    results = {}
    for parameter, values in all_parameters.items():
        results[parameter] = {}
        logger.info(f"Running parameter: {parameter}")
        for value in values:
            results[parameter][value] = {}
            for seed in SEEDS:
                results[parameter][value][seed] = {}
                parameters = {**default_parameters, parameter: value, "seed": seed}
                stats = measurer.measure_script(script_name, parameters)
                results[parameter][value][seed] = stats
    return results


def save_results(results: Dict):
    """Save results to JSON file."""
    results_dir = Path("./results/bench")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    """Run all benchmarks."""
    logger.info("Starting comprehensive benchmark...")
    logger.info(f"Using seeds: {SEEDS}")

    measurer = Measurer()
    results = {}

    logger.info("Running `_base` task...")
    stats = measurer.measure_script("scripts/bench/tasks/_base.py", {})
    results["base"] = stats

    for task_name, scripts in TASKS.items():
        results[task_name] = {}
        for script in scripts:
            logger.info(f"Running task `{task_name}` with script `{script}`")
            script_name = f"scripts/bench/tasks/{task_name}/_{script}.py"
            task = import_module(f"scripts.bench.tasks.{task_name}")
            results[task_name][script] = run_task(task, script_name, measurer)

    save_results(results)

    logger.success("Benchmark completed!")
    logger.info("Results saved to './results/bench/results.json'")


if __name__ == "__main__":
    main()
