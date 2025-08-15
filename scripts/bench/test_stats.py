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


def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    results_dir = Path(output_file).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main(args):
    """Run all benchmarks."""
    logger.info("Starting test stats...")

    measurer = Measurer()
    results = {}

    # Filter tasks if specific ones are requested
    tasks_to_run = TASKS
    if args.tasks:
        tasks_to_run = {k: v for k, v in TASKS.items() if k in args.tasks}

    for task_name, scripts in tasks_to_run.items():
        results[task_name] = {}
        for script in scripts:
            logger.info(f"Running task `{task_name}` with script `{script}`")
            script_name = f"scripts/bench/tasks/{task_name}/_{script}.py"
            task = import_module(f"scripts.bench.tasks.{task_name}")
            results[task_name][script] = run_default_task(task, script_name, measurer)

    save_results(results, args.output_file)

    logger.success("Test stats completed!")
    logger.info(f"Results saved to '{args.output_file}'")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("test-stats")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()), help="Specific tasks to run (default: all)")
    parser.add_argument(
        "--output-file", type=str, default="./results/bench/test-results.json", help="Output file path for results"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
