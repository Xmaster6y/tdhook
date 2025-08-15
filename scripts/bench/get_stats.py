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

SEEDS = [42, 123, 456]
TASKS = {
    "gpt2_cache": ["tdhook", "nnsight", "transformer_lens"],
    "integrated_gradients": ["tdhook", "captum_add", "captum"],
    "lrp": ["tdhook", "zennit"],
    "mlp_intervene": ["tdhook", "nnsight"],
}


def run_task(task, script_name: str, measurer: Measurer, seeds):
    """Run a task."""
    default_parameters = task.default_parameters
    results = {}
    for parameter, values in task.impact_parameters.items():
        results[parameter] = {}
        logger.info(f"Running parameter: {parameter}")
        for value in values:
            results[parameter][value] = {}
            for seed in seeds:
                results[parameter][value][seed] = {}
                parameters = {**default_parameters, parameter: value, "seed": seed}
                stats = measurer.measure_script(script_name, parameters)
                results[parameter][value][seed] = stats
    return results


def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    results_dir = Path(output_file).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main(args):
    """Run all benchmarks."""
    logger.info("Starting comprehensive benchmark...")
    logger.info(f"Using seeds: {args.seeds}")

    measurer = Measurer()
    results = {}

    # Filter tasks if specific ones are requested
    tasks_to_run = TASKS
    if args.tasks:
        tasks_to_run = {k: v for k, v in TASKS.items() if k in args.tasks}

    logger.info("Running `_base` task...")
    stats = measurer.measure_script("scripts/bench/tasks/_base.py", {})
    results["base"] = stats

    for task_name, scripts in tasks_to_run.items():
        results[task_name] = {}
        for script in scripts:
            logger.info(f"Running task `{task_name}` with script `{script}`")
            script_name = f"scripts/bench/tasks/{task_name}/_{script}.py"
            task = import_module(f"scripts.bench.tasks.{task_name}")
            results[task_name][script] = run_task(task, script_name, measurer, args.seeds)

    save_results(results, args.output_file)

    logger.success("Benchmark completed!")
    logger.info(f"Results saved to '{args.output_file}'")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("get-stats")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()), help="Specific tasks to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seeds to use for benchmarking")
    parser.add_argument(
        "--output-file", type=str, default="./results/bench/results.json", help="Output file path for results"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
