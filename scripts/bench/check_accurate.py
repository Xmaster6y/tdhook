"""
Check if tasks are accurate.

Run with:

```
uv run --group scripts -m scripts.bench.check_accurate
```
"""

from loguru import logger
from importlib import import_module


SEEDS = [42, 123, 456]
TASKS = {
    "lrp": ["tdhook", "zennit"],
    "integrated_gradients": ["tdhook", "captum_add", "captum"],
    "gpt2_cache": ["tdhook", "nnsight", "transformer_lens"],
    "mlp_intervene": ["tdhook", "nnsight"],
}


def main(args):
    """Run all benchmarks."""
    logger.info("Checking if tasks are accurate...")

    # Filter tasks if specific ones are requested
    tasks_to_run = TASKS
    if args.tasks:
        tasks_to_run = {k: v for k, v in TASKS.items() if k in args.tasks}

    for task_name, scripts in tasks_to_run.items():
        all_outs = []
        for script in scripts:
            out = import_module(f"scripts.bench.tasks.{task_name}._{script}").main()
            all_outs.append(out)
        assert_accuracy = import_module(f"scripts.bench.tasks.{task_name}").assert_accuracy
        for out in all_outs[1:]:
            assert_accuracy(all_outs[0], out)
        logger.success(f"Task {task_name} is accurate")

    logger.success("All tasks are accurate")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser("check-accurate")
    parser.add_argument(
        "--tasks", nargs="+", choices=list(TASKS.keys()), help="Specific tasks to check (default: all)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
