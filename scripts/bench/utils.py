"""
Utils for benchmarking.
"""

import re
import subprocess
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger


def run_command(command, cwd=None, return_stderr=False):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {result.stderr}")
            if return_stderr:
                return None, None
            return None
        if return_stderr:
            return result.stdout.strip(), result.stderr.strip()
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Exception running command '{command}': {e}")
        if return_stderr:
            return None, None
        return None


def _extract_metrics_from_run(runs: Dict) -> Dict:
    """Extract metrics from a single run dictionary.

    Args:
        runs: Dictionary containing run data with spawn_cpu, spawn_gpu, run_cpu, run_gpu

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}

    # Extract spawn metrics (same for both base and task experiments)
    metrics.update(
        {
            "spawn_cpu_time": runs["spawn_cpu"]["wall_time"],
            "spawn_gpu_time": runs["spawn_gpu"]["wall_time"],
            "spawn_cpu_ram": runs["spawn_cpu"]["max_ram_used_kb"] / 1024,
            "spawn_gpu_ram": runs["spawn_gpu"]["max_ram_used_kb"] / 1024,
            "spawn_gpu_vram": runs["spawn_gpu"].get("max_gpu_memory_kb", 0.0) / 1024,
        }
    )

    # Extract run metrics (same for both base and task experiments)
    metrics.update(
        {
            "run_cpu_time": runs.get("run_cpu", {}).get("wall_time", np.nan),
            "run_gpu_time": runs.get("run_gpu", {}).get("wall_time", np.nan),
            "run_cpu_ram": runs.get("run_cpu", {}).get("max_ram_used_kb", np.nan) / 1024,
            "run_gpu_ram": runs.get("run_gpu", {}).get("max_ram_used_kb", np.nan) / 1024,
            "run_gpu_vram": runs.get("run_gpu", {}).get("max_gpu_memory_kb", np.nan) / 1024,
        }
    )

    # Calculate run_only metrics (actual execution time minus spawn overhead)
    if "task" in runs or "run_cpu" in runs:
        # This is a task experiment, calculate run_only metrics for time only
        run_cpu_time = runs.get("run_cpu", {}).get("wall_time", np.nan)
        run_gpu_time = runs.get("run_gpu", {}).get("wall_time", np.nan)

        # Calculate run_only metrics and filter out negative values
        run_only_cpu_time = np.nan if np.isnan(run_cpu_time) else run_cpu_time - runs["spawn_cpu"]["wall_time"]
        run_only_gpu_time = run_gpu_time - runs["spawn_gpu"]["wall_time"] if not np.isnan(run_gpu_time) else np.nan

        # Filter out negative values (they don't make sense for run_only)
        if run_only_cpu_time is not None and run_only_cpu_time < 0:
            run_only_cpu_time = np.nan
        if run_only_gpu_time is not None and run_only_gpu_time < 0:
            run_only_gpu_time = np.nan

        metrics.update(
            {
                "run_only_cpu_time": run_only_cpu_time,
                "run_only_gpu_time": run_only_gpu_time,
            }
        )
    else:
        # This is the base experiment
        # For run_only time metrics: run_only = run_time - spawn_time
        # In base case, run_time is essentially the same as spawn_time (no actual task execution)
        # So run_only should represent the spawn time values
        metrics.update(
            {
                "run_only_cpu_time": runs["spawn_cpu"]["wall_time"],
                "run_only_gpu_time": runs["spawn_gpu"]["wall_time"],
            }
        )

    return metrics


def flatten_results(raw: Dict) -> pd.DataFrame:
    """Return a tidy DataFrame from the benchmark json structure.

    This function flattens the nested JSON structure from benchmark results into a pandas DataFrame
    that can be easily analyzed and plotted. It handles the split spawn experiments (spawn_cpu and spawn_gpu).
    It also includes the base experiment data for baseline comparison.

    Args:
        raw: Dictionary containing benchmark results with structure:
             base -> seed -> {spawn_cpu, spawn_gpu, run_cpu, run_gpu}
             task -> library -> parameter_name -> parameter_value -> seed -> {spawn_cpu, spawn_gpu, run_cpu, run_gpu}

    Returns:
        DataFrame with columns: task, lib, parameter, value, seed, and all metrics

    Raises:
        ValueError: If the results json doesn't contain a 'base' key
    """
    if "base" not in raw:
        raise ValueError("results json must contain a 'base' key with baseline numbers")

    rows: List[Dict] = []

    # Handle base experiment first
    base_runs = raw["base"]

    # Base experiments are mapped from seed → runs.
    for seed_str, runs in base_runs.items():
        base_metrics = _extract_metrics_from_run(runs)
        base_row = {
            "task": "base",
            "lib": "base",
            "parameter": "baseline",
            "value": "baseline",
            "seed": int(seed_str),
            **base_metrics,
        }
        rows.append(base_row)

    # Handle task experiments
    for task, libs in raw.items():
        if task == "base":
            continue

        for lib, params in libs.items():
            for param_name, param_vals in params.items():
                for param_val, seeds in param_vals.items():
                    for seed_str, runs in seeds.items():
                        task_metrics = _extract_metrics_from_run(runs)
                        task_row = {
                            "task": task,
                            "lib": lib,
                            "parameter": param_name,
                            "value": str(param_val),
                            "seed": int(seed_str),
                            **task_metrics,
                        }
                        rows.append(task_row)

    return pd.DataFrame(rows)


class Measurer:
    """Memory measurement class that tracks maximum memory usage during execution."""

    def __init__(self, time_fmt: str = "%E/%P/%M"):
        self.time_fmt = time_fmt

    def _parse_time_output(self, cuda_log: str, time_output: str) -> Dict[str, Any]:
        """Parse time output."""
        gpu_memory_match = re.search(r"Max GPU memory: (\d+\.\d+) KB", cuda_log)
        if gpu_memory_match:
            max_gpu_memory = float(gpu_memory_match.group(1))
        else:
            max_gpu_memory = 0.0

        time_output = time_output.split("/")
        mins, secs = time_output[0].split(":")
        return {
            "wall_time": float(mins) * 60 + float(secs),
            "cpu_percent": float(time_output[1].replace("%", "")),
            "max_ram_used_kb": float(time_output[2]),
            "max_gpu_memory_kb": max_gpu_memory,
        }

    def measure_script(self, script_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        params = " ".join([f"--{k}={v}" for k, v in parameters.items()])
        command = f"time -f '{self.time_fmt}' .venv/bin/python {script_name} {params}"

        args = {
            "spawn_cpu": "--no-run --no-cuda",
            "spawn_gpu": "--no-run --cuda",
            "run_cpu": "--run --no-cuda",
            "run_gpu": "--run --cuda",
        }
        results = {}
        for key, value in args.items():
            _, stderr = run_command(f"{command} {value}", return_stderr=True)
            results[key] = self._parse_time_output(*stderr.split("\n")[-2:])

        return results
