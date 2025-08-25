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


def flatten_results(raw: Dict) -> pd.DataFrame:
    """Return a tidy DataFrame from the benchmark json structure.

    This function flattens the nested JSON structure from benchmark results into a pandas DataFrame
    that can be easily analyzed and plotted. It handles the split spawn experiments (spawn_cpu and spawn_gpu).

    Args:
        raw: Dictionary containing benchmark results with structure:
             task -> library -> parameter_name -> parameter_value -> seed -> {spawn_cpu, spawn_gpu, run_cpu, run_gpu}

    Returns:
        DataFrame with columns: task, lib, parameter, value, seed, and all metrics

    Raises:
        ValueError: If the results json doesn't contain a 'base' key
    """
    if "base" not in raw:
        raise ValueError("results json must contain a 'base' key with baseline numbers")

    rows: List[Dict] = []

    for task, libs in raw.items():
        if task == "base":
            continue

        for lib, params in libs.items():
            for param_name, param_vals in params.items():
                for param_val, seeds in param_vals.items():
                    for seed_str, runs in seeds.items():
                        row = {
                            "task": task,
                            "lib": lib,
                            "parameter": param_name,
                            "value": str(param_val),
                            "seed": int(seed_str),
                            # times (seconds)
                            "spawn_cpu_time": runs["spawn_cpu"]["wall_time"],
                            "spawn_gpu_time": runs["spawn_gpu"]["wall_time"],
                            "run_cpu_time": runs.get("run_cpu", {}).get("wall_time", np.nan),
                            "run_gpu_time": runs.get("run_gpu", {}).get("wall_time", np.nan),
                            # host RAM (MB)
                            "spawn_cpu_ram": runs["spawn_cpu"]["max_ram_used_kb"] / 1024,
                            "spawn_gpu_ram": runs["spawn_gpu"]["max_ram_used_kb"] / 1024,
                            "run_cpu_ram": runs.get("run_cpu", {}).get("max_ram_used_kb", np.nan) / 1024,
                            "run_gpu_ram": runs.get("run_gpu", {}).get("max_ram_used_kb", np.nan) / 1024,
                            # vram (MB)
                            "spawn_cpu_gpu_vram": runs["spawn_cpu"].get("max_gpu_memory_kb", 0.0) / 1024,
                            "spawn_gpu_gpu_vram": runs["spawn_gpu"].get("max_gpu_memory_kb", 0.0) / 1024,
                            "run_gpu_vram": runs.get("run_gpu", {}).get("max_gpu_memory_kb", np.nan) / 1024,
                        }
                        rows.append(row)

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
