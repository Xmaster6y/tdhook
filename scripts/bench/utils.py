"""
Utils for benchmarking.
"""

import re
import subprocess
from typing import Dict, Any
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
            "spawn": "--no-run",
            "run_cpu": "--run --no-cuda",
            "run_gpu": "--run --cuda",
        }
        results = {}
        for key, value in args.items():
            _, stderr = run_command(f"{command} {value}", return_stderr=True)
            results[key] = self._parse_time_output(*stderr.split("\n")[-2:])

        return results
