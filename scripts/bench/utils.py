"""
Utils for benchmarking.
"""

import numpy as np
import torch
import subprocess
from typing import Dict, Any
from loguru import logger


def run_command(command, cwd=None, return_stderr=False):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/zsh",
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {result.stderr}")
            return None
        if return_stderr:
            return result.stdout.strip(), result.stderr.strip()
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Exception running command '{command}': {e}")
        if return_stderr:
            return None, None
        return None


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def assert_accuracy(outs1, outs2):
    """Assert accuracy of two outputs."""
    for out1, out2 in zip(outs1, outs2):
        assert torch.allclose(out1, out2, atol=1e-6)


class Measurer:
    """Memory measurement class that tracks maximum memory usage during execution."""

    def __init__(self, time_fmt: str = "%*E/%P/%M"):
        self.time_fmt = time_fmt

    def _parse_time_output(self, time_output: str) -> Dict[str, Any]:
        """Parse time output."""
        time_output = time_output.split("/")
        return {
            "wall_time": float(time_output[0]),
            "cpu_percent": float(time_output[1].replace("%", "")),
            "max_memory_used_kb": float(time_output[2]),
        }

    def measure_script(self, script_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        params = " ".join([f"--{k}={v}" for k, v in parameters.items()])
        command = f"export TIMEFMT='{self.time_fmt}' && time uv run --group scripts {script_name} {params}"

        _, spawn_stderr = run_command(f"{command} --no-run", return_stderr=True)
        spawn_stats = self._parse_time_output(spawn_stderr.split("\n")[-1])

        _, run_stderr = run_command(f"{command} --run", return_stderr=True)
        run_stats = self._parse_time_output(run_stderr.split("\n")[-1])

        return {
            "spawn": spawn_stats,
            "run": run_stats,
        }
