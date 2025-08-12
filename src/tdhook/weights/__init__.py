"""
Weights module for tdhook

Weight analysis and adapters for RL interpretability:
- Task vectors
"""

from .task_vectors import TaskVectors, TaskVectorsConfig, ComputeAlphaConfig

__all__ = [
    "TaskVectors",
    "TaskVectorsConfig",
    "ComputeAlphaConfig",
]

# TODO: Implement Clement's model diffing
# TODO: Implement circuits tracer from Anthropic
