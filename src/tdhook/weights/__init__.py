"""
Weights module for tdhook

Weight analysis and adapters for RL interpretability:
- Task vectors
"""

from .task_vectors import TaskVectors

__all__ = [
    "TaskVectors",
]

# TODO: Implment SAE
# TODO: Implment crosscoders
# TODO: Implement Clement's model diffing
# TODO: Implement circuits tracer from Anthropic
