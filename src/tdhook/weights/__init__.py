"""
Weights module for tdhook

Weight analysis and adapters for RL interpretability:
- Task vectors
"""

from .adapters import Adapters
from .pruning import Pruning
from .task_vectors import TaskVectors

__all__ = [
    "Adapters",
    "Pruning",
    "TaskVectors",
]

# TODO: Implment crosscoders
# TODO: Implement circuits tracer from Anthropic
