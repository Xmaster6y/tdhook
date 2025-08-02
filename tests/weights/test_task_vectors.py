"""
Tests for the weights functionality.
"""

import torch.nn as nn
from tensordict import TensorDict

from tdhook.weights.task_vectors import TaskVectors, TaskVectorsConfig, ComputeAlphaConfig


class TestTaskVectors:
    """Test the TaskVectors class."""

    def test_task_vectors_creation(self):
        """Test creating a TaskVectors instance."""

        def get_test_accuracy(module):
            return 0.8

        def get_control_adequacy(module):
            return True

        compute_alpha_config = ComputeAlphaConfig(
            values=[0.1, 0.5, 1.0], get_test_accuracy=get_test_accuracy, get_control_adequacy=get_control_adequacy
        )

        config = TaskVectorsConfig(compute_alpha_config=compute_alpha_config)
        pretrained_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        task_vectors = TaskVectors(config, pretrained_model)
        assert isinstance(task_vectors, TaskVectors)
        assert task_vectors.config == config

    def test_get_task_vector(self):
        """Test computing task vector."""

        def get_test_accuracy(module):
            return 0.8

        def get_control_adequacy(module):
            return True

        compute_alpha_config = ComputeAlphaConfig(
            values=[0.1, 0.5, 1.0], get_test_accuracy=get_test_accuracy, get_control_adequacy=get_control_adequacy
        )

        config = TaskVectorsConfig(compute_alpha_config=compute_alpha_config)
        pretrained_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        finetuned_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        task_vectors = TaskVectors(config, pretrained_model)
        task_vector = task_vectors.get_task_vector(finetuned_model)

        assert isinstance(task_vector, TensorDict)
