"""
Tests for the weights functionality.
"""

import torch
import torch.nn as nn

from tdhook.weights.task_vectors import TaskVectors


class TestTaskVectors:
    """Test the TaskVectors class."""

    def test_compute_alpha(self):
        """Test computing alpha."""

        def get_test_accuracy(module):
            return 0.8

        def get_control_adequacy(module):
            return True

        task_vectors = TaskVectors(
            alphas=[0.1, 0.5, 1.0], get_test_accuracy=get_test_accuracy, get_control_adequacy=get_control_adequacy
        )

        pretrained_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        finetuned_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        hooking_context = task_vectors.prepare(pretrained_model)
        with hooking_context as hooked_module:
            vector = hooked_module.get_task_vector(finetuned_model)
            alpha = hooking_context.compute_alpha(vector)
            assert alpha == 0.1

    def test_get_task_vectors(self):
        """Test getting task vectors."""

        def get_test_accuracy(module):
            return 0.8

        def get_control_adequacy(module):
            return True

        task_vectors = TaskVectors(
            alphas=[0.1, 0.5, 1.0], get_test_accuracy=get_test_accuracy, get_control_adequacy=get_control_adequacy
        )
        pretrained_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        finetuned_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        with task_vectors.prepare(pretrained_model) as hooked_module:
            learn_vector = hooked_module.get_task_vector(finetuned_model)
            forget_vector = hooked_module.get_forget_vector(finetuned_model)
            new_weights = hooked_module.get_weights(learn_vector, forget_vector, alpha=0.1)
            for new_v, v in zip(new_weights.flatten_keys().values(), hooked_module._weights.flatten_keys().values()):
                assert torch.allclose(v, new_v)
