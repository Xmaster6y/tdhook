"""
Tests for the weights functionality.
"""

import torch
import torch.nn as nn
import pytest

from tdhook.runtime import HookProgram, HookSpec
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

        context = task_vectors.prepare(pretrained_model)
        with context as hooked_module:
            vector = hooked_module.get_task_vector(finetuned_model)
            alpha = context.compute_alpha(vector)
            assert alpha == 0.1
            inferred_weights = hooked_module.get_weights(vector)
            expected_weights = hooked_module._weights + vector * alpha
            for inferred, expected in zip(
                inferred_weights.flatten_keys().values(), expected_weights.flatten_keys().values()
            ):
                torch.testing.assert_close(inferred, expected)

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

    def test_applied_vectors_report_and_restore_parameter_state(self):
        task_vectors = TaskVectors(
            alphas=[0.5],
            get_test_accuracy=lambda _: 1.0,
            get_control_adequacy=lambda _: True,
        )
        model = nn.Linear(3, 2)
        finetuned = nn.Linear(3, 2)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}

        with task_vectors.prepare(model) as hooked:
            vector = hooked.get_task_vector(finetuned)
            with hooked.with_applied_vectors(vector, alpha=0.5):
                assert hooked.applied_program == HookProgram((HookSpec("", "replace_parameters", None),))
                assert any(not torch.equal(value, original[key]) for key, value in model.state_dict().items())

        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_applied_vectors_restore_after_failure(self):
        task_vectors = TaskVectors(
            alphas=[0.5],
            get_test_accuracy=lambda _: 1.0,
            get_control_adequacy=lambda _: True,
        )
        model = nn.Linear(3, 2)
        finetuned = nn.Linear(3, 2)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}

        with task_vectors.prepare(model) as hooked:
            vector = hooked.get_task_vector(finetuned)
            with pytest.raises(RuntimeError, match="evaluation failed"):
                with hooked.with_applied_vectors(vector, alpha=0.5):
                    raise RuntimeError("evaluation failed")

        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])
