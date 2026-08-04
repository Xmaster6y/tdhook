import pytest

from tdhook.attribution import ActivationMaximisation, IntegratedGradients, Saliency
from tdhook.contexts import HookingContextFactory
from tdhook.execution import ExecutionSpec, GradientMode


def test_base_method_declares_one_optional_gradient_model_pass():
    assert HookingContextFactory().execution_spec == ExecutionSpec()


def test_gradient_methods_own_their_execution_requirements():
    assert Saliency().execution_spec == ExecutionSpec(gradient_mode=GradientMode.REQUIRED)
    assert IntegratedGradients(n_steps=2).execution_spec == ExecutionSpec(gradient_mode=GradientMode.REQUIRED)
    assert IntegratedGradients(n_steps=2, compute_convergence_delta=True).execution_spec == ExecutionSpec(
        model_passes=3,
        gradient_mode=GradientMode.REQUIRED,
    )
    assert ActivationMaximisation(["linear"], n_steps=4).execution_spec == ExecutionSpec(
        model_passes=4,
        gradient_mode=GradientMode.REQUIRED,
    )


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"model_passes": 0}, ValueError, "at least one model pass"),
        ({"model_passes": True}, TypeError, "must be an integer"),
        ({"gradient_mode": "required"}, TypeError, "must be a GradientMode"),
    ],
)
def test_execution_spec_rejects_invalid_requirements(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        ExecutionSpec(**kwargs)
