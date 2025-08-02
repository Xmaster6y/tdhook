import torch
from tensordict import TensorDict

from tdhook.metrics import SensitivityMetric, InfidelityMetric
from tdhook.attribution import InputTimesGradient


class TestSensitivityMetric:
    def test_sensitivity_basic(self, default_test_model):
        """Test basic sensitivity calculation."""
        with InputTimesGradient().prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])
            hooked_module(data)

            sensitivity = SensitivityMetric(perturb_radius=0.01)
            result = sensitivity(hooked_module, data)

            assert "input" in result
            assert result["input"].shape == (2,)
            assert torch.all(result["input"] >= 0)


class TestInfidelityMetric:
    def test_infidelity_basic(self, default_test_model):
        """Test basic infidelity calculation."""
        with InputTimesGradient().prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])
            hooked_module(data)

            infidelity = InfidelityMetric(n_perturb_samples=5)
            result = infidelity(hooked_module, data)

            assert "input" in result
            assert result["input"].shape == (2,)
            assert torch.all(result["input"] >= 0)  # MSE should be non-negative
