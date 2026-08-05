import torch
import pytest
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from tdhook.metrics import SensitivityMetric, InfidelityMetric
from tdhook.attribution import Saliency


class TestSensitivityMetric:
    def test_sensitivity_basic(self, default_test_model):
        """Test basic sensitivity calculation."""
        with Saliency(clean_intermediate_keys=False).prepare(default_test_model) as hooked_module:
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
        with Saliency(clean_intermediate_keys=False).prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])
            hooked_module(data)

            infidelity = InfidelityMetric(n_perturb_samples=5)
            result = infidelity(hooked_module, data)

            assert "input" in result
            assert result["input"].shape == (2,)
            assert torch.all(result["input"] >= 0)  # MSE should be non-negative


class _NestedAttributionModule(TensorDictModuleBase):
    in_keys = [("inputs", "value")]
    out_keys = [("_mod_out", "output"), ("attr", "inputs", "value")]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        value = data.get(("inputs", "value"))
        data.set(("_mod_out", "output"), value.square().sum(dim=-1))
        data.set(("attr", "inputs", "value"), 2 * value)
        return data


def test_metrics_preserve_nested_keys_and_report_exact_additional_calls():
    original = TensorDict({"inputs": {"value": torch.randn(2, 4)}}, batch_size=[2])
    module = _NestedAttributionModule()
    module(original)
    baseline = original.clone()
    module.calls = 0

    sensitivity = SensitivityMetric(perturb_radius=0.01)
    sensitivity_result = sensitivity(module, original)

    assert sensitivity.additional_model_passes(module) == 1
    assert module.calls == 1
    assert sensitivity_result.get(("inputs", "value")).shape == (2,)

    module.calls = 0
    infidelity = InfidelityMetric(n_perturb_samples=3)
    infidelity_result = infidelity(module, original)

    assert infidelity.additional_model_passes(module) == 3
    assert module.calls == 3
    assert infidelity_result.get(("inputs", "value")).shape == (2,)
    assert (infidelity_result.get(("inputs", "value")) >= 0).all()
    assert set(original.keys(True, True)) == set(baseline.keys(True, True))
    for key, value in original.items(True, True):
        torch.testing.assert_close(value, baseline.get(key))


@pytest.mark.parametrize("samples", [0, -1])
def test_infidelity_rejects_non_positive_sample_counts(samples):
    with pytest.raises(ValueError, match="positive"):
        InfidelityMetric(samples)


@pytest.mark.parametrize("samples", [True, 1.5])
def test_infidelity_rejects_non_integer_sample_counts(samples):
    with pytest.raises(TypeError, match="int"):
        InfidelityMetric(samples)


def test_sensitivity_rejects_negative_radius():
    with pytest.raises(ValueError, match="non-negative"):
        SensitivityMetric(-0.1)
