"""
Tests for the probing functionality.
"""

import pytest
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensordict import TensorDict

from tdhook.acteng.linear_probing import LinearProbing, SklearnProbeManager


class TestProbe:
    def __init__(self):
        self.called = False

    def step(self, data):
        self.called = True


class TestLinearProbing:
    """Test the LinearProbing class."""

    @pytest.mark.parametrize(
        "relative_n_key",
        (
            (False, "td_module.module.linear2"),
            (True, "linear2"),
        ),
    )
    def test_simple_probing(self, default_test_model, relative_n_key):
        """Test creating a LinearProbing."""
        relative, key = relative_n_key

        probes = {}

        def probe_factory(key, direction):
            probes[key] = TestProbe()
            return probes[key]

        context = LinearProbing(key, probe_factory, relative=relative)

        with context.prepare(default_test_model) as hooked_module:
            inputs = TensorDict({"input": torch.randn(2, 10)})
            hooked_module(inputs)
            assert key in probes
            assert probes[key].called

    @pytest.mark.parametrize(
        "relative_n_key",
        (
            (False, "td_module.module.linear2"),
            (True, "linear2"),
        ),
    )
    def test_sklearn_probing(self, default_test_model, relative_n_key):
        """Test creating a LinearProbing."""
        relative, key = relative_n_key
        storage_key = f"{key}_fwd"

        probe_manager = SklearnProbeManager(LogisticRegression, {}, lambda x, y: {"accuracy": accuracy_score(x, y)})
        context = LinearProbing(
            key, probe_manager.probe_factory, relative=relative, additional_keys=["labels", "step_type"]
        )

        with context.prepare(default_test_model) as hooked_module:
            inputs = TensorDict({"input": torch.randn(2, 10), "labels": torch.tensor([1, 4]), "step_type": "fit"})
            hooked_module(inputs)
            assert storage_key in probe_manager.probes
            assert storage_key not in probe_manager.metrics

            inputs["step_type"] = "predict"
            hooked_module(inputs)
            assert storage_key in probe_manager.metrics
