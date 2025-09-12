"""
Tests for the probing functionality.
"""

import pytest
import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensordict import TensorDict

from tdhook.latent.probing import Probing, SklearnProbeManager, MeanDifferenceClassifier


class ExampleProbe:
    def __init__(self):
        self.called = False

    def step(self, data):
        self.called = True


class TestProbing:
    """Test the Probing class."""

    @pytest.mark.parametrize(
        "relative_n_key",
        (
            (False, "td_module.module.linear2"),
            (True, "linear2"),
        ),
    )
    def test_simple_probing(self, default_test_model, relative_n_key):
        """Test creating a Probing."""
        relative, key = relative_n_key

        probes = {}

        def probe_factory(key, direction):
            probes[key] = ExampleProbe()
            return probes[key]

        context = Probing(key, probe_factory, relative=relative)

        with context.prepare(default_test_model) as hooked_module:
            inputs = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
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
        """Test creating a Probing."""
        relative, key = relative_n_key
        storage_key = f"{key}_fwd"

        probe_manager = SklearnProbeManager(LogisticRegression, {}, lambda x, y: {"accuracy": accuracy_score(x, y)})
        context = Probing(key, probe_manager.probe_factory, relative=relative, additional_keys=["labels", "step_type"])

        with context.prepare(default_test_model) as hooked_module:
            inputs = TensorDict(
                {"input": torch.randn(2, 10), "labels": torch.tensor([1, 4]), "step_type": "fit"}, batch_size=2
            )
            hooked_module(inputs)
            assert storage_key in probe_manager.probes
            assert storage_key in probe_manager.fit_metrics
            assert storage_key not in probe_manager.predict_metrics

            inputs["step_type"] = "predict"
            hooked_module(inputs)
            assert storage_key in probe_manager.predict_metrics


class TestMeanDifferenceClassifier:
    """Test the MeanDifferenceClassifier class."""

    @pytest.mark.parametrize(
        "X,y,normalize,expected_coef,expected_intercept",
        [
            (
                np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]),
                np.array([1, 1, 0, 0]),
                True,
                np.array([[1.0, 0.0]]),
                np.array([0.0]),
            ),
            (
                np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]),
                np.array([1, 1, 0, 0]),
                False,
                np.array([[3.0, 0.0]]),
                np.array([0.0]),
            ),
        ],
    )
    def test_fit_and_predict(self, X, y, normalize, expected_coef, expected_intercept):
        """Test fit and predict functionality with expected coefficients and intercept."""
        classifier = MeanDifferenceClassifier(normalize=normalize)
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        assert np.array_equal(predictions, [True, True, False, False])

        proba = classifier.predict_proba(X)
        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

        assert np.allclose(classifier.coef_, expected_coef)
        assert np.allclose(classifier.intercept_, expected_intercept)

        # Test that midpoint between positive and negative means has proba 0.5
        pos_mean = X[y == 1].mean(axis=0)
        neg_mean = X[y == 0].mean(axis=0)
        midpoint = (pos_mean + neg_mean) / 2
        midpoint_proba = classifier.predict_proba(midpoint.reshape(1, -1))
        assert np.isclose(midpoint_proba[0, 1], 0.5)
