"""
Tests for the probing functionality.
"""

from typing import Any

import pytest
import torch
import numpy as np

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from tdhook.latent.probing import (
    LinearEstimator,
    LowRankBilinearEstimator,
    MeanDifferenceClassifier,
    Probe,
    ProbeManager,
    Probing,
)
from tdhook.latent.probing.estimators import BilinearEstimator
from tdhook.latent.probing.managers import BilinearProbe, BilinearProbeManager


class ExampleProbe(TensorDictModuleBase):
    """Minimal probe for testing; implements step() for Probing context."""

    in_keys = ["h"]
    out_keys = ["called"]

    def __init__(self):
        super().__init__()
        self.called = False

    def step(self, data: Any, **kwargs) -> None:
        """Probe protocol: called by Probing context hooks."""
        self.called = True

    def forward(self, td: TensorDict) -> TensorDict:
        self.called = True
        td["called"] = torch.ones(td.batch_size, dtype=torch.bool)
        return td


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

        def probe_factory(k, direction):
            probes[k] = ExampleProbe()
            return probes[k]

        context = Probing(key, probe_factory, relative=relative)

        with context.prepare(default_test_model) as hooked_module:
            inputs = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
            hooked_module(inputs)
            assert key in probes
            assert probes[key].called

    def test_probing_pattern(self, default_test_model):
        """Test creating a Probing with pattern."""
        probes = {}

        def probe_factory(k, direction):
            probes[k] = ExampleProbe()
            return probes[k]

        context = Probing("linear1|linear2", probe_factory)

        with context.prepare(default_test_model) as hooked_module:
            inputs = TensorDict({"input": torch.randn(2, 10)}, batch_size=2)
            hooked_module(inputs)

            assert "linear1" in probes
            assert "linear2" in probes
            assert "linear3" not in probes

            assert probes["linear1"].called
            assert probes["linear2"].called


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

        pos_mean = X[y == 1].mean(axis=0)
        neg_mean = X[y == 0].mean(axis=0)
        midpoint = (pos_mean + neg_mean) / 2
        midpoint_proba = classifier.predict_proba(midpoint.reshape(1, -1))
        assert np.isclose(midpoint_proba[0, 1], 0.5)

    def test_unfitted_properties_raise(self):
        classifier = MeanDifferenceClassifier()
        with pytest.raises(ValueError, match="not fitted"):
            _ = classifier.coef_
        with pytest.raises(ValueError, match="not fitted"):
            _ = classifier.intercept_

    def test_multiclass_target_raises(self):
        classifier = MeanDifferenceClassifier()
        X = np.random.randn(4, 3)
        y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        with pytest.raises(ValueError, match="Multiclass"):
            classifier.fit(X, y)

    def test_fit_with_zero_coef_norm_skips_normalization(self):
        X = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
        y = np.array([1, 1, 0, 0])
        classifier = MeanDifferenceClassifier(normalize=True)

        classifier.fit(X, y)

        assert np.array_equal(classifier.coef_, np.array([[0.0, 0.0]]))
        assert np.array_equal(classifier.intercept_, np.array([0.0]))
        proba = classifier.predict_proba(X)
        assert np.allclose(proba, 0.5)

    @pytest.mark.parametrize("y", [np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0])])
    def test_fit_requires_both_classes(self, y):
        X = np.random.randn(4, 3)
        classifier = MeanDifferenceClassifier()
        with pytest.raises(ValueError, match="Both classes must be present"):
            classifier.fit(X, y)


class TestTorchEstimators:
    def test_linear_forward_input_count_validation(self):
        estimator = LinearEstimator(d_latent=4, num_classes=3)
        with pytest.raises(ValueError, match="expects 1 input tensor"):
            estimator.forward(torch.randn(2, 4), torch.randn(2, 4))

    def test_fit_verbose_prints_epoch_progress(self, capsys):
        torch.manual_seed(0)
        estimator = LinearEstimator(d_latent=3, num_classes=2, epochs=10, batch_size=2, verbose=True)
        X = torch.randn(6, 3)
        y = torch.randint(0, 2, (6,))
        estimator.fit(X, y=y)
        captured = capsys.readouterr()
        assert "Epoch 10/10" in captured.out

    def test_loss_shape_mismatch_regression_raises(self):
        estimator = LinearEstimator(d_latent=3, num_classes=None)
        output = torch.randn(5, 1)
        target = torch.randn(5)
        with pytest.raises(ValueError, match="does not match target shape"):
            estimator._loss_fn(output, target)

    def test_loss_shape_mismatch_classification_raises(self):
        estimator = LinearEstimator(d_latent=3, num_classes=3)
        output = torch.randn(5, 2)
        target = torch.randint(0, 3, (5,))
        with pytest.raises(ValueError, match="does not match target shape"):
            estimator._loss_fn(output, target)

    @pytest.mark.parametrize("bias", [True, False])
    def test_low_rank_bilinear_estimator_init_and_forward(self, bias):
        estimator = LowRankBilinearEstimator(d_latent1=4, d_latent2=5, num_classes=3, bias=bias)
        if bias:
            assert estimator.bias is not None
        else:
            assert estimator.bias is None

        h1 = torch.randn(7, 4)
        h2 = torch.randn(7, 5)
        out = estimator(h1, h2)
        assert out.shape == (7, 3)


class TestProbeAndProbeManager:
    def test_probe_invalid_step_type_raises(self):
        probe = Probe(
            estimator=object(),
            predict_callback=lambda preds, labels: None,
        )
        with pytest.raises(ValueError, match="step_type must be 'fit' or 'predict'"):
            probe.step(torch.randn(2, 3), labels=torch.zeros(2), step_type="invalid")

    def test_probe_fit_and_predict_callbacks_are_called(self):
        class MockEstimator:
            def __init__(self):
                self.fit_calls = 0

            def fit(self, X, y):
                self.fit_calls += 1

            def predict(self, X):
                return torch.ones(X.shape[0], dtype=torch.long)

        fit_results = []
        predict_results = []
        probe = Probe(
            estimator=MockEstimator(),
            predict_callback=lambda preds, labels: predict_results.append((preds.shape, labels.shape)),
            fit_callback=lambda preds, labels: fit_results.append((preds.shape, labels.shape)),
        )
        data = torch.randn(4, 2, 3)
        labels = torch.zeros(4, dtype=torch.long)

        probe.step(data, labels=labels, step_type="fit")
        probe.step(data, labels=labels, step_type="predict")

        assert fit_results == [((4,), (4,))]
        assert predict_results == [((4,), (4,))]

    def test_probe_manager_overwrite_and_reset_behaviour(self):
        class DummyEstimator:
            def fit(self, X, y):
                return None

            def predict(self, X):
                return torch.zeros(X.shape[0], dtype=torch.long)

        manager = ProbeManager(
            estimator_class=DummyEstimator,
            estimator_kwargs={},
            compute_metrics=lambda preds, labels: {"acc": float((preds == labels).float().mean().item())},
            allow_overwrite=False,
        )

        probe = manager.probe_factory("linear1", "fwd")
        data = torch.randn(4, 2)
        labels = torch.zeros(4, dtype=torch.long)

        probe.step(data, labels=labels, step_type="predict")
        with pytest.raises(ValueError, match="Metrics for linear1_fwd already exist"):
            probe.step(data, labels=labels, step_type="predict")
        assert "linear1_fwd" in manager.predict_metrics

        manager.reset_metrics()
        probe.step(data, labels=labels, step_type="fit")
        with pytest.raises(ValueError, match="Metrics for linear1_fwd already exist"):
            probe.step(data, labels=labels, step_type="fit")

        with pytest.raises(ValueError, match="already exists"):
            manager.probe_factory("linear1", "fwd")

        assert manager.estimators
        assert manager.fit_metrics
        assert manager.predict_metrics == {}

        manager.reset_estimators()
        manager.reset_metrics()
        assert manager.estimators == {}
        assert manager.fit_metrics == {}
        assert manager.predict_metrics == {}


class TestBilinearProbeManager:
    """Test BilinearProbe and BilinearProbeManager."""

    def test_key_pattern_derivation(self):
        """Test key_pattern is union of keys from pairs, escaped for regex."""
        manager = BilinearProbeManager(
            pairs=[("linear1", "linear2"), ("linear2", "linear3")],
            estimator_class=BilinearEstimator,
            estimator_kwargs={"d_latent1": 20, "d_latent2": 20, "num_classes": 5},
            compute_metrics=lambda p, labels: {"acc": 0.0},
        )
        pattern = manager.key_pattern
        assert "linear1" in pattern
        assert "linear2" in pattern
        assert "linear3" in pattern
        assert pattern == "linear1$|linear2$|linear3$"

    def test_key_pattern_single_pair(self):
        manager = BilinearProbeManager(
            pairs=[("a", "b")],
            estimator_class=BilinearEstimator,
            estimator_kwargs={"d_latent1": 10, "d_latent2": 10, "num_classes": 2},
            compute_metrics=lambda p, labels: {},
        )
        assert manager.key_pattern == "a$|b$"

    def test_key_pattern_self_bilinear(self):
        manager = BilinearProbeManager(
            pairs=[("x", "x")],
            estimator_class=BilinearEstimator,
            estimator_kwargs={"d_latent1": 10, "d_latent2": 10, "num_classes": 2},
            compute_metrics=lambda p, labels: {},
        )
        assert manager.key_pattern == "x$"

    def test_bilinear_probe_self_bilinear_runs_immediately(self):
        """When h1==h2, probe runs fit/predict immediately without waiting."""
        fit_called = []
        pred_called = []

        class MockEstimator:
            def fit(self, h1, h2, y):
                fit_called.append((h1.shape, h2.shape))

            def predict(self, h1, h2):
                pred_called.append((h1.shape, h2.shape))
                return torch.zeros(h1.shape[0], dtype=torch.long)

        probe = BilinearProbe(
            h1_key="x",
            h2_key="x",
            estimator=MockEstimator(),
            predict_callback=lambda p, labels: pred_called.append("pred_cb"),
            fit_callback=lambda p, labels: fit_called.append("fit_cb"),
        )
        data = torch.randn(4, 20)
        labels = torch.randint(0, 5, (4,))

        probe.step(data, key="x", labels=labels, step_type="fit")
        assert fit_called
        probe.step(data, key="x", labels=labels, step_type="predict")
        assert pred_called

    def test_bilinear_probe_cross_pair_waits_then_runs(self):
        """When h1!=h2, probe caches first key then runs when second arrives."""
        run_args = []

        class MockEstimator:
            def fit(self, h1, h2, y):
                run_args.append(("fit", h1.shape, h2.shape))

            def predict(self, h1, h2):
                run_args.append(("predict", h1.shape, h2.shape))
                return torch.zeros(h1.shape[0], dtype=torch.long)

        probe = BilinearProbe(
            h1_key="linear1",
            h2_key="linear2",
            estimator=MockEstimator(),
            predict_callback=lambda p, labels: None,
            fit_callback=lambda p, labels: None,
        )
        h1_data = torch.randn(4, 20)
        h2_data = torch.randn(4, 20)
        labels = torch.randint(0, 5, (4,))

        probe.step(h1_data, key="linear1", labels=labels, step_type="predict")
        assert not run_args
        probe.step(h2_data, key="linear2", labels=labels, step_type="predict")
        assert len(run_args) == 1
        assert run_args[0][1] == h1_data.shape
        assert run_args[0][2] == h2_data.shape

    def test_bilinear_probe_ignores_unrelated_key_and_waiting_state(self):
        class MockEstimator:
            def fit(self, h1, h2, y):
                return None

            def predict(self, h1, h2):
                return torch.zeros(h1.shape[0], dtype=torch.long)

        probe = BilinearProbe(
            h1_key="linear1",
            h2_key="linear2",
            estimator=MockEstimator(),
            predict_callback=lambda p, labels: None,
            fit_callback=lambda p, labels: None,
        )
        labels = torch.zeros(3, dtype=torch.long)
        probe.step(torch.randn(3, 5), key="other", labels=labels, step_type="predict")
        assert not probe.is_waiting
        probe.step(torch.randn(3, 5), key="linear1", labels=labels, step_type="predict")
        assert probe.is_waiting
        waiting = probe.after_all()
        assert waiting == [("linear1", "linear2")]
        assert not probe.is_waiting

    def test_bilinear_probe_invalid_step_type_raises(self):
        class MockEstimator:
            def fit(self, h1, h2, y):
                return None

            def predict(self, h1, h2):
                return torch.zeros(h1.shape[0], dtype=torch.long)

        probe = BilinearProbe(
            h1_key="x",
            h2_key="x",
            estimator=MockEstimator(),
            predict_callback=lambda p, labels: None,
            fit_callback=lambda p, labels: None,
        )
        with pytest.raises(ValueError, match="step_type must be 'fit' or 'predict'"):
            probe.step(torch.randn(2, 4), key="x", labels=torch.zeros(2), step_type="invalid")

    def test_bilinear_probe_after_all_missing_first_key(self):
        class MockEstimator:
            def fit(self, h1, h2, y):
                return None

            def predict(self, h1, h2):
                return torch.zeros(h1.shape[0], dtype=torch.long)

        probe = BilinearProbe(
            h1_key="linear1",
            h2_key="linear2",
            estimator=MockEstimator(),
            predict_callback=lambda p, labels: None,
            fit_callback=lambda p, labels: None,
        )
        labels = torch.zeros(2, dtype=torch.long)
        probe.before_all()
        probe.step(torch.randn(2, 5), key="linear2", labels=labels, step_type="predict")
        waiting = probe.after_all()
        assert waiting == [("linear1", "linear2")]

    def test_bilinear_probe_manager_caching_duplicate_and_resets(self):
        class DummyEstimator:
            def fit(self, h1, h2, y):
                return None

            def predict(self, h1, h2):
                return torch.zeros(h1.shape[0], dtype=torch.long)

        manager = BilinearProbeManager(
            pairs=[("a", "a"), ("a", "b")],
            estimator_class=DummyEstimator,
            estimator_kwargs={},
            compute_metrics=lambda preds, labels: {"n": int(labels.shape[0])},
            allow_overwrite=False,
        )

        _ = manager.probe_factory("a", "fwd")
        # Cache hit path.
        _ = manager.probe_factory("a", "fwd")
        assert ("a", "fwd") in manager._key_to_probes

        with pytest.raises(ValueError, match="already exists"):
            manager._create_pair_probe("a", "a", "fwd")

        dispatcher = manager.probe_factory("a", "bwd")
        labels = torch.zeros(3, dtype=torch.long)
        dispatcher.step(torch.randn(3, 5), labels=labels, step_type="predict")
        with pytest.raises(ValueError, match="Metrics for a_a_bwd already exist"):
            dispatcher.step(torch.randn(3, 5), labels=labels, step_type="predict")

        manager.reset_metrics()
        dispatcher.step(torch.randn(3, 5), labels=labels, step_type="fit")
        with pytest.raises(ValueError, match="Metrics for a_a_bwd already exist"):
            dispatcher.step(torch.randn(3, 5), labels=labels, step_type="fit")

        manager.reset_estimators()
        manager.reset_metrics()
        assert manager.estimators == {}
        assert manager.fit_metrics == {}
        assert manager.predict_metrics == {}
        assert manager._pair_probes == {}
        assert manager._key_to_probes == {}

    def test_bilinear_probe_manager_with_probing_context(self, default_test_model):
        """BilinearProbeManager works with Probing context for fit/predict."""

        def acc_fn(preds, labels_np):
            return {"accuracy": float((np.asarray(preds) == np.asarray(labels_np)).mean())}

        manager = BilinearProbeManager(
            pairs=[("linear1", "linear2")],
            estimator_class=BilinearEstimator,
            estimator_kwargs={
                "d_latent1": 20,
                "d_latent2": 20,
                "num_classes": 5,
                "epochs": 15,
                "verbose": False,
            },
            compute_metrics=acc_fn,
            allow_overwrite=True,
        )
        context = Probing(
            manager.key_pattern,
            manager.probe_factory,
            additional_keys=["labels", "step_type"],
        )

        manager.before_all()
        with context.prepare(default_test_model) as hooked_module:
            for _ in range(3):
                batch = TensorDict(
                    {
                        "input": torch.randn(8, 10),
                        "labels": torch.randint(0, 5, (8,)),
                        "step_type": "fit",
                    },
                    batch_size=8,
                )
                hooked_module(batch)
            for _ in range(2):
                batch = TensorDict(
                    {
                        "input": torch.randn(8, 10),
                        "labels": torch.randint(0, 5, (8,)),
                        "step_type": "predict",
                    },
                    batch_size=8,
                )
                hooked_module(batch)
        manager.after_all()

        assert "linear1_linear2_fwd" in manager.fit_metrics
        assert "linear1_linear2_fwd" in manager.predict_metrics
        assert "accuracy" in manager.fit_metrics["linear1_linear2_fwd"]
        assert "accuracy" in manager.predict_metrics["linear1_linear2_fwd"]

    def test_bilinear_probe_manager_after_all_raises_when_keys_missing(self):
        """after_all raises if some probes still wait on missing keys."""
        manager = BilinearProbeManager(
            pairs=[("linear1", "linear2")],
            estimator_class=BilinearEstimator,
            estimator_kwargs={"d_latent1": 20, "d_latent2": 20, "num_classes": 5},
            compute_metrics=lambda p, labels: {},
        )
        manager.probe_factory("linear1", "fwd")
        manager.before_all()
        probe = list(manager._pair_probes.values())[0]
        probe.step(torch.randn(4, 20), key="linear1", labels=torch.zeros(4), step_type="predict")
        with pytest.raises(ValueError, match="still waiting"):
            manager.after_all()

    def test_bilinear_probe_manager_after_all_succeeds_when_complete(self):
        """after_all succeeds when all pairs received both activations."""
        manager = BilinearProbeManager(
            pairs=[("linear1", "linear2")],
            estimator_class=BilinearEstimator,
            estimator_kwargs={"d_latent1": 20, "d_latent2": 20, "num_classes": 5},
            compute_metrics=lambda p, labels: {},
        )
        manager.probe_factory("linear1", "fwd")
        manager.probe_factory("linear2", "fwd")
        manager.before_all()
        for probe in manager._pair_probes.values():
            probe.step(torch.randn(4, 20), key="linear1", labels=torch.zeros(4), step_type="predict")
            probe.step(torch.randn(4, 20), key="linear2", labels=torch.zeros(4), step_type="predict")
        manager.after_all()
