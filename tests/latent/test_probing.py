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
from tdhook.runtime import HookProgram, HookSpec


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

        assert hooked_module.hooking_context.program == HookProgram((HookSpec(key, "probe", "fwd"),))
        assert all(not submodule._forward_hooks for submodule in default_test_model.modules())

    def test_inspection_callbacks_are_inert_and_key_pattern_is_mutable(self, default_test_model):
        created = []

        def probe_factory(key, direction):
            created.append((key, direction))
            return ExampleProbe()

        method = Probing("linear1", probe_factory)
        assert method.key_pattern == "linear1"
        method.key_pattern = "linear2"
        assert method.key_pattern == "linear2"

        context = method.prepare(default_test_model)
        with context.inspect() as prepared:
            prepared(TensorDict({"input": torch.ones(2, 10)}, batch_size=[2]))

        assert created == []

        with pytest.raises(TypeError, match="additional_keys"):
            Probing("linear1", probe_factory, additional_keys=[object()])

    def test_additional_keys_require_the_root_capture(self, default_test_model):
        context = Probing("linear1", lambda *_: ExampleProbe(), additional_keys=["labels"])

        with context.prepare(default_test_model) as prepared:
            prepared(
                TensorDict(
                    {"input": torch.ones(1, 10), "labels": torch.ones(1)},
                    batch_size=[1],
                )
            )
            with pytest.raises(RuntimeError, match="before additional inputs were captured"):
                default_test_model.linear1(torch.ones(1, 10))

    def test_backward_probes_keep_additional_keys_until_backward(self, default_test_model):
        observed = []

        class BackwardProbe:
            def step(self, gradient, **metadata):
                observed.append((gradient, metadata["labels"]))

        context = Probing(
            "linear1",
            lambda *_: BackwardProbe(),
            directions=["bwd_pre"],
            additional_keys=["labels"],
        )
        data = TensorDict(
            {"input": torch.ones(1, 10, requires_grad=True), "labels": torch.tensor([3])},
            batch_size=[1],
        )

        with context.prepare(default_test_model) as prepared:
            prepared(data)["output"].sum().backward()

        assert len(observed) == 1
        torch.testing.assert_close(observed[0][1], data["labels"])

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

    def test_probe_manager_reuses_fitted_estimator_across_bindings(self, default_test_model):
        class DummyEstimator:
            def __init__(self):
                self.fitted = False

            def fit(self, data, y):
                self.fitted = True

            def predict(self, data):
                assert self.fitted
                return torch.zeros(data.shape[0], dtype=torch.long)

        manager = ProbeManager(
            estimator_class=DummyEstimator,
            estimator_kwargs={},
            compute_metrics=lambda predictions, labels: {"accuracy": float((predictions == labels).float().mean())},
        )
        context = Probing(
            "linear2",
            manager.probe_factory,
            additional_keys=["labels", "step_type"],
        )

        fit_data = TensorDict(
            {
                "input": torch.randn(4, 10),
                "labels": torch.zeros(4, dtype=torch.long),
                "step_type": "fit",
            },
            batch_size=4,
        )
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(fit_data)
        estimator = manager.estimators["linear2_fwd"]

        predict_data = fit_data.clone().set("step_type", "predict")
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(predict_data)

        assert manager.estimators["linear2_fwd"] is estimator
        assert manager.results["fit", "linear2_fwd", "accuracy"].item() == 1.0
        assert manager.results["predict", "linear2_fwd", "accuracy"].item() == 1.0


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

    @pytest.mark.parametrize("training_context", [torch.no_grad, torch.inference_mode])
    def test_fit_owns_grad_context_when_model_forward_disables_it(self, training_context):
        estimator = LinearEstimator(d_latent=3, num_classes=2, epochs=1, batch_size=2)

        with training_context():
            X = torch.randn(4, 3)
            y = torch.randint(0, 2, (4,))
            estimator.fit(X, y=y)

        assert any(parameter.grad is not None for parameter in estimator.parameters())

    def test_fit_rejects_an_estimator_constructed_in_inference_mode(self):
        with torch.inference_mode():
            estimator = LinearEstimator(d_latent=3, num_classes=2, epochs=1, batch_size=2)

        with pytest.raises(RuntimeError, match="must be constructed outside torch.inference_mode"):
            estimator.fit(torch.randn(4, 3), y=torch.randint(0, 2, (4,)))

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

    def test_probe_manager_results_and_independent_resets(self):
        class DummyEstimator:
            def fit(self, X, y):
                return None

            def predict(self, X):
                return torch.zeros(X.shape[0], dtype=torch.long)

        manager = ProbeManager(
            estimator_class=DummyEstimator,
            estimator_kwargs={},
            compute_metrics=lambda preds, labels: {"acc": float((preds == labels).float().mean().item())},
        )

        probe = manager.probe_factory("linear1", "fwd")
        data = torch.randn(4, 2)
        labels = torch.zeros(4, dtype=torch.long)

        result = probe.step(data, labels=labels, step_type="predict")
        assert isinstance(result, TensorDict)
        assert result["acc"].item() == 1.0
        with pytest.raises(ValueError, match="Result for linear1_fwd already exists"):
            probe.step(data, labels=labels, step_type="predict")
        assert manager.results["predict", "linear1_fwd", "acc"].item() == 1.0

        manager.reset_results()
        probe.step(data, labels=labels, step_type="fit")
        with pytest.raises(ValueError, match="Result for linear1_fwd already exists"):
            probe.step(data, labels=labels, step_type="fit")

        assert manager.probe_factory("linear1", "fwd") is probe

        assert manager.estimators
        assert manager.results["fit", "linear1_fwd", "acc"].item() == 1.0
        assert manager.results.get("predict", None) is None

        manager.reset_estimators()
        manager.reset_results()
        assert manager.estimators == {}
        assert manager._probes == {}
        assert list(manager.results.keys(True, True)) == []

    def test_probe_manager_rejects_non_mapping_metrics(self):
        class DummyEstimator:
            def predict(self, data):
                return data

        manager = ProbeManager(
            estimator_class=DummyEstimator,
            estimator_kwargs={},
            compute_metrics=lambda predictions, labels: 1.0,
        )

        with pytest.raises(TypeError, match="compute_metrics must return a dict"):
            manager.probe_factory("linear1", "fwd").step(
                torch.ones(2, 1),
                labels=torch.ones(2, 1),
                step_type="predict",
            )


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
        with pytest.raises(ValueError, match="Result for a_a_bwd already exists"):
            dispatcher.step(torch.randn(3, 5), labels=labels, step_type="predict")

        manager.reset_results()
        dispatcher.step(torch.randn(3, 5), labels=labels, step_type="fit")
        with pytest.raises(ValueError, match="Result for a_a_bwd already exists"):
            dispatcher.step(torch.randn(3, 5), labels=labels, step_type="fit")

        manager.reset_estimators()
        manager.reset_results()
        assert manager.estimators == {}
        assert list(manager.results.keys(True, True)) == []
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
            overwrite_results=True,
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

        assert manager.results["fit", "linear1_linear2_fwd", "accuracy"].ndim == 0
        assert manager.results["predict", "linear1_linear2_fwd", "accuracy"].ndim == 0
        assert hooked_module.hooking_context.program == HookProgram(
            (
                HookSpec("", "capture_inputs", "fwd_pre"),
                HookSpec("linear1", "probe", "fwd"),
                HookSpec("linear2", "probe", "fwd"),
            )
        )

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
