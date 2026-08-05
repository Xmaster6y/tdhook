import re
from typing import Callable, Optional, Any, Dict, List, Tuple

from tensordict import TensorDict

from tdhook.hooks import (
    HookDirection,
)


class Probe:
    def __init__(
        self,
        estimator: Any,
        predict_callback: Callable[[Any, Any], Any],
        fit_callback: Optional[Callable[[Any, Any], Any]] = None,
        data_preprocess_callback: Optional[Callable[[Any], Any]] = None,
    ):
        self._estimator = estimator
        self._predict_callback = predict_callback
        self._fit_callback = fit_callback
        self._data_preprocess_callback = data_preprocess_callback or self._default_data_preprocess_callback

    def step(self, data: Any, **kwargs):
        labels = kwargs.get("labels")
        step_type = kwargs.get("step_type")
        if step_type not in ("fit", "predict"):
            raise ValueError(f"step_type must be 'fit' or 'predict', got {step_type!r}")
        data = self._data_preprocess_callback(data)
        if step_type == "fit":
            self._estimator.fit(data, y=labels)
            if self._fit_callback is not None:
                return self._fit_callback(self._estimator.predict(data), labels)
            return None
        return self._predict_callback(self._estimator.predict(data), labels)

    def _default_data_preprocess_callback(self, data: Any) -> Any:
        return data.detach().flatten(1)


class BilinearProbe(Probe):
    """Probe for bilinear estimators; caches first activation when h1 != h2."""

    def __init__(
        self,
        h1_key: str,
        h2_key: str,
        estimator: Any,
        predict_callback: Callable[[Any, Any], Any],
        fit_callback: Optional[Callable[[Any, Any], Any]] = None,
        data_preprocess_callback: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(estimator, predict_callback, fit_callback, data_preprocess_callback)
        self._h1_key = h1_key
        self._h2_key = h2_key
        self._cached: Dict[str, Any] = {}
        self._waiting_active = False

    def step(self, data: Any, key: str, labels: Any, step_type: str, **kwargs):
        data = self._data_preprocess_callback(data)
        if self._h1_key == self._h2_key:
            return self._run(data, data, labels, step_type)

        if key == self._h1_key:
            self._cached["h1"] = data
        elif key == self._h2_key:
            self._cached["h2"] = data
        else:
            return

        if "h1" in self._cached and "h2" in self._cached:
            h1, h2 = self._cached["h1"], self._cached["h2"]
            self._cached.clear()
            return self._run(h1, h2, labels, step_type)
        return None

    def _run(self, h1: Any, h2: Any, labels: Any, step_type: str):
        if step_type not in ("fit", "predict"):
            raise ValueError(f"step_type must be 'fit' or 'predict', got {step_type!r}")
        if step_type == "fit":
            self._estimator.fit(h1, h2, y=labels)
            if self._fit_callback is not None:
                return self._fit_callback(self._estimator.predict(h1, h2), labels)
            return None
        return self._predict_callback(self._estimator.predict(h1, h2), labels)

    def before_all(self):
        self._waiting_active = True
        self._cached.clear()

    def after_all(self) -> List[Tuple[str, str]]:
        self._waiting_active = False
        still_waiting = []
        if self._h1_key != self._h2_key and self._cached:
            missing = []
            if "h1" not in self._cached:
                missing.append(self._h1_key)
            if "h2" not in self._cached:
                missing.append(self._h2_key)
            if missing:
                still_waiting.append((self._h1_key, self._h2_key))
        self._cached.clear()
        return still_waiting

    @property
    def is_waiting(self) -> bool:
        return (
            self._h1_key != self._h2_key
            and ("h1" in self._cached or "h2" in self._cached)
            and not ("h1" in self._cached and "h2" in self._cached)
        )


class ProbeManager:
    """Own persistent estimators and TensorDict-native metric results."""

    def __init__(
        self,
        estimator_class: Any,
        estimator_kwargs: dict,
        compute_metrics: Callable[[Any, Any], Dict[str, Any]],
        overwrite_results: bool = False,
        data_preprocess_callback: Callable[[Any], Any] = None,
    ):
        self._estimator_class = estimator_class
        self._estimator_kwargs = estimator_kwargs
        self._compute_metrics = compute_metrics
        self._overwrite_results = overwrite_results
        self._data_preprocess_callback = data_preprocess_callback

        self._estimators = {}
        self._probes = {}
        self._results = TensorDict(batch_size=[])

    @property
    def estimators(self) -> dict[str, Any]:
        return self._estimators

    @property
    def results(self) -> TensorDict:
        """Return TensorDict-native fit and predict metrics."""

        return self._results

    def probe_factory(self, key: str, direction: HookDirection) -> Probe:
        _key = f"{key}_{direction}"
        if _key in self._probes:
            return self._probes[_key]
        estimator = self._estimator_class(**self._estimator_kwargs)
        self._estimators[_key] = estimator

        def predict_callback(predictions: Any, labels: Any):
            return self._record_result("predict", _key, predictions, labels)

        def fit_callback(predictions: Any, labels: Any):
            return self._record_result("fit", _key, predictions, labels)

        probe = Probe(estimator, predict_callback, fit_callback, self._data_preprocess_callback)
        self._probes[_key] = probe
        return probe

    def _record_result(self, phase: str, key: str, predictions: Any, labels: Any) -> TensorDict:
        result_key = (phase, key)
        if self._results.get(result_key, None) is not None and not self._overwrite_results:
            raise ValueError(
                f"Result for {key} already exists in phase {phase!r}; "
                "call reset_results() or use overwrite_results=True"
            )
        metrics = self._compute_metrics(predictions, labels)
        if not isinstance(metrics, dict):
            raise TypeError("compute_metrics must return a dict")
        result = TensorDict.from_dict(metrics, batch_size=[])
        self._results.set(result_key, result)
        return result

    def reset_estimators(self):
        """Discard fitted estimators and their stable probe objects."""

        self._estimators = {}
        self._probes = {}

    def reset_results(self):
        """Discard fit and prediction metrics without changing estimators."""

        self._results = TensorDict(batch_size=[])


class BilinearProbeManager(ProbeManager):
    """Manager for bilinear probes; one probe per (h1, h2) pair."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        estimator_class: Any,
        estimator_kwargs: dict,
        compute_metrics: Callable[[Any, Any], Dict[str, Any]],
        overwrite_results: bool = False,
        data_preprocess_callback: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(
            estimator_class=estimator_class,
            estimator_kwargs=estimator_kwargs,
            compute_metrics=compute_metrics,
            overwrite_results=overwrite_results,
            data_preprocess_callback=data_preprocess_callback,
        )
        self._pairs = list(pairs)
        self._pair_probes: Dict[Tuple[str, str, str], BilinearProbe] = {}
        self._key_to_probes: Dict[Tuple[str, str], List[BilinearProbe]] = {}

    @property
    def key_pattern(self) -> str:
        """Read-only regex alternation of all keys present in pairs."""
        keys = sorted(set(k for pair in self._pairs for k in pair))
        return "|".join(re.escape(k) + "$" for k in keys)

    def probe_factory(self, key: str, direction: HookDirection) -> Probe:
        key_dir = (key, direction)
        if key_dir in self._key_to_probes:
            probes = self._key_to_probes[key_dir]
        else:
            probes = []
            for h1, h2 in self._pairs:
                if key not in (h1, h2):
                    continue
                pair_key = (h1, h2, direction)
                if pair_key not in self._pair_probes:
                    probe = self._create_pair_probe(h1, h2, direction)
                    self._pair_probes[pair_key] = probe
                probes.append(self._pair_probes[pair_key])
            self._key_to_probes[key_dir] = probes

        def dispatcher_step(data: Any, **kwargs):
            for probe in probes:
                probe.step(data, key=key, **kwargs)

        class DispatcherProbe:
            def step(self, data: Any, **kwargs):
                dispatcher_step(data, **kwargs)

        return DispatcherProbe()

    def _create_pair_probe(self, h1: str, h2: str, direction: HookDirection) -> BilinearProbe:
        pair_key = f"{h1}_{h2}_{direction}"
        if pair_key in self._estimators:
            raise ValueError(f"Probe {pair_key} already exists; call reset_estimators() before replacing it")
        estimator = self._estimator_class(**self._estimator_kwargs)
        self._estimators[pair_key] = estimator

        def predict_callback(predictions: Any, labels: Any):
            return self._record_result("predict", pair_key, predictions, labels)

        def fit_callback(predictions: Any, labels: Any):
            return self._record_result("fit", pair_key, predictions, labels)

        return BilinearProbe(
            h1_key=h1,
            h2_key=h2,
            estimator=estimator,
            predict_callback=predict_callback,
            fit_callback=fit_callback,
            data_preprocess_callback=self._data_preprocess_callback,
        )

    def before_all(self):
        """Initialize waiting state on all bilinear probes for a run."""
        for probe in self._pair_probes.values():
            probe.before_all()

    def after_all(self):
        """Clear waiting state and raise if any probes still wait on missing keys."""
        still_waiting: List[Tuple[str, str]] = []
        for probe in self._pair_probes.values():
            still_waiting.extend(probe.after_all())
        if still_waiting:
            keys = sorted(set(k for pair in still_waiting for k in pair))
            raise ValueError(f"Bilinear probes still waiting on keys: {keys}. Unresolved pairs: {still_waiting}")

    def reset_estimators(self):
        super().reset_estimators()
        self._pair_probes.clear()
        self._key_to_probes.clear()
