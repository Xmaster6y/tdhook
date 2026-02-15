import re
from typing import Callable, Optional, Any, Dict, List, Tuple

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
                self._fit_callback(self._estimator.predict(data), labels)
        elif step_type == "predict":
            self._predict_callback(self._estimator.predict(data), labels)
        else:
            raise ValueError(f"Invalid step type: {step_type}")

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
            self._run(data, data, labels, step_type)
            return

        if key == self._h1_key:
            self._cached["h1"] = data
        elif key == self._h2_key:
            self._cached["h2"] = data
        else:
            return

        if "h1" in self._cached and "h2" in self._cached:
            h1, h2 = self._cached["h1"], self._cached["h2"]
            self._cached.clear()
            self._run(h1, h2, labels, step_type)

    def _run(self, h1: Any, h2: Any, labels: Any, step_type: str):
        if step_type not in ("fit", "predict"):
            raise ValueError(f"step_type must be 'fit' or 'predict', got {step_type!r}")
        if step_type == "fit":
            self._estimator.fit(h1, h2, y=labels)
            if self._fit_callback is not None:
                self._fit_callback(self._estimator.predict(h1, h2), labels)
        elif step_type == "predict":
            self._predict_callback(self._estimator.predict(h1, h2), labels)
        else:
            raise ValueError(f"Invalid step type: {step_type}")

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
    def __init__(
        self,
        estimator_class: Any,
        estimator_kwargs: dict,
        compute_metrics: Callable[[Any, Any], Dict[str, Any]],
        allow_overwrite: bool = False,
        data_preprocess_callback: Callable[[Any], Any] = None,
    ):
        self._estimator_class = estimator_class
        self._estimator_kwargs = estimator_kwargs
        self._compute_metrics = compute_metrics
        self._allow_overwrite = allow_overwrite
        self._data_preprocess_callback = data_preprocess_callback

        self._estimators = {}
        self._fit_metrics = {}
        self._predict_metrics = {}

    @property
    def estimators(self) -> dict[str, Any]:
        return self._estimators

    @property
    def fit_metrics(self) -> dict[str, Any]:
        return self._fit_metrics

    @property
    def predict_metrics(self) -> dict[str, Any]:
        return self._predict_metrics

    def probe_factory(self, key: str, direction: HookDirection) -> Probe:
        _key = f"{key}_{direction}"
        if _key in self._estimators and not self._allow_overwrite:
            raise ValueError(
                f"Probe {_key} already exists, call reset_estimators() to reset the estimators or use allow_overwrite=True to overwrite the existing estimators"
            )
        estimator = self._estimator_class(**self._estimator_kwargs)
        self._estimators[_key] = estimator

        def predict_callback(predictions: Any, labels: Any):
            nonlocal self
            if _key in self._predict_metrics and not self._allow_overwrite:
                raise ValueError(
                    f"Metrics for {_key} already exist, call reset_metrics() to reset the metrics or use allow_overwrite=True to overwrite the existing metrics"
                )
            self._predict_metrics[_key] = self._compute_metrics(predictions, labels)

        def fit_callback(predictions: Any, labels: Any):
            nonlocal self
            if _key in self._fit_metrics and not self._allow_overwrite:
                raise ValueError(
                    f"Metrics for {_key} already exist, call reset_metrics() to reset the metrics or use allow_overwrite=True to overwrite the existing metrics"
                )
            self._fit_metrics[_key] = self._compute_metrics(predictions, labels)

        return Probe(estimator, predict_callback, fit_callback, self._data_preprocess_callback)

    def reset_estimators(self):
        self._estimators = {}

    def reset_metrics(self):
        self._fit_metrics = {}
        self._predict_metrics = {}


class BilinearProbeManager(ProbeManager):
    """Manager for bilinear probes; one probe per (h1, h2) pair."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        estimator_class: Any,
        estimator_kwargs: dict,
        compute_metrics: Callable[[Any, Any], Dict[str, Any]],
        allow_overwrite: bool = False,
        data_preprocess_callback: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(
            estimator_class=estimator_class,
            estimator_kwargs=estimator_kwargs,
            compute_metrics=compute_metrics,
            allow_overwrite=allow_overwrite,
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
        if pair_key in self._estimators and not self._allow_overwrite:
            raise ValueError(
                f"Probe {pair_key} already exists, call reset_estimators() to reset or use allow_overwrite=True"
            )
        estimator = self._estimator_class(**self._estimator_kwargs)
        self._estimators[pair_key] = estimator

        def predict_callback(predictions: Any, labels: Any):
            if pair_key in self._predict_metrics and not self._allow_overwrite:
                raise ValueError(
                    f"Metrics for {pair_key} already exist, call reset_metrics() or use allow_overwrite=True"
                )
            self._predict_metrics[pair_key] = self._compute_metrics(predictions, labels)

        def fit_callback(predictions: Any, labels: Any):
            if pair_key in self._fit_metrics and not self._allow_overwrite:
                raise ValueError(
                    f"Metrics for {pair_key} already exist, call reset_metrics() or use allow_overwrite=True"
                )
            self._fit_metrics[pair_key] = self._compute_metrics(predictions, labels)

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

    def reset_metrics(self):
        super().reset_metrics()
