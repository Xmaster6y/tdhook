"""
Probing
"""

from typing import Callable, Optional, List, Protocol, Any, Dict

from tensordict import TensorDict

from tdhook.contexts import HookingContextFactory
from tdhook.hooks import MultiHookManager, HookFactory, HookDirection, MultiHookHandle, DIRECTION_TO_RETURN
from tdhook.module import HookedModule


class Probe(Protocol):
    def step(self, data: Any, **kwargs) -> Any: ...


class Probing(HookingContextFactory):
    def __init__(
        self,
        key_pattern: str,
        probe_factory: Callable[[str, str], Probe],
        relative: bool = True,
        directions: Optional[List[HookDirection]] = None,
        additional_keys: Optional[List[str]] = None,
        submodules_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self._key_pattern = key_pattern
        self._hook_manager = MultiHookManager(key_pattern, relative=relative)
        self._probe_factory = probe_factory
        self._directions = directions or ["fwd"]
        self._additional_keys = additional_keys
        self._submodules_paths = submodules_paths

    @property
    def key_pattern(self) -> str:
        return self._key_pattern

    @key_pattern.setter
    def key_pattern(self, key_pattern: str):
        self._key_pattern = key_pattern
        self._hook_manager.pattern = key_pattern

    def _hook_module(self, module: HookedModule) -> MultiHookHandle:
        handles = []
        if self._additional_keys is not None:
            tmp_cache = TensorDict()
            handle, additional_items = module.get(
                cache=tmp_cache,
                module_key="td_module",
                cache_key="_additional_keys",
                callback=lambda **kwargs: kwargs["args"][0].select(*self._additional_keys),
                direction="fwd_pre",
                relative=False,
            )
            handles.append(handle)
        else:
            additional_items = None

        def hook_factory(name: str, direction: HookDirection) -> Callable:
            nonlocal self, additional_items
            probe = self._probe_factory(name, direction)

            def callback(**kwargs):
                nonlocal additional_items
                if additional_items is not None:
                    resolved_additional_items = additional_items.resolve()
                else:
                    resolved_additional_items = {}
                return probe.step(kwargs[DIRECTION_TO_RETURN[direction]], **resolved_additional_items)

            return HookFactory.make_reading_hook(callback=callback, direction=direction)

        if self._submodules_paths is not None:
            submodules_n_prefixes = []
            for submodule_path in self._submodules_paths:
                submodule = module._resolve_submodule_path(submodule_path, relative=False)
                prefix = f"{submodule_path}."
                submodules_n_prefixes.append((submodule, prefix))
        else:
            submodules_n_prefixes = [(module, "")]

        for submodule, prefix in submodules_n_prefixes:
            for direction in self._directions:
                handles.append(
                    self._hook_manager.register_hook(
                        submodule, lambda name: hook_factory(f"{prefix}{name}", direction), direction=direction
                    )
                )

        return MultiHookHandle(handles)


class SklearnProbe:
    def __init__(
        self,
        probe: Any,
        predict_callback: Callable[[Any, Any], Any],
        data_preprocess_callback: Optional[Callable[[Any], Any]] = None,
    ):
        self._probe = probe
        self._predict_callback = predict_callback
        self._data_preprocess_callback = data_preprocess_callback or self._default_data_preprocess_callback

    def step(self, data: Any, labels: Any, step_type: str):
        data = self._data_preprocess_callback(data)
        if step_type == "fit":
            self._probe.fit(data, labels)
        elif step_type == "predict":
            self._predict_callback(self._probe.predict(data), labels)
        else:
            raise ValueError(f"Invalid step type: {step_type}")

    def _default_data_preprocess_callback(self, data: Any) -> Any:
        return data.detach().flatten(1)


class SklearnProbeManager:
    def __init__(
        self,
        probe_class: Any,
        probe_kwargs: dict,
        compute_metrics: Callable[[Any, Any], Dict[str, Any]],
        allow_overwrite: bool = False,
        data_preprocess_callback: Callable[[Any], Any] = None,
    ):
        self._probe_class = probe_class
        self._probe_kwargs = probe_kwargs
        self._compute_metrics = compute_metrics
        self._allow_overwrite = allow_overwrite
        self._data_preprocess_callback = data_preprocess_callback

        self._probes = {}
        self._metrics = {}

    @property
    def probes(self) -> dict[str, SklearnProbe]:
        return self._probes

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    def probe_factory(self, key: str, direction: HookDirection) -> SklearnProbe:
        _key = f"{key}_{direction}"
        if _key in self._probes and not self._allow_overwrite:
            raise ValueError(
                f"Probe {_key} already exists, call reset_probes() to reset the probes or use allow_overwrite=True to overwrite the existing probes"
            )
        probe = self._probe_class(**self._probe_kwargs)
        self._probes[_key] = probe

        def predict_callback(predictions: Any, labels: Any):
            nonlocal self
            if _key in self._metrics and not self._allow_overwrite:
                raise ValueError(
                    f"Metrics for {_key} already exist, call reset_metrics() to reset the metrics or use allow_overwrite=True to overwrite the existing metrics"
                )
            self._metrics[_key] = self._compute_metrics(predictions, labels)

        return SklearnProbe(probe, predict_callback, self._data_preprocess_callback)

    def reset_probes(self):
        self._probes = {}

    def reset_metrics(self):
        self._metrics = {}
