"""Executable, typed pipeline stages for the public method families.

The underlying public methods keep their existing TensorDict layouts.  These
stages are the composition boundary: callers use stable artifact paths while
the adapter owns the temporary legacy storage used for one model pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn
from tensordict import TensorDict, TensorDictBase

from tdhook.artifacts import ArtifactAdapter, ArtifactContract
from tdhook.contexts import HookingContextFactory
from tdhook.pipeline import PipelineKey, Stage


@dataclass(frozen=True)
class StageCapability:
    """Code-backed public capability declaration for a built-in stage."""

    method: str
    required_keys: tuple[PipelineKey, ...]
    provided_keys: tuple[PipelineKey, ...]
    effects: frozenset[str]
    model_passes: int


def _execute(factory: HookingContextFactory, model: nn.Module, storage: TensorDictBase) -> TensorDictBase:
    with factory.prepare(model) as method:
        result = method(storage)
    return storage if result is None else result


class _FactoryStage(Stage):
    """Base class for stages backed by an existing public method factory."""

    def __init__(self, name: str, factory: HookingContextFactory, adapter: ArtifactAdapter, *, effects: Iterable[str]):
        super().__init__(
            name, artifact_contract=adapter.contract, effects=("model_execution", *effects), method_id=adapter.method
        )
        self.factory = factory
        self.adapter = adapter

    def _storage(self, artifacts: TensorDictBase) -> TensorDictBase:
        return self.adapter.prepare(artifacts, TensorDict())


class ActivationCachingStage(_FactoryStage):
    """Execute :class:`ActivationCaching` and publish its actual cache."""

    capability = StageCapability(
        "ActivationCaching",
        (("inputs", "input"),),
        (("activations", "cache"),),
        frozenset({"model_execution", "activation_read"}),
        1,
    )

    def __init__(
        self,
        name: str,
        factory: HookingContextFactory,
        *,
        input_key: PipelineKey = ("inputs", "input"),
        cache_key: PipelineKey = ("activations", "cache"),
    ):
        adapter = ArtifactAdapter(
            "ActivationCaching",
            ArtifactContract(requires={"input": input_key}, provides={"cache": cache_key}),
            {"input": "input", "cache": "cache"},
        )
        super().__init__(name, factory, adapter, effects=("activation_read",))

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        storage = self._storage(artifacts)
        # ActivationCaching and Adapters expose their context cache through
        # this documented factory setting. Restore it so standalone reuse is
        # unaffected after the pipeline pass.
        kwargs = self.factory._hooking_context_kwargs
        previous = kwargs.get("cache")
        kwargs["cache"] = storage.get("cache") if "cache" in storage else TensorDict()
        try:
            _execute(self.factory, model, storage)
            storage.set("cache", kwargs["cache"])
        finally:
            kwargs["cache"] = previous
        return self.adapter.finalize(artifacts, storage)


class AttributionStage(_FactoryStage):
    """Execute an attribution factory and publish its computed attribution."""

    capability = StageCapability(
        "Attribution",
        (("inputs", "input"),),
        (("attributions", "values"),),
        frozenset({"model_execution", "gradient"}),
        1,
    )

    def __init__(
        self,
        name: str,
        factory: HookingContextFactory,
        *,
        input_key: PipelineKey = ("inputs", "input"),
        attribution_key: PipelineKey = ("attributions", "values"),
        legacy_attribution_key: PipelineKey = "attr",
    ):
        adapter = ArtifactAdapter(
            "Attribution",
            ArtifactContract(requires={"input": input_key}, provides={"attributions": attribution_key}),
            {"input": "input", "attributions": legacy_attribution_key},
        )
        super().__init__(name, factory, adapter, effects=("gradient",))

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        storage = self._storage(artifacts)
        return self.adapter.finalize(artifacts, _execute(self.factory, model, storage))


class ProbingStage(_FactoryStage):
    """Execute :class:`Probing` and publish the manager that holds its results."""

    capability = StageCapability(
        "Probing", (("inputs", "input"),), (("probes", "results"),), frozenset({"model_execution", "probe_update"}), 1
    )

    def __init__(
        self,
        name: str,
        factory: HookingContextFactory,
        results: object,
        *,
        input_key: PipelineKey = ("inputs", "input"),
        result_key: PipelineKey = ("probes", "results"),
    ):
        adapter = ArtifactAdapter(
            "Probing",
            ArtifactContract(requires={"input": input_key}, provides={"results": result_key}),
            {"input": "input", "results": "results"},
        )
        super().__init__(name, factory, adapter, effects=("probe_update",))
        self.results = results

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        storage = self._storage(artifacts)
        _execute(self.factory, model, storage)
        storage.set("results", self.results)
        return self.adapter.finalize(artifacts, storage)


class WeightInterventionStage(_FactoryStage):
    """Run weight/adapter intervention factories and publish their model output.

    This deliberately does not label a context activation cache as weights:
    the output is the real product of an intervention pass.
    """

    capability = StageCapability(
        "WeightIntervention",
        (("inputs", "input"),),
        (("outputs", "model"),),
        frozenset({"model_execution", "weight_intervention"}),
        1,
    )

    def __init__(
        self,
        name: str,
        factory: HookingContextFactory,
        *,
        input_key: PipelineKey = ("inputs", "input"),
        output_key: PipelineKey = ("outputs", "model"),
        legacy_output_key: PipelineKey = "output",
    ):
        adapter = ArtifactAdapter(
            "WeightIntervention",
            ArtifactContract(requires={"input": input_key}, provides={"output": output_key}),
            {"input": "input", "output": legacy_output_key},
        )
        super().__init__(name, factory, adapter, effects=("weight_intervention",))

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        storage = self._storage(artifacts)
        return self.adapter.finalize(artifacts, _execute(self.factory, model, storage))


BUILTIN_STAGE_CAPABILITIES = {
    capability.method: capability
    for capability in (
        ActivationCachingStage.capability,
        AttributionStage.capability,
        ProbingStage.capability,
        WeightInterventionStage.capability,
    )
}

# Public symbols represented by corresponding composition-matrix rows.  The
# mapping is deliberately code-owned so documentation tests catch a row that
# loses its executable implementation contract.
DOCUMENTED_STAGE_CAPABILITIES = {
    "ActivationCaching": "ActivationCaching",
    "Probing": "Probing",
    "Adapters": "WeightIntervention",
}


def capability_for_stage(method: str) -> StageCapability:
    """Return the executable contract for a supported built-in stage."""
    return BUILTIN_STAGE_CAPABILITIES[method]


def activation_caching_stage(name: str, factory: HookingContextFactory, **kwargs: object) -> ActivationCachingStage:
    """Build an executable activation-caching stage."""
    return ActivationCachingStage(name, factory, **kwargs)


def attribution_stage(name: str, factory: HookingContextFactory, **kwargs: object) -> AttributionStage:
    """Build an executable attribution stage."""
    return AttributionStage(name, factory, **kwargs)


def probing_stage(name: str, factory: HookingContextFactory, results: object, **kwargs: object) -> ProbingStage:
    """Build an executable probing stage."""
    return ProbingStage(name, factory, results, **kwargs)


def weight_intervention_stage(name: str, factory: HookingContextFactory, **kwargs: object) -> WeightInterventionStage:
    """Build an executable weight-intervention stage."""
    return WeightInterventionStage(name, factory, **kwargs)
