Artifact contracts and provenance
=================================

Pipeline artifacts have a stable public schema. Public keys start with one of
``inputs``, ``outputs``, ``activations``, ``gradients``, ``attributions``,
``probes``, ``interventions``, or ``metrics``. Temporary implementation values
are private and must be placed below ``("_private", method, ...)``. Private
keys are owned by the creating context or manager and must be cleared when that
owner exits; they are never a supported downstream dependency.

Stages can declare an :class:`tdhook.artifacts.ArtifactContract`, which gives
each requirement and product a method-facing name and a public TensorDict key.
An :class:`tdhook.artifacts.ArtifactAdapter` maps this contract onto existing
method storage without changing that method's standalone return behaviour.
Use :class:`tdhook.pipeline.AdapterStage` to copy declared public requirements
into that storage before execution and publish declared products afterwards.
Its ``execute`` callback receives ``(model, artifacts, storage)``; pass the
provided storage as the legacy method's cache or result TensorDict. The built-in
adapter helpers cover activation caching, probing, attribution, and weight
adapters.

``Pipeline.run`` returns provenance alongside artifacts and stage metadata.
Pass the caller's stable ``model_id``, an optional ``seed``, and per-stage
configuration to record model identity, device/dtype, package version, parent
artifact keys, and method configuration. This is in-memory metadata only:
serialization remains the responsibility of the caller.

``ArtifactRegistry`` can be used by a host that keeps artifacts across runs.
It assigns ownership by generation and rejects a key from an earlier generation
when it is required as fresh, preventing stale cache reuse.
