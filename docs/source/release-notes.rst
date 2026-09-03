Release Notes
=============

v0.2.1
------

TDHook 0.2.1 consolidates the managed hook-session and workflow architecture,
adds explicit selection and evidence for repeatedly called targets, and expands
the executable method and research notebooks.

Breaking cleanup
~~~~~~~~~~~~~~~~

* ``Target.occurrences`` is the single occurrence-selection API. It accepts an
  ordered tuple of zero-based call indices, or ``None`` for every call. The
  transitional ``OccurrenceSelector`` and integer forms have been removed.
* ``CapturedTarget.values`` is the single capture result. It preserves every
  observation in call order; use ``values[-1]`` when one latest observation is
  expected. The compatibility ``value`` alias has been removed.
* The internal ``CacheProxy`` routing layer and the empty ``tdhook.auto`` and
  ``tdhook.sources`` modules have been removed.
* Retired ``HookedModule.run``, ``save/get/set``, ``return_context``,
  ``disable_context_hooks``, and ``restore`` examples have been removed from
  the bundled API guidance. Use ``HookSession`` for imperative capture and
  intervention, and ``method.prepare(model)`` as a context manager.

Highlights
~~~~~~~~~~

* Workflows expose stable occurrence plans and fail-closed execution evidence.
* Hook sessions support live replacements, gradients, structured inputs and
  outputs, early stopping, repeated executions, and rank-local DDP semantics.
* Workflow artifacts support shared-memory and distributed-rank handoff.
* Activation caches support caller-owned memory-mapped storage.
* CircuitLens, circuit clustering, WeightLens, intervention optimization, and
  maintained Othello, ROME, chess, probing, and workflow notebooks are included.
* Bilinear probe fitting now owns its gradient context, so it works when model
  inference intentionally runs under ``torch.no_grad()``.

Validation scope
~~~~~~~~~~~~~~~~

The maintained unit and integration suite, warning-strict Sphinx build,
fresh-kernel deterministic notebooks, built wheel, and installed-wheel smoke
test form the release gate. GPU and scientific-reproduction notebooks have
their own declared resource and provenance requirements; rendering or static
contract checks alone do not establish their scientific results.
