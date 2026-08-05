Migrating from v0.1 to v0.2
===========================

v0.2 makes TensorDict the data model for every public execution path.  The
model and scientific values stay caller-owned; TDHook temporarily owns hooks,
method-specific model state, and the facts needed to plan safe execution.

Choose one interface
--------------------

The interface follows the use case:

.. list-table::
   :header-rows: 1

   * - Use case
     - Interface
     - Result
   * - Run one configured method
     - ``with method.prepare(model) as prepared: prepared(data)``
     - The method's native TensorDict outputs
   * - Inspect or intervene interactively
     - ``with HookSession(model) as session: ...``
     - Values requested from the live session
   * - Declare several dependent operations
     - ``Workflow(method, operator, ...)``
     - The final TensorDict

There is no separate artifact container, key schema, or user-constructed stage
model.  Native ``in_keys`` and ``out_keys`` describe data dependencies.
``Workflow.plan(model, data)`` reports executions and model-pass cost before
``Workflow(model, data)`` runs them.

The v0.1 ``HookedModule.run/get/set`` helpers and unmanaged
``prepare(return_context=False)`` path are removed. Use ``HookSession`` for
imperative capture or replacement and the managed ``prepare`` context for a
configured method. This leaves one hook-registration primitive and one cleanup
owner for every execution path.

Activation capture
------------------

``ActivationCaching`` still exposes ``context.cache`` for interactive and
standalone use.  Prepared TensorDict execution now also publishes a stable
snapshot at ``cache_key`` so a later workflow step can consume it naturally.

.. code-block:: python

   from tensordict import TensorDict
   from tdhook.latent import ActivationCaching

   capture = ActivationCaching("encoder", cache_key=("activations", "encoder"))
   data = TensorDict({"input": inputs}, batch_size=[len(inputs)])

   with capture.prepare(model) as prepared:
       result = prepared(data)

   activations = result["activations", "encoder"]

Use ``cache_key=None`` when execution should remain context-only.  The cache
key becomes part of the prepared module's native ``out_keys`` and therefore
must not collide with a model output.

String ``ActivationCaching`` selectors retain their v0.1 regular-expression
matching behavior. Pass a ``Target`` when one exact model-relative module path
and a unit or channel selection should be captured and reported in the bound
``HookProgram``::

   from tdhook.targets import Target

   target = Target("encoder.layers[-1]", "activation", -1, (3,))
   capture = ActivationCaching(target)

Targeted prepared caching is forward-only. Use ``HookSession`` for targeted
gradient capture or other caller-driven backward-hook lifecycles.

Probing state and results
-------------------------

Probe estimators remain deliberately stateful across bindings until
``reset_estimators()`` is called.  Fit and prediction metrics now live in one
TensorDict at ``manager.results``.  Use ``reset_results()`` to clear metrics
without discarding fitted estimators, and ``overwrite_results=True`` when a
later execution may replace an existing result.  The v0.1 ``fit_metrics``,
``predict_metrics``, ``reset_metrics()``, and ``allow_overwrite`` names are
removed.

Attribution metrics
-------------------

``SensitivityMetric`` and ``InfidelityMetric`` are post-evaluation operations.
They receive a prepared TensorDict module and an already evaluated TensorDict,
then return a separate result TensorDict without mutating caller data.  Their
``additional_model_passes(prepared)`` method exposes the exact extra execution
cost.

Nested keys and operators
-------------------------

Public keys use TensorDict's ``NestedKey`` semantics directly: a string or a
non-empty tuple of strings.  Dimension estimators and representation metrics
are ordinary ``TensorDictModuleBase`` operators with configurable native
``in_keys`` and ``out_keys``.  They can run alone or as zero-pass workflow
steps; no adapter is required.

Following development versions
------------------------------

The ``Pipeline``, ``Stage``, artifact registry, family stage wrappers,
``CompositeHookingContextFactory``, and ``HookGroup`` appeared only in
unreleased development revisions between v0.1.3 and v0.2.0.  They have no v0.2
compatibility shims.  Replace them with configured methods and ordinary
TensorDict modules inside ``Workflow``.

Migration checklist
-------------------

1. Keep all exchanged values in one TensorDict and use native nested keys.
2. Read each prepared method's ``in_keys`` and ``out_keys`` as its data
   contract.
3. Replace probing metric dictionaries with ``manager.results``.
4. Treat metric evaluation as a separate operation with an explicit pass
   budget.
5. Use ``Workflow.plan`` before multi-method execution when pass count or
   coexecution matters.
6. Replace direct ``HookedModule`` hook operations with ``HookSession``.
7. Do not retain prepared wrappers, hook handles, or temporary model state
   outside their context manager.

See :doc:`architecture` for ownership and lifecycle boundaries and
:doc:`composition` for the exact planning and coexecution rules.
