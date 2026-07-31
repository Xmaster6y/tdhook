Composition contract
====================

.. note::

   This is the Phase 0 architecture decision for `issue 57
   <https://github.com/Xmaster6y/tdhook/issues/57>`_.  It defines the
   composability target and records the implementation work required to reach
   it.  The pipeline runtime itself is implemented by later roadmap phases.

Decision
--------

TDHook uses three distinct composition terms:

**Composed model**
   A PyTorch or TensorDict model whose execution may have multiple inputs,
   outputs, or nested submodules.  Preparing one TDHook method around such a
   model is *model support*, not evidence that two interpretability methods can
   share a run.

**Same-run hook composition**
   Two or more compatible hook operations installed on one prepared model for
   one forward/backward execution.  Compatibility depends on hook direction,
   read/write effects, registration order, path resolution, context type, and
   cleanup.  Shared inheritance from :class:`HookingContextFactory
   <tdhook.contexts.HookingContextFactory>` is not evidence of compatibility.

**Multi-stage pipeline**
   An ordered sequence of model methods and/or TensorDict transforms that
   exchanges named artifacts across zero or more model executions.  A stage
   boundary is an artifact boundary: required keys must exist before the stage
   starts and produced keys become available only after it succeeds.

The terms above are canonical for source documentation, tutorials, and the ECML
demo.  A workflow may use more than one term: for example, a multi-stage
pipeline may run a same-run hook composition inside one stage on a composed
model.

Design goal
-----------

TDHook is designed to make every public method usable in a declared
multi-stage pipeline.  Within that pipeline, the planner should coalesce
compatible hooks into the fewest safe model executions.  A method that cannot
share one execution is not outside the composability goal: the planner places
it at an explicit stage boundary and carries its named TensorDict artifacts
forward.

The target contract is therefore:

* every public method is representable as a model stage, TensorDict transform,
  weight-mutation stage, or owned support component;
* every stage declares the complete capability record below;
* compatible reads and writes can share a run with deterministic ordering;
* incompatible hook paths, context classes, mutations, devices, or gradient
  requirements cause a planned stage split rather than implicit behavior;
* the planner minimizes model executions subject to those constraints; and
* preflight returns an executable stage plan or an actionable incompatibility,
  before the first model call.

This is a strong workflow-level composability promise, not a promise that every
pair of methods must occupy the same forward/backward execution.

Capability model
----------------

A future preflight validator must evaluate each stage against the following
fields.  The inventory below records them for the current public methods and
operators.

``execution``
   Whether the stage executes a model, transforms existing TensorDict
   artifacts, or only supports another stage.

``hooks``
   Every forward/backward and pre/post direction used by the stage.

``effects_and_ordering``
   Reads, writes, state changes, and the order they require.  On the same
   module and direction, hooks execute in registration order unless a hook is
   explicitly prepended.  A later reader observes the value returned by an
   earlier writer.

``model_passes``
   The number of model calls, including conditional or user-controlled calls.
   An expanded batch is recorded separately from repeated calls.

``required_keys`` and ``produced_keys``
   TensorDict artifact contracts.  Configurable key names are represented by
   their constructor parameter; context caches and manager state are not
   silently treated as TensorDict artifacts.

``model_mutation`` and ``specialisation``
   Temporary module/weight changes and any specialised context or
   HookedModule class.  Both affect same-run compatibility and cleanup.

``device_batch_gradient``
   Device, shape/batch, and autograd requirements that preflight must check or
   explicitly delegate to a callback.

Current evidence states
-----------------------

The capability inventory uses the following states to track implementation
progress toward the design goal.  They describe the current release, not the
roadmap ceiling.

``supported``
   The current public contract has enough information to validate the
   operation.  This is not a blanket claim about every callback or key choice.

``unsupported``
   The current implementation has a known incompatibility in that composition
   mode.  The planner target is to repair it or isolate the method at an
   explicit stage boundary; the reason is stated below or in the capability
   inventory.

``untested``
   The implementation may be mechanically eligible, but TDHook does not yet
   promise the combination.  Users must not infer support from inheritance or
   from an example that passes Python values manually.

``not-applicable``
   The composition mode does not describe that component.

Method and operator inventory
-----------------------------

The inventory scope is every user-facing symbol exported by ``__all__`` from
``tdhook.attribution``, ``tdhook.latent``,
``tdhook.latent.dimension_estimation``, and ``tdhook.weights``.  Supporting
probing objects are included so the public surface is complete, but are marked
``probing support`` rather than executable stages.  Implementation helpers that
are not exported by those modules are not stable public methods.

.. csv-table:: Public capability inventory
   :file: _static/composition-capabilities.csv
   :header-rows: 1
   :class: longtable

Same-run compatibility
----------------------

Same-run composition is an optimization inside the broader pipeline contract.
The matrix records today's implementation blockers and the intended resolution.
It groups methods by the property that determines whether they can currently
share one execution:

``simple hooks``
   :class:`Probing <tdhook.latent.probing.Probing>` and
   :class:`SteeringVectors <tdhook.latent.SteeringVectors>`.  They use the
   standard context/module classes and do not wrap the model.

``wrapped methods``
   Attribution methods, :class:`ActivationAddition
   <tdhook.latent.ActivationAddition>`, and :class:`ActivationPatching
   <tdhook.latent.ActivationPatching>`.  Each installs a child-specific relative
   path into a generated TensorDict wrapper.

``specialised methods``
   :class:`ActivationCaching <tdhook.latent.ActivationCaching>`,
   :class:`Adapters <tdhook.weights.Adapters>`,
   :class:`Pruning <tdhook.weights.Pruning>`, and
   :class:`TaskVectors <tdhook.weights.TaskVectors>`.  They require a
   specialised context and/or HookedModule class.

``TensorDict operators``
   Dimension and representation-similarity estimators.  They are pipeline
   transforms, not hook compositions.

.. list-table:: Initial same-run compatibility matrix
   :header-rows: 1
   :stub-columns: 1

   * - First / second group
     - Simple hooks
     - Wrapped methods
     - Specialised methods
     - TensorDict operators
   * - Simple hooks
     - Untested: registration order is deterministic, but cross-method
       conformance and cleanup tests are pending.  Target: promote compatible
       pairs after conformance coverage.
     - Current blocker: the composite factory does not preserve the wrapped
       child's relative path.  Target: path-safe composition or a planned stage
       split.
     - Current blocker: the generic composite cannot select or merge the
       required specialised context/module classes.  Target: isolate the
       specialised stage unless it declares a merge strategy.
     - Not applicable: place the operator at a pipeline boundary.
   * - Wrapped methods
     - Current blocker: the wrapped child's relative path is lost.  Target:
       path-safe composition or a planned stage split.
     - Current blocker: child wrappers and relative paths are not merged.
       Target: merge compatible wrappers and split the rest.
     - Current blocker: wrapper and specialised context/module requirements are
       not merged.  Target: an explicit merge strategy or stage isolation.
     - Not applicable: place the operator at a pipeline boundary.
   * - Specialised methods
     - Current blocker: the generic composite rejects the specialised
       context/module requirement.  Target: stage isolation by default.
     - Current blocker: specialised classes and wrappers are not merged.
       Target: an explicit merge strategy or stage isolation.
     - Current blocker: there is no rule for selecting or merging competing
       specialised classes.  Target: stage isolation unless both declare the
       same compatible owner.
     - Not applicable: place the operator at a pipeline boundary.
   * - TensorDict operators
     - Not applicable: place the operator at a pipeline boundary.
     - Not applicable: place the operator at a pipeline boundary.
     - Not applicable: place the operator at a pipeline boundary.
     - Not applicable to same-run hooks; supported as a multi-stage transform
       when required and produced keys match.

Low-level same-run operators
----------------------------

The low-level :class:`HookedModule <tdhook.modules.HookedModule>` API remains
the supported way to compose operations inside a single run:

.. list-table::
   :header-rows: 1

   * - Operator
     - Effect
     - Directions
     - Ordering contract
   * - ``get`` / ``save``
     - Read an activation into a cache.
     - ``fwd`` by default; input, gradient, and gradient-output aliases select
       ``fwd_pre``, ``bwd``, and ``bwd_pre``.
     - A reader observes the value returned by earlier writers in the same
       direction.
   * - ``set``
     - Replace an activation or gradient.
     - The same forward/backward directions as reads.
     - Writers run in registration order; ``prepend=True`` moves that writer
       before existing hooks for the same module/direction.
   * - ``stop``
     - Stop execution after the selected module using the partial cache.
     - Forward output only.
     - Must be registered after any same-module reads/writes whose results are
       required before stopping.

Aliases such as ``get_input``, ``set_grad``, and ``save_grad_output`` inherit
the capability of their base read/write operator with the direction shown
above.  Hook handles are removed when the run exits, including exceptional
exit.

Multi-stage planner contract
----------------------------

The pipeline runtime must make all of the following decisions before the first
model execution:

#. Every stage names one row in the capability inventory or declares an
   equivalent custom capability record.
#. Its required TensorDict keys are present initially or are produced by an
   earlier stage.  A context cache, Python local, manager field, or closure is
   not an artifact unless an explicit adapter stage names and owns it.
#. Produced keys have a single owner, unless a stage explicitly declares a
   write/update policy.
#. Device, batch/shape, and gradient requirements agree at each boundary.
#. The requested model-pass budget includes expanded batches, optional endpoint
   evaluations, and callback-controlled evaluation loops.
#. Temporary model mutation has a scoped owner and restoration occurs before a
   later incompatible stage starts.
#. A same-run incompatibility produces a stage split when the required artifact
   boundary is available.  It fails preflight only when no valid split or
   adapter exists.
#. Any remaining ``unsupported`` combination fails preflight with the reason
   in this document.  Any ``untested`` same-run combination remains split by
   default; it must never be promoted to ``supported`` from shared inheritance
   alone.

Consequences
------------

This decision makes composability the organizing architecture of TDHook.  The
pipeline is the public abstraction; same-run hook composition is its
execution-minimizing optimization, and TensorDict artifacts are its stable
boundaries.  Current wrapper, path, and specialised-context limitations become
concrete runtime and conformance work rather than permanent API exclusions.
Later roadmap issues promote matrix cells as implementation and evidence land,
while preserving a usable pipeline through explicit stage splits.
