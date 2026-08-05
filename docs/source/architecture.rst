Architecture
============

TDHook is a TensorDict-native interpretability library. TensorDict is its data
model and execution vocabulary; TDHook adds semantic model targets,
interpretability methods, deterministic hook lifecycles, temporary model-state
management, and capability-aware execution planning.

Interfaces
----------

TDHook has three interfaces backed by one runtime:

Standalone methods
   A configured method is prepared against a model and then consumes and
   produces TensorDict keys. The context owns binding and cleanup, but never
   permanently owns the supplied model.

Interactive sessions
   Direct capture and intervention remain available for exploratory work. The
   interface is imperative and flexible. ``HookSession`` owns the temporary
   lifecycle and records the ordered ``HookProgram`` it actually installs::

       from tdhook.session import HookSession
       from tdhook.targets import Target

       target = Target("features.0", "activation", -1, (3,))
       with HookSession(model) as session:
           captured = session.capture(target)
           session.replace(target, 0)
           output = model(inputs)

       activation = captured.value
       program = session.program

   ``Target`` only describes and validates the selection; it never installs a
   hook or mutates the model itself.

Declared workflows
   A workflow composes methods and ordinary TensorDict modules by their
   ``in_keys`` and ``out_keys``. Planning validates dependencies, bound hook
   programs, and safe co-execution before model execution::

       from tensordict.nn import TensorDictModule
       from tdhook.attribution import Saliency
       from tdhook.workflow import Workflow

       summarise = TensorDictModule(
           lambda attribution: attribution.mean(-1),
           in_keys=[("attr", "input")],
           out_keys=[("summary", "mean")],
       )
       workflow = Workflow(Saliency(), summarise)

       plan = workflow.plan(model, data)
       result = workflow(model, data)

   ``Workflow`` returns the final TensorDict directly. The immutable plan is
   requested separately; it is report metadata, not a second artifact
   container.

TensorDict ownership
--------------------

Scientific inputs, outputs, and intermediate artifacts remain in a
``TensorDictBase``. TDHook does not introduce a parallel artifact store or key
contract. Prepared methods and transforms expose the native ``in_keys`` and
``out_keys`` of ``TensorDictModuleBase``. In-place updates follow TensorDict's
normal module semantics rather than a second TDHook mutation model.

Model ownership
---------------

The caller owns the model. TDHook may temporarily install hooks or change model
state inside a prepared execution, and must restore both after success or
failure. A method, session, workflow, or execution report must not retain a
hidden ownership claim over the model.

Execution contracts
-------------------

TensorDict modules already declare the data they consume and produce. A
configured TDHook method declares only execution requirements that TensorDict
cannot express:

* its model-pass and gradient requirements;
* the hooks and temporary model changes installed when it is bound;
* whether its bound hook program can share a model execution.

Compatibility belongs to the bound hook programs because module targets,
ordering, callbacks, model signatures, and temporary state determine whether
one execution is safe. Unknown compatibility produces separate executions.
Internal execution nodes are planner machinery, not a second user-facing
workflow language.

Planning uses an inspection binding: it installs and removes the same hooks as
execution but makes no model call and does not consume execution state such as
clearing an activation cache. Before an inspected plan is executed, the method
is rebound and its signature, execution requirements, hook program, and direct
wrapper status must still match. A changed fact fails before the model call.

The initial same-run proof is intentionally narrow. Adjacent one-pass methods
may share one model call only when their prepared TensorDict signatures and
autograd modes match, both wrappers execute the caller's model directly, and
every operation in every bound program is a read-only ``capture``. A
transformed wrapper, intervention, missing program, or other unknown fact
produces separate executions, and the plan records the reason.

Prepared methods may publish products after that shared call through their
native TensorDict wrapper. For example, ``ActivationCaching`` adds its
configured ``cache_key`` to ``out_keys`` and publishes a stable snapshot of
the context cache. Two activation methods can therefore capture different
layers into different output keys during one model pass. Shared-execution
compatibility compares the bound model signature, while dependency validation
uses each prepared method's full ``in_keys`` and ``out_keys``.

Artifact-only analysis remains ordinary TensorDict composition. The
conditioned-dimension workflow is one ``ActivationCaching`` method followed by
``ActivationSamples``, ``DimensionEstimation``, and ``DimensionSummary``
operators. Condition axes that do not match the model batch remain owned by
TensorDict as shape-neutral ``NonTensorData`` values; TDHook does not introduce
an artifact adapter or result wrapper for them.

Concept-conditioned attribution follows the same ownership rule. The first
``LRP`` method publishes a TensorDict attribution subtree,
``ConceptSelection`` derives a named selection with no model pass, and
``ChannelConditionedLRP`` reads that selection from its declared ``in_keys``
during the second pass. The selected channel is execution data: it is not
stored in a stage, notebook callback, or method-level side cache.

Shared hook runtime
-------------------

``HookProgramBuilder`` is the single primitive that installs ordered hooks and
owns their reverse-order cleanup. Both ``HookSession`` and migrated prepared
methods use it. A successful binding exposes a model-free ``HookProgram`` for
inspection, while the bound cleanup object retains the live hook handles only
for the duration of execution. Callback state and model objects are never
stored in the reportable program.
Structural equality between programs is therefore not a coexecution claim;
compatibility remains unknown until a binding supplies an explicit proof.

Activation capture, patching, direct steering, and steering-vector extraction
all bind through this runtime. Their programs distinguish observation
(``capture``) from intervention (``replace``), including hook direction and
prepend ordering.

Gradient attribution uses the same operations for activation and backward-hook
state. LRP rule installation is also a bound program operation
(``apply_rule``), so temporary ``forward`` rewrites have the same failure-safe
ownership and reporting as hooks.

Probing follows the same boundary. ``Probing`` owns only transient ``probe``
hooks and TensorDict input capture. ``ProbeManager`` owns fitted estimators
across bindings until ``reset_estimators()`` is called, while fit and prediction
metrics live in its ``results`` TensorDict and can be reset independently with
``reset_results()``. Arbitrary probe factories remain supported; their state is
owned by the factory provider, not by the hook runtime.

Weight methods use the same lifecycle without conflating two kinds of
intervention. ``Adapters`` reports activation ``capture`` and ``replace``
hooks, while ``Pruning`` reports temporary ``prune_parameters`` state. The
bound runtime restores the caller-owned model on both successful and failed
context exits; pruning no longer requires a separate context implementation.
Task-vector evaluation applies TensorDict module state through the same runtime
and reports a scoped ``replace_parameters`` program.

Artifact-only analysis stays outside the hook runtime. Intrinsic-dimension and
representation-similarity estimators are ordinary ``TensorDictModuleBase``
operators with native string or nested-tuple ``in_keys`` and ``out_keys``.
Attribution metrics are model-evaluated operations instead: they consume an
already evaluated TensorDict, return a separate TensorDict of metric values,
and expose their exact additional prepared-module pass count. Metric evaluation
does not mutate the caller's original TensorDict.

Methods migrate onto this runtime incrementally. Until a method exposes a
``HookProgram``, compatibility with another method is unknown and the planner
must keep their executions separate.

Dependency direction
--------------------

Pure targets, model paths, and execution requirements sit at the bottom of
TDHook. The shared runtime depends on those descriptions and on PyTorch and
TensorDict. Methods depend on the runtime. Workflow planning depends on method
protocols rather than concrete method families. Reporting depends on immutable
plans and provenance records.

The runtime must not import concrete methods or workflow code. Methods must not
import the workflow planner. Targets describe selections; the runtime performs
capture and replacement.

Scope
-----

TDHook owns interpretability semantics, hook lifecycle, safe model-state
restoration, method execution requirements, and execution planning. TensorDict
owns tensor storage, nested keys, batching, devices, persistence, parameter
containers, and module composition. PyTorch owns modules, hooks, and autograd.

TDHook is not a generic DAG engine, distributed scheduler, experiment tracker,
artifact database, or replacement for TensorDict.
