Composition
===========

``Workflow`` is TDHook's only declared-composition interface. It combines
configured interpretability methods with ordinary ``TensorDictModuleBase``
operators and returns the final TensorDict directly.

Natural interfaces
------------------

A configured TDHook method remains useful on its own::

   with method.prepare(model) as prepared:
       result = prepared(data)

Use ``Workflow`` only when several methods or TensorDict operators form one
declared computation::

   from tensordict.nn import TensorDictModule
   from tdhook.latent import ActivationCaching
   from tdhook.workflow import Workflow

   summarise = TensorDictModule(
       lambda activation: activation.mean(-1),
       in_keys=[("activations", "cache", "features")],
       out_keys=[("summary", "features")],
   )
   workflow = Workflow(
       ActivationCaching("features", cache_key=("activations", "cache")),
       summarise,
   )

   plan = workflow.plan(model, data)
   result = workflow(model, data)

For exploratory capture or intervention, use :class:`tdhook.session.HookSession`
instead. A session is intentionally imperative; a workflow is intentionally
declared and inspectable.

Data ownership
--------------

TensorDict owns scientific inputs, outputs, and intermediate values. Every
workflow step exposes native ``in_keys`` and ``out_keys``. TDHook does not add
an artifact registry, adapter schema, stage contract, or parallel result
container.

Two steps cannot own the same output key, or ancestor and descendant output
keys, by accident. Planning rejects that overlap before execution. Wrap a step
in ``WorkflowUpdate(step)`` only when replacing the earlier value is an
intentional part of the declared computation.

Nested keys, batch dimensions, devices, TensorDict parameter containers, and
persistence retain their TensorDict meaning. A zero-pass analysis or reshape
is an ordinary TensorDict operator. Values whose condition axes do not match
the model batch can remain shape-neutral ``NonTensorData`` inside the same
TensorDict.

Planning
--------

``Workflow.plan(model, data)`` binds each method for inspection, validates its
native dependencies, and returns immutable execution metadata without calling
the model. A plan reports:

* the ordered steps in each execution;
* whether a step is a model method or zero-pass operator;
* native input and output keys;
* model-pass and gradient requirements;
* every accepted or rejected co-execution decision.

Execution rebinds each method and verifies that its model signature,
requirements, hook program, and wrapper behavior still match the inspected
facts. Dependency keys are checked again immediately before each consumer
runs, so a producer cannot satisfy a nested dependency merely by declaring a
parent namespace it did not actually populate.

Safe co-execution
-----------------

Unknown compatibility means separate executions. The initial proof permits
adjacent one-pass methods to share a model call only when:

* their prepared model signatures and gradient modes match;
* their wrappers execute the caller's TensorDict model directly;
* every bound operation is a read-only activation ``capture``.

Interventions, backward replacements, transformed wrappers, temporary model
state, missing hook programs, and other unknown behavior split conservatively.
This is a safety decision, not a claim that the methods can never be optimized
together in a future implementation.

Concrete workflows
------------------

Concept-conditioned attribution has two model passes:

#. ``LRP`` publishes per-input and per-feature relevance below a configured
   attribution root.
#. ``ConceptSelection`` derives ``("metrics", "concept_selection")`` with no
   model call.
#. ``ChannelConditionedLRP`` reads the selected channel and direction through
   declared nested keys during the second pass.

Conditioned intrinsic dimension has one model pass:

#. ``ActivationCaching`` publishes its native cache.
#. ``ActivationSamples`` reshapes one cached activation.
#. ``DimensionEstimation`` runs a native TensorDict estimator.
#. ``DimensionSummary`` publishes finite count, mean, and standard deviation.

The offline :doc:`notebooks/tutorials/declared-workflows` notebook runs both
examples.

Conformance evidence
--------------------

The following table is generated from the same expected plans asserted by the
tests. ``supported`` means that the exact built-in combination, lifecycle, key
exchange, and model-pass budget are exercised; it is not a universal claim
about every possible method pair.

.. csv-table:: Workflow conformance
   :file: _static/composition-conformance.csv
   :header-rows: 1
   :widths: 26, 10, 30, 54, 10

Scope
-----

``Workflow`` is a deterministic ordered composition, not a generic DAG,
distributed scheduler, experiment tracker, or artifact database. TensorDict
already supplies the data and module composition model; TDHook adds only the
interpretability lifecycle and execution planning facts TensorDict cannot
express.
