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
   interface is imperative and flexible, while its implementation records the
   same targets and hook specifications used by declared methods.

Declared workflows
   A workflow composes methods and ordinary TensorDict modules by their
   ``in_keys`` and ``out_keys``. Planning validates dependencies, bound hook
   programs, and safe co-execution before model execution.

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

Dependency direction
--------------------

Pure targets and execution requirements sit at the bottom of TDHook. The shared runtime
depends on those descriptions and on PyTorch and TensorDict. Methods depend on
the runtime. Workflow planning depends on method protocols rather than concrete
method families. Reporting depends on immutable plans and provenance records.

The runtime must not import concrete methods or workflow code. Methods must not
import the workflow planner. Targets describe selections; the runtime performs
capture and replacement.

Scope
-----

TDHook owns interpretability semantics, hook lifecycle, safe model-state
restoration, method execution requirements, and execution planning. TensorDict owns tensor
storage, nested keys, batching, devices, persistence, parameter containers, and
module composition. PyTorch owns modules, hooks, and autograd.

TDHook is not a generic DAG engine, distributed scheduler, experiment tracker,
artifact database, or replacement for TensorDict.
