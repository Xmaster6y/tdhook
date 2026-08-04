Architecture
============

.. note::

   This page defines the target architecture for TDHook 0.2. The refactor is
   developed in `pull request 96 <https://github.com/Xmaster6y/tdhook/pull/96>`_.

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
   ``in_keys`` and ``out_keys``. Planning validates dependencies, effects, and
   safe co-execution before model execution.

TensorDict ownership
--------------------

Scientific inputs, outputs, and intermediate artifacts remain in a
``TensorDictBase``. TDHook does not introduce a parallel artifact store.
Methods declare the keys they read and write. Replacing an existing key is
allowed only when the method declares that overwrite explicitly.

Standalone and interactive execution may mutate the working TensorDict when
requested. Declared workflows default to controlled key publication. A
structural copy does not clone tensor leaves, so full rollback of arbitrary
in-place tensor mutation is not implied.

Model ownership
---------------

The caller owns the model. TDHook may temporarily install hooks or change model
state inside a prepared execution, and must restore both after success or
failure. A method, session, workflow, or execution report must not retain a
hidden ownership claim over the model.

Execution contracts
-------------------

Each method declares:

* TensorDict input, output, and overwrite keys;
* whether it executes a model or only transforms a TensorDict;
* its model-pass and gradient requirements;
* activation, gradient, parameter, and method-state effects;
* runtime constraints and explicit co-execution compatibility.

The planner uses these declarations conservatively. Unknown compatibility,
conflicting writes, different gradient modes, or different runtime constraints
produce separate executions. Internal execution nodes are planner machinery,
not a second user-facing workflow language.

Dependency direction
--------------------

Pure targets and contracts sit at the bottom of TDHook. The shared runtime
depends on those descriptions and on PyTorch and TensorDict. Methods depend on
the runtime. Workflow planning depends on method protocols rather than concrete
method families. Reporting depends on immutable plans and provenance records.

The runtime must not import concrete methods or workflow code. Methods must not
import the workflow planner. Targets describe selections; the runtime performs
capture and replacement.

Scope
-----

TDHook owns interpretability semantics, hook lifecycle, safe model-state
restoration, method contracts, and execution planning. TensorDict owns tensor
storage, nested keys, batching, devices, persistence, parameter containers, and
module composition. PyTorch owns modules, hooks, and autograd.

TDHook is not a generic DAG engine, distributed scheduler, experiment tracker,
artifact database, or replacement for TensorDict.
