Getting Started
===============

Composable interpretability for PyTorch with ``TensorDict`` and ``torch``
hooks.

Installation
------------

.. code-block:: console

   pip install tdhook

Your first attribution
----------------------

TDHook methods wrap an ordinary PyTorch model for the lifetime of a context
manager. Inputs, baselines, model outputs, and interpretability results use
explicit ``TensorDict`` keys:

.. code-block:: python

   import torch
   from torch import nn
   from tensordict import TensorDict
   from tdhook.attribution import IntegratedGradients

   model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
   inputs = torch.tensor([[0.2, -0.1, 0.4, 0.7]])

   def select_score(outputs, _):
       score = outputs["output"][..., 0]
       return TensorDict(score=score, batch_size=outputs.batch_size)

   data = TensorDict(
       {
           "input": inputs,
           ("baseline", "input"): torch.zeros_like(inputs),
       },
       batch_size=[1],
   )

   with IntegratedGradients(init_attr_targets=select_score).prepare(model) as hooked_model:
       result = hooked_model(data)

   attributions = result["attr", "input"]

``prepare(model)`` installs the method's hooks on entry and removes them on
exit. The model reads ``"input"`` and writes ``"output"``; Integrated
Gradients reads ``("baseline", "input")`` and writes
``("attr", "input")``. Here, ``attributions.shape`` is ``(1, 4)``, matching
the input.

Compose methods in a workflow
-----------------------------

:class:`~tdhook.workflow.Workflow` is TDHook's composition interface. It
combines configured interpretability methods with ordinary TensorDict modules
and validates their named inputs and outputs before running the model. For
example, the attribution above can feed a native summary operation:

.. code-block:: python

   from tensordict.nn import TensorDictModule
   from tdhook.workflow import Workflow

   workflow = Workflow(
       IntegratedGradients(init_attr_targets=select_score),
       TensorDictModule(
           lambda attribution: attribution.abs().sum(-1),
           in_keys=[("attr", "input")],
           out_keys=["attribution_mass"],
       ),
   )
   result = workflow(model, data)

The workflow returns one TensorDict containing both ``("attr", "input")`` and
``"attribution_mass"``. Use :meth:`~tdhook.workflow.Workflow.plan` when you
also need to inspect model-pass and compatibility decisions before execution.

Hand workflow artifacts to a local process
------------------------------------------

Workflow inputs and results can use TensorDict's native process-handoff
storage. Preallocate every declared output before calling ``share_memory_()``
or ``consolidate()`` because those operations lock the artifact's structure:

.. code-block:: python

   handoff_workflow = Workflow(
       TensorDictModule(
           torch.neg,
           in_keys=["input"],
           out_keys=[("result", "negated")],
       )
   )
   artifact = TensorDict(
       {
           "input": torch.ones(1, 4),
           ("result", "negated"): torch.zeros(1, 4),
       },
       batch_size=[1],
       device="cpu",
   ).share_memory_()

   # Pass artifact directly to a multiprocessing worker, then run there.
   result = handoff_workflow(nn.Identity(), artifact)

The returned object is the original TensorDict. Its nested keys, batch size,
device metadata, and shared or consolidated storage are preserved, while
declared output tensors are detached and updated in place, making them visible
to processes that share the storage. Any other in-place mutation to an existing
leaf is likewise caller-visible. CPU input leaves must be detached, output keys
must already exist with matching shape, dtype, and device, and workflow steps
must retain in-place TensorDict semantics. Deferred-backward workflows cannot
use a handoff artifact because their graph cannot cross processes. Violations
raise :class:`~tdhook.workflow.WorkflowHandoffError` before TDHook can silently
replace or materialize artifact storage. Process creation and transport remain
caller-owned; pass the TensorDict directly through native multiprocessing APIs.

Where to go next
----------------

* Continue with the full :doc:`Integrated Gradients notebook
  <notebooks/methods/integrated-gradients>`.
* Learn imperative capture, intervention, cleanup, and early stopping in the
  :doc:`HookSession notebook <notebooks/tutorials/hook-session>`.
* Browse :doc:`tutorials` for attribution, probing, representation analysis,
  steering, and complete workflows.
* Use the generated :doc:`api/index` for exact signatures and field
  definitions.
