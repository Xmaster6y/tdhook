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

Interactive operations across a workflow
----------------------------------------

Use :meth:`~tdhook.workflow.Workflow.session` when captures or interventions
must wrap every model execution in a declared workflow. The managed object
exposes the normal :class:`~tdhook.session.HookSession` operations and runs the
bound workflow without requiring a second, manually nested context:

.. code-block:: python

   from tdhook.latent import ActivationCaching
   from tdhook.targets import Target
   from tdhook.workflow import Workflow

   workflow = Workflow(
       ActivationCaching("0", cache_key=("activations", "first")),
       ActivationCaching("2", cache_key=("activations", "second")),
   )

   with workflow.session(model) as session:
       observed = session.capture(Target("0", "activation", -1, (0,)))
       execution = session.run(data)

   execution.plan       # the exact WorkflowPlan that ran
   execution.program    # the HookProgram applied around that plan
   observed.values      # one entry per matching model execution, in call order

Session operations wrap the complete run; they are not workflow steps and do
not change planning or co-execution decisions. If
:meth:`~tdhook.session.HookSession.stop` reaches its target, it aborts the
workflow run, so no complete workflow result is returned. Its
:class:`~tdhook.session.EarlyStopResult` still exposes the partial module
output, and the surrounding managed context restores workflow hooks, session
hooks, and temporary model state.

Where to go next
----------------

* Continue with the full :doc:`Integrated Gradients notebook
  <notebooks/methods/integrated-gradients>`.
* Browse :doc:`tutorials` for attribution, probing, representation analysis,
  steering, and complete workflows.
* Use the generated :doc:`api/index` for exact signatures and field
  definitions.
