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

Select repeated module calls
----------------------------

A module instance can run more than once during one root model pass. Use an
integer ``Target.occurrence`` to select one zero-based call, or an immutable
:class:`~tdhook.targets.OccurrenceSelector` to select several calls:

.. code-block:: python

   import torch
   from torch import nn
   from tdhook.session import HookSession
   from tdhook.targets import OccurrenceSelector, Target

   class SharedModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.shared_layer = nn.Identity()

       def forward(self, value):
           calls = (self.shared_layer(value + offset) for offset in (1, 2, 3))
           return torch.cat(tuple(calls), dim=-1)

   model = SharedModel()
   inputs = torch.zeros(1, 1)
   target = Target(
       "shared_layer",
       "activation",
       -1,
       (0,),
       occurrence=OccurrenceSelector((0, 2)),
   )

   with HookSession(model) as session:
       captured = session.capture(target)
       model(inputs)

   plan = session.program.occurrence_plans
   evidence = session.occurrence_evidence

Occurrence indices reset for every root model pass. Each evidence record is
tensor-free and contains the target path, selected indices, and every observed
call index. TDHook raises before the model result is returned if a selected
call is missing. Duplicate or reordered selections are rejected when the
selector is constructed, and every temporary selector hook is removed when
the session exits, including after failures.

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
