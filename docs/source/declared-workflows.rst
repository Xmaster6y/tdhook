Declared workflows: an offline walkthrough
===========================================

This compact tutorial is the canonical starting point for TDHook's declared
workflow API.  It uses a deterministic local model and tensors only: no model
download, checkpoint, or dataset is required.  The accompanying
``declared-workflows`` notebook contains the same executable examples.

Every workflow has two phases. :meth:`~tdhook.workflow.Workflow.plan` is a
side-effect-free preflight that exposes model-pass cost, native key exchange,
and execution boundaries. Calling the workflow then returns the final
TensorDict. A split is intentional: TDHook only co-executes bound methods with
an explicit safety proof.

Offline fixture
---------------

Run this setup once before either workflow.  The concept model exposes the
``linear1`` feature selected by LRP; the dimension model exposes a four-
dimensional ``features`` activation for channel-conditioned sampling.

.. code-block:: python

   import torch
   from torch import nn

   torch.manual_seed(0)

   class TinyConceptModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear1 = nn.Linear(10, 10)
           self.relu = nn.ReLU()
           self.linear2 = nn.Linear(10, 2)

       def forward(self, input):
           return self.linear2(self.relu(self.linear1(input)))

   class FeatureMap(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(10, 12)

       def forward(self, input):
           return self.linear(input).relu().reshape(-1, 3, 2, 2)

   class TinyFeatureModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.features = FeatureMap()
           self.head = nn.Linear(12, 2)

       def forward(self, input):
           return self.head(self.features(input).flatten(1))

   concept_model = TinyConceptModel().eval()
   dimension_model = TinyFeatureModel().eval()
   examples = torch.randn(6, 10)
   labels = torch.tensor([1, 1, 1, 0, 0, 0])

Minimal concept-conditioned attribution
---------------------------------------

The declared concept workflow has exactly two model passes. The first LRP
method records relevance for labelled concept examples, the ordinary
``ConceptSelection`` operator chooses a channel, and ``ChannelConditionedLRP``
reads that selection from the execution TensorDict. No callback, prepared hook
context, or activation cache is transferred between passes.

.. code-block:: python

   from tensordict import TensorDict
   from tdhook.attribution import LRP
   from tdhook.concepts import ChannelConditionedLRP, ConceptSelection
   from tdhook.workflow import Workflow

   concept_workflow = Workflow(
       LRP(
           input_modules=["linear1"],
           attribution_key=("attributions", "concept_examples"),
           warn_on_missing_rule=False,
       ),
       ConceptSelection(("attributions", "concept_examples", "linear1")),
       ChannelConditionedLRP(
           LRP(warn_on_missing_rule=False), condition_module="linear1",
       ),
   )
   artifacts = TensorDict(
       {"input": examples, "concept_labels": labels}, batch_size=[len(examples)]
   )
   plan = concept_workflow.plan(concept_model, artifacts)
   assert plan.model_passes == 2
   print([(execution.steps, execution.model_passes) for execution in plan.executions])
   result = concept_workflow(concept_model, artifacts)

The selected concept is at ``("metrics", "concept_selection")`` and the
conditioned input relevance is at
``("attributions", "conditioned", "input")``. The native ``in_keys`` and
``out_keys`` record that selection depends on concept relevances and labels,
while conditioned attribution depends on both the model input and selection.
This is the workflow exercised by
``test_concept_attribution_workflow_is_declared_inspectable_and_matches_frozen_fixture``
in the conformance matrix.

Conditioned intrinsic dimension
-------------------------------

Use :func:`tdhook.dimension.conditioned_dimension_workflow` when a captured
activation should become a conditioned estimator input.  Its only model pass
is activation capture; sample selection, estimation, and summary are
ordinary TensorDict operators. For image or board features, use
:func:`tdhook.dimension.channel_conditioned_samples` for ``(sample, channel,
...)`` activations and keep rendering or plotting downstream.

.. code-block:: python

   from tdhook.dimension import channel_conditioned_samples, conditioned_dimension_workflow
   from tdhook.latent import ActivationCaching
   from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator

   dimension_workflow = conditioned_dimension_workflow(
       ActivationCaching("features", cache_key=("activations", "cache")),
       "features",
       channel_conditioned_samples,
       TwoNnDimensionEstimator(),
   )
   artifacts = TensorDict({"input": examples}, batch_size=[len(examples)])
   plan = dimension_workflow.plan(dimension_model, artifacts)
   assert plan.model_passes == 1
   print([(execution.steps, execution.model_passes) for execution in plan.executions])
   result = dimension_workflow(dimension_model, artifacts)

The result publishes the real cache at ``("activations", "cache")``, shaped
samples at ``("activations", "samples")``, estimates at ``("metrics",
"dimension")``, and finite-value summary at ``("metrics", "dimension_summary")``.
These latter three can have a condition axis unrelated to the original batch,
so TensorDict stores them as shape-neutral ``NonTensorData`` rather than
copying them to Python dictionaries.

Conformance and scope
---------------------

The :doc:`composition` page is the source of truth for supported combinations.
In particular, a workflow does not promise universal same-run execution:
unknown or incompatible pairs split conservatively before hooks are installed.
The conformance matrix names the concrete tested combinations, expected plans,
and pass budgets; it does not claim that every pair of public methods
co-executes.
