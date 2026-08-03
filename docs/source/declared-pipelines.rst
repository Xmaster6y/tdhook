Declared pipelines: an offline walkthrough
===========================================

This compact tutorial is the canonical starting point for TDHook's declared
workflow API.  It uses a deterministic local model and tensors only: no model
download, checkpoint, or dataset is required.  The accompanying
``declared-pipelines`` notebook contains the same executable examples.

Every pipeline has two phases.  :meth:`~tdhook.pipeline.Pipeline.plan` is a
side-effect-free preflight that exposes model-pass cost and stage boundaries;
:meth:`~tdhook.pipeline.Pipeline.run` then produces named artifacts and their
provenance.  A split is intentional: TDHook only co-executes stages with an
explicit compatible capability contract.

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

The declared concept workflow has exactly two model passes.  The first LRP
stage records relevance for labelled concept examples, the artifact-only
selection stage chooses a channel, and the second LRP stage consumes that
selection artifact.  No callback, prepared hook context, or activation cache
is transferred by the notebook.

.. code-block:: python

   from tensordict import TensorDict
   from tdhook.attribution import LRP
   from tdhook.concepts import ChannelConditionedLRPStage, ConceptSelectionStage
   from tdhook.pipeline import Pipeline
   from tdhook.stages import AttributionStage

   concept_pipeline = Pipeline([
       AttributionStage(
           "concept-relevances", LRP(input_modules=["features"], warn_on_missing_rule=False),
           attribution_key=("attributions", "concept_examples"),
           legacy_attribution_key=("attr", "features"),
       ),
       ConceptSelectionStage("select-concept"),
       ChannelConditionedLRPStage(
           "conditioned-relevance", LRP(warn_on_missing_rule=False), condition_module="features",
       ),
   ])
   artifacts = TensorDict(
       {"inputs": {"input": examples, "concept_labels": labels}}, batch_size=[len(examples)]
   )
   plan = concept_pipeline.plan(artifacts)
   assert plan.model_passes == 2
   print([(run.stages, run.model_passes) for run in plan.runs])
   result = concept_pipeline.run(concept_model, artifacts, model_id="offline-tiny", seed=0)

The selected concept is at ``("metrics", "concept_selection")`` and the
conditioned input relevance is at ``("attributions", "conditioned")``.  The
result's ``provenance`` records that the selection depends on the concept
relevances and labels, while the conditioned attribution depends on the input
and selection artifact.  This is the workflow exercised by
``test_concept_attribution_workflow_is_declared_inspectable_and_matches_frozen_fixture``
in the conformance matrix.

Conditioned intrinsic dimension
-------------------------------

Use :func:`tdhook.dimension.conditioned_dimension_pipeline` when a captured
activation should become a conditioned estimator input.  Its only model pass
is activation capture; sample selection, estimation, and summary are
artifact-only stages.  For image or board features, use
:func:`tdhook.dimension.channel_conditioned_samples` for ``(sample, channel,
...)`` activations and keep rendering or plotting downstream.

.. code-block:: python

   from tdhook.dimension import channel_conditioned_samples, conditioned_dimension_pipeline
   from tdhook.latent import ActivationCaching
   from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator

   dimension_pipeline = conditioned_dimension_pipeline(
       ActivationCaching("features"),
       "features",
       channel_conditioned_samples,
       TwoNnDimensionEstimator(),
   )
   artifacts = TensorDict({"inputs": {"input": examples}}, batch_size=[len(examples)])
   plan = dimension_pipeline.plan(artifacts)
   assert plan.model_passes == 1
   print([(run.stages, run.model_passes) for run in plan.runs])
   result = dimension_pipeline.run(dimension_model, artifacts, model_id="offline-tiny", seed=0)

The result publishes the real cache at ``("activations", "cache")``, shaped
samples at ``("activations", "samples")``, estimates at ``("metrics",
"dimension")``, and finite-value summary at ``("metrics", "dimension_summary")``.
These latter three can have a condition axis unrelated to the original batch,
so TDHook preserves them as shape-neutral named artifacts instead of copying
them to Python dictionaries.

Conformance and scope
---------------------

The :doc:`composition` page is the source of truth for supported combinations.
In particular, a pipeline does not promise universal same-run execution:
unknown or incompatible pairs split conservatively before hooks are installed.
The phrase ``25+ methods`` describes the public API inventory's exported
classes and documented method variants; it is not a claim that every pair of
those methods co-executes.  The capability and conformance matrices name the
tested combinations, expected plans, and pass budgets.
