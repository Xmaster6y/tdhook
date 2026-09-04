Tutorials
=========

These notebooks are the maintained guides to TDHook. Start with one method or
follow a complete workflow that combines methods with a model or domain.

Core interfaces
---------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Interactive Hook Sessions
      :link: notebooks/tutorials/hook-session
      :link-type: doc
      :class-card: surface

      :octicon:`terminal;2em;sd-text-primary`

      Capture and intervene imperatively with an explicit managed lifecycle.

   .. grid-item-card:: Declared Workflows
      :link: notebooks/tutorials/declared-workflows
      :link-type: doc
      :class-card: surface

      :octicon:`workflow;2em;sd-text-primary`

      Compose methods and TensorDict operations with an inspectable plan, and
      keep activation caches on caller-owned disk storage.

Advanced execution
------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Process Handoff and DDP
      :link: notebooks/tutorials/process-and-distributed-workflows
      :link-type: doc
      :class-card: surface

      :octicon:`server;2em;sd-text-primary`

      Preserve local shared artifacts and keep hook execution rank-local under
      DistributedDataParallel.

Learn the methods
-----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Integrated Gradients
      :link: notebooks/methods/integrated-gradients
      :link-type: doc
      :class-card: surface

      :octicon:`graph;2em;sd-text-primary`

      Attribute predictions to inputs with an accumulated gradient path.

   .. grid-item-card:: Steering Vectors
      :link: notebooks/methods/steering-vectors
      :link-type: doc
      :class-card: surface

      :octicon:`iterations;2em;sd-text-primary`

      Modify model behavior through intermediate activation directions.

   .. grid-item-card:: Linear Probing
      :link: notebooks/methods/linear-probing
      :link-type: doc
      :class-card: surface

      :octicon:`search;2em;sd-text-primary`

      Test which concepts are linearly available in learned representations.

   .. grid-item-card:: Bilinear Probing
      :link: notebooks/methods/bilinear-probing
      :link-type: doc
      :class-card: surface

      :octicon:`git-compare;2em;sd-text-primary`

      Capture interactions between paired layer representations.

   .. grid-item-card:: Dimension Estimation
      :link: notebooks/methods/dimension-estimation
      :link-type: doc
      :class-card: surface

      :octicon:`number;2em;sd-text-primary`

      Estimate intrinsic dimension with TwoNN, local PCA, and related methods.

   .. grid-item-card:: Representation Similarity
      :link: notebooks/methods/representation-similarity
      :link-type: doc
      :class-card: surface

      :octicon:`diff;2em;sd-text-primary`

      Compare learned representations with CKA and information imbalance.

Complete workflows
------------------

Composition in the demonstration notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The concept-attribution and chess-dimension notebooks correspond to the two
use cases in the accepted ECML demonstration paper. Start with these when
presenting composition: they combine existing methods through named artifacts,
not new attribution or dimension estimators.

Each workflow declares the keys that its steps consume and produce. A step
receives the result of its producer through the TensorDict; plotting and
experiment sweeps remain ordinary notebook code. Functions and workflow
construction are shown separately so that the dependencies remain readable.

.. list-table:: Artifact dependencies
   :header-rows: 1
   :widths: 22 48 30

   * - Notebook
     - Composed analysis
     - Artifact passed between stages
   * - Concept attribution
     - Batched LRP → ConceptSelection; calibrated selection →
       ChannelConditionedLRP → relevance maps
     - ``metrics/channel_relevance`` and ``metrics/concept_selection``
   * - Chess dimension estimation
     - ActivationCaching → ActivationSamples → TwoNN → DimensionSummary,
       for channel- and square-conditioned samples
     - ``activations/cache``, ``samples``, ``dimensions``, ``summaries``
   * - Othello
     - Capture → probe; capture → construct replacement → intervention →
       metric; optimized intervention → decoded board
     - Activation caches, replacement tensors, predictions and scores
   * - WeightLens and CircuitLens
     - Weight projection → candidate selection; activation capture → circuit
       analysis; produced circuit batch → signature clustering
     - Projections, captured activations and ``artifacts/circuit``
   * - ROME
     - Causal-tracing intervention → answer probability; rank-one edit
       construction → temporary edit evaluation
     - ``edit/deltas`` and ``metrics/edited_case``

Calibration and query execution are separate phases in concept attribution,
so the same selection can be reused on new images. WeightLens and CircuitLens
are complementary analyses, not a claimed producer-consumer dependency between
weight-only candidates and attention circuits. ROME keeps its official edit
solver and evaluator; native TensorDict operators connect them inside Workflow.
Othello's context-based intervention and optimization APIs are similarly exposed
as operators with explicit inputs and outputs.

The additional reproduction notebooks demonstrate integration with existing
research pipelines. Their controls and bounded evaluation populations are
specified in the notebooks. Successful execution is integration evidence;
scientific conclusions require the reported metrics and comparison criteria.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: TorchRL PPO Action Probing
      :link: notebooks/tutorials/torchrl-ppo
      :link-type: doc
      :class-card: surface

      :octicon:`dependabot;2em;sd-text-primary`

      Probe action representations in a TorchRL PPO agent.

   .. grid-item-card:: Chess Value Saliency
      :link: notebooks/tutorials/chess-value-saliency
      :link-type: doc
      :class-card: surface

      :octicon:`eye;2em;sd-text-primary`

      Attribute chess value predictions with saliency methods.

   .. grid-item-card:: Concept Attribution
      :link: notebooks/tutorials/concept-attribution
      :link-type: doc
      :class-card: surface

      :octicon:`project-roadmap;2em;sd-text-primary`

      Visualize concept attribution on natural images.

   .. grid-item-card:: Chess Dimension Estimation
      :link: notebooks/tutorials/chess-dimension-estimation
      :link-type: doc
      :class-card: surface

      :octicon:`telescope;2em;sd-text-primary`

      Estimate and plot dimensions for chess activations.

   .. grid-item-card:: Othello Research Reproduction
      :link: notebooks/tutorials/othello-research-reproduction
      :link-type: doc
      :class-card: surface

      :octicon:`beaker;2em;sd-text-primary`

      Reproduce published behavior and linear-probe results, then validate
      intervention parity.

   .. grid-item-card:: WeightLens and CircuitLens Reproduction
      :link: notebooks/tutorials/weight-circuit-research-reproduction
      :link-type: doc
      :class-card: surface

      :octicon:`git-compare;2em;sd-text-primary`

      Reproduce bounded Gemma-2-2B WeightLens candidates and CircuitLens
      contributors, then cluster circuit signatures.

   .. grid-item-card:: ROME Causal Tracing and Editing
      :link: notebooks/tutorials/rome-research-reproduction
      :link-type: doc
      :class-card: surface

      :octicon:`pencil;2em;sd-text-primary`

      Compare TDHook causal tracing and temporary rank-one edits with the
      official ROME implementation on a preregistered CounterFact slice.

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/methods/integrated-gradients
   notebooks/methods/steering-vectors
   notebooks/methods/linear-probing
   notebooks/methods/bilinear-probing
   notebooks/methods/dimension-estimation
   notebooks/methods/representation-similarity
   notebooks/tutorials/hook-session
   notebooks/tutorials/declared-workflows
   notebooks/tutorials/process-and-distributed-workflows
   notebooks/tutorials/torchrl-ppo
   notebooks/tutorials/chess-value-saliency
   notebooks/tutorials/concept-attribution
   notebooks/tutorials/chess-dimension-estimation
   notebooks/tutorials/othello-research-reproduction
   notebooks/tutorials/weight-circuit-research-reproduction
   notebooks/tutorials/rome-research-reproduction
