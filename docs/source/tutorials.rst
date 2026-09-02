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
