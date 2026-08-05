Tutorials
=========

Tutorials may demonstrate a method on a **composed model**, a **same-run hook
composition**, or a declared **workflow**. These terms follow the
:doc:`composition` contract. Canonical workflows exchange named TensorDict
values and show their preflight model-pass budget before running.

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', (event) => {
      document.querySelectorAll('h5.card-title').forEach(el => {
      el.style.margin = '0';
      });
   });
   </script>

   <style>
      .toctree-wrapper {
         display: none !important;
      }
   </style>

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :link: declared-pipelines
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-diagram-project fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Declared Workflows (Offline)</h5>
               <p class="card-text">Plan and run deterministic concept-attribution and conditioned-dimension workflows with named artifacts.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/tutorials/torchrl-ppo.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-robot fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">TorchRL PPO Action Probing</h5>
               <p class="card-text">Set up a TorchRL PPO agent and use tdhook to probe action representations.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/tutorials/chess-value-saliency.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-chess fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Chess Value Saliency</h5>
               <p class="card-text">Compute attribution maps for chess model predictions using saliency methods.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/tutorials/concept-attribution.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-map fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Concept Attribution Visualisation (Extended)</h5>
               <p class="card-text">Optional natural-image visualisation that builds on the declared concept-attribution workflow.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/tutorials/chess-dimension-estimation.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-chess-board fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Chess Dimension Visualisation (Extended)</h5>
               <p class="card-text">Optional chess rendering and plots downstream of declared conditioned-dimension artifacts.</p>
            </div>
         </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/tutorials/torchrl-ppo.ipynb
   declared-pipelines
   notebooks/tutorials/declared-pipelines.ipynb
   notebooks/tutorials/chess-value-saliency.ipynb
   notebooks/tutorials/concept-attribution.ipynb
   notebooks/tutorials/chess-dimension-estimation.ipynb
