Tutorials
=========

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
               <h5 class="card-title">Concept Attribution (LRP + RelMax)</h5>
               <p class="card-text">Build a striped concept and visualize concept-conditioned relevance on natural images.</p>
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
               <h5 class="card-title">Chess Dimension Estimation</h5>
               <p class="card-text">Estimate channel-wise intrinsic dimension from chess activations and compare layers, heads, and opening splits.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/tutorials/chess-cross-model-probing.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-chess fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Chess Cross-Model Probing</h5>
               <p class="card-text">Linear probing on Maia models: can one model's latents predict another's moves? Compare absolute best vs best legal.</p>
            </div>
         </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/tutorials/torchrl-ppo.ipynb
   notebooks/tutorials/chess-value-saliency.ipynb
   notebooks/tutorials/concept-attribution.ipynb
   notebooks/tutorials/chess-dimension-estimation.ipynb
   notebooks/tutorials/chess-cross-model-probing.ipynb
