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

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/tutorials/torchrl-ppo.ipynb
