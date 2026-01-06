Methods
=======

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
      :link: notebooks/methods/integrated-gradients.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-chart-line fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Integrated Gradients</h5>
               <p class="card-text">Compute attribution using integrated gradients.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/methods/steering-vectors.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-compass fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Steering Vectors</h5>
               <p class="card-text">Modify model behavior by adding vectors to intermediate activations.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/methods/linear-probing.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-search fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Linear Probing</h5>
               <p class="card-text">Train classifiers on model representations to understand what information is encoded.</p>
            </div>
         </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/methods/integrated-gradients.ipynb
   notebooks/methods/steering-vectors.ipynb
   notebooks/methods/linear-probing.ipynb
