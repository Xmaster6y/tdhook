Colab notebook support
======================

The method notebooks listed in the README are the supported Colab paths. They
install the current released ``tdhook`` package, so each demo follows the
public API as it evolves. Run the setup cell first in a fresh Python 3 Colab
runtime, then run the notebook from top to bottom.

Supported paths
---------------

All six method notebooks run on CPU; an accelerator is optional and changes
only execution time.  The dimension-estimation notebook additionally installs
``scikit-learn``.  Notebook examples may download standard model weights or
datasets when their execution cells request them; Colab's ephemeral filesystem
does not retain those downloads across runtime resets.

The lightweight compatibility smoke path is the Integrated Gradients notebook.
The test suite validates that every supported method notebook installs the
released package through an executable setup cell without requiring network
access or a hosted Colab runtime.

Extended tutorials
------------------

The tutorials page also contains extended examples.  They are not part of the
supported Colab smoke path: ``torchrl-ppo`` requires ``torchrl`` and an RL
environment; the chess notebooks require ``lczerolens`` (and, for dimension
visualisation, ``datasets``); concept attribution can download an image model
and dataset.  Use a local environment with their declared dependencies for
those tutorials, and select a GPU only when their model or runtime makes CPU
execution impractical.
