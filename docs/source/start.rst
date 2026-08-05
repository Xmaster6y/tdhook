Getting Started
===============

**tdhook** applies interpretability methods to ``torch`` models with hooks and
``TensorDict``.

.. _installation:

Installation
------------

.. code-block:: console

   pip install tdhook


Basic Example
-------------

Run Integrated Gradients on a VGG16 model:

.. code-block:: python

    import torch
    from tensordict import TensorDict
    from tdhook.attribution import IntegratedGradients

    # Define attribution target (e.g., zebra class = 340)
    def init_attr_targets(targets, _):
        zebra_logit = targets["output"][..., 340]
        return TensorDict(out=zebra_logit, batch_size=targets.batch_size)

    # Compute attribution
    with IntegratedGradients(init_attr_targets=init_attr_targets).prepare(model) as hooked_model:
        td = TensorDict({
            "input": image_tensor,
            ("baseline", "input"): torch.zeros_like(image_tensor)  # required for integrated gradients
        }).unsqueeze(0)
        td = hooked_model(td)  # Access attribution with td.get(("attr", "input"))

Start with the offline :doc:`notebooks/tutorials/declared-workflows` notebook.
For individual methods, see :doc:`methods`.

Composition terminology
-----------------------

Use a declared **workflow** to run several methods or TensorDict operators.
See :doc:`composition` for execution rules and tested combinations.
