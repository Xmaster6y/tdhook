Getting Started
===============

**tdhook** is a package for explaining ``torch`` deep neural networks based on ``tensordict`` and ``torch`` hooks.
It is designed to be easy to use and to work with the most common interpretability methods.

.. _installation:

Installation
------------

To get started with ``tdhook``, install it with ``pip``.

.. code-block:: console

   pip install tdhook


Basic Example
-------------

Most methods should work with minimal configuration. Here's a basic example of running Integrated Gradients on a VGG16 model:

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

For more detailed examples, see the :doc:`methods` page.

Composition terminology
-----------------------

TDHook's goal is to make every public method available to a declared
**multi-stage pipeline**, coalescing compatible hooks into the fewest safe model
executions.  The contract distinguishes a **composed model** (a model with
multiple inputs, outputs, or submodules), **same-run hook composition**, and
pipeline stages exchanging named TensorDict artifacts.  See the
:doc:`composition` contract for the planner target and capability matrix.
