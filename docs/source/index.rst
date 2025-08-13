:html_theme.sidebar_secondary.remove: true
:sd_hide_title:

`tdhook`
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    start
    features
    tutorials
    api/index
    About <about>

.. grid:: 1 1 2 2
    :class-container: hero
    :reverse:

    .. grid-item::
        .. div::

          .. image:: _static/images/tdhook-logo.png
            :width: 300
            :height: 300

    .. grid-item::

        .. div:: sd-fs-1 sd-font-weight-bold title-bot sd-text-primary image-container

            tdhook

        .. div:: sd-fs-4 sd-font-weight-bold sd-my-0 sub-bot image-container

            Interpreting ``torch`` Deep Neural Networks

        **tdhook** is a package for explaining ``torch`` deep neural networks based on ``tensordict`` and ``torch`` hooks.

        .. div:: button-group

          .. button-ref:: start
            :color: primary
            :shadow:

                  Get Started

          .. button-ref:: tutorials
            :color: primary
            :outline:

                Tutorials

          .. button-ref:: api/index
            :color: primary
            :outline:

                API Reference


.. div:: sd-fs-1 sd-font-weight-bold sd-text-center sd-text-primary sd-mb-5

  Key Features

.. grid:: 1 1 2 2
    :class-container: features

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 150

        .. div::

          **Efficiency**

          Memory and time efficient, no large bundle, only what you need.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **Interpretability**

          Wide range of interpretability methods ready to be plugged in, with a general API to have full control over the interpretability process.
