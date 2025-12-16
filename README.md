<img src="https://raw.githubusercontent.com/Xmaster6y/tdhook/refs/heads/main/docs/source/_static/images/tdhook-logo.png" alt="logo" width="200"/>

# tdhook 🤖🪝

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://tdhook.readthedocs.io)
[![tdhook](https://img.shields.io/pypi/v/tdhook?color=purple)](https://pypi.org/project/tdhook/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/Xmaster6y/tdhook/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python versions](https://img.shields.io/pypi/pyversions/tdhook.svg)](https://www.python.org/downloads/)

[![codecov](https://codecov.io/gh/Xmaster6y/tdhook/graph/badge.svg?token=JKJAWB451A)](https://codecov.io/gh/Xmaster6y/tdhook)
![ci](https://github.com/Xmaster6y/tdhook/actions/workflows/ci.yml/badge.svg)
![publish](https://github.com/Xmaster6y/tdhook/actions/workflows/publish.yml/badge.svg)
[![docs](https://readthedocs.org/projects/tdhook/badge/?version=latest)](https://tdhook.readthedocs.io/en/latest/?badge=latest)

Interpretability with `tensordict` and `torch` hooks.

## Getting Started

Most methods should work with minimal configuration. Here's a basic exemple of running integrated gradients on a VGG16 model:

```python
from tdhook.attribution import Saliency, IntegratedGradients

# Define attribution target (e.g., zebra class = 340)
def init_attr_targets(targets, _):
    zebra_logit = targets["output"][..., 340]
    return TensorDict(out=zebra_logit, batch_size=targets.batch_size)

# Compute attribution
with Saliency(
    IntegratedGradients(init_attr_targets=init_attr_targets)
).prepare(model) as hooked_model:
    td = TensorDict({
        "input": image_tensor,
        ("baseline", "input"): torch.zeros_like(image_tensor) # required for integrated gradients
    }).unsqueeze(0)
    td = hooked_model(td) # Access attribution with td.get(("attr", "input"))
```

For more examples, see the [documentation](https://tdhook.readthedocs.io).

## Python Config

Using `uv` to manage python dependencies and run scripts.

## Scripts

This project uses [Just](https://github.com/casey/just) to manage scripts, refer to their instructions for installation.
