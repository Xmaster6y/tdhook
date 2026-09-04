<div align="center">
<img src="https://raw.githubusercontent.com/Xmaster6y/tdhook/refs/heads/main/docs/source/_static/images/tdhook-logo.png" alt="logo" width="200"/>
</div>

<h1 align=center><code>tdhook</code> 🤖🪝</h1>

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://tdhook.readthedocs.io)
[![tdhook](https://img.shields.io/pypi/v/tdhook?color=purple)](https://pypi.org/project/tdhook/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/Xmaster6y/tdhook/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python versions](https://img.shields.io/pypi/pyversions/tdhook.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25475-b31b1b.svg)](https://arxiv.org/abs/2509.25475)

[![codecov](https://codecov.io/gh/Xmaster6y/tdhook/graph/badge.svg?token=JKJAWB451A)](https://codecov.io/gh/Xmaster6y/tdhook)
![ci](https://github.com/Xmaster6y/tdhook/actions/workflows/ci.yml/badge.svg)
![publish](https://github.com/Xmaster6y/tdhook/actions/workflows/publish.yml/badge.svg)
[![docs](https://readthedocs.org/projects/tdhook/badge/?version=latest)](https://tdhook.readthedocs.io/en/latest/?badge=latest)

Composable interpretability for PyTorch with `TensorDict` and `torch` hooks.

## Getting Started

Install TDHook from PyPI:

```console
pip install tdhook
```

TDHook methods wrap an ordinary PyTorch model for the lifetime of a context
manager. Inputs, baselines, model outputs, and interpretability results use
explicit `TensorDict` keys:

```python
import torch
from torch import nn
from tensordict import TensorDict
from tdhook.attribution import IntegratedGradients

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
inputs = torch.tensor([[0.2, -0.1, 0.4, 0.7]])

def select_score(outputs, _):
    score = outputs["output"][..., 0]
    return TensorDict(score=score, batch_size=outputs.batch_size)

data = TensorDict(
    {
        "input": inputs,
        ("baseline", "input"): torch.zeros_like(inputs),
    },
    batch_size=[1],
)

with IntegratedGradients(init_attr_targets=select_score).bind(model) as hooked_model:
    result = hooked_model(data)

attributions = result["attr", "input"]
```

The context installs and removes the hooks; the returned attribution has the
same shape as `inputs`. See [Getting Started](https://tdhook.readthedocs.io/en/latest/start.html)
for the annotated version.

## Learn by example

The [tutorial gallery](https://tdhook.readthedocs.io/en/latest/tutorials.html)
collects all maintained method and end-to-end notebooks. Launch a method
notebook directly in Colab:

- [Integrated Gradients](https://tdhook.readthedocs.io/en/latest/notebooks/methods/integrated-gradients.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/tdhook/blob/main/docs/source/notebooks/methods/integrated-gradients.ipynb)
- [Steering Vectors](https://tdhook.readthedocs.io/en/latest/notebooks/methods/steering-vectors.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/tdhook/blob/main/docs/source/notebooks/methods/steering-vectors.ipynb)
- [Linear Probing](https://tdhook.readthedocs.io/en/latest/notebooks/methods/linear-probing.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/tdhook/blob/main/docs/source/notebooks/methods/linear-probing.ipynb)
- [Bilinear Probing](https://tdhook.readthedocs.io/en/latest/notebooks/methods/bilinear-probing.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/tdhook/blob/main/docs/source/notebooks/methods/bilinear-probing.ipynb)
- [Dimension Estimation](https://tdhook.readthedocs.io/en/latest/notebooks/methods/dimension-estimation.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/tdhook/blob/main/docs/source/notebooks/methods/dimension-estimation.ipynb)
- [Representation Similarity](https://tdhook.readthedocs.io/en/latest/notebooks/methods/representation-similarity.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/tdhook/blob/main/docs/source/notebooks/methods/representation-similarity.ipynb)

Use the generated [API reference](https://tdhook.readthedocs.io/en/latest/api/index.html)
for exact signatures. The [TDHook agent skill](skills/tdhook/SKILL.md) provides
guidance for attribution, activation analysis, probing, steering, and
weight-level interventions.

## Config

This project uses [`uv`](https://docs.astral.sh/uv/) to manage python dependencies and run scripts, as well as [`just`](https://github.com/casey/just) to run commands.

## Benchmarks

The [maintained benchmark suite](benchmarks/README.md) checks current TDHook
attribution, capture, and intervention behavior against reference libraries
before recording versioned timing and memory results. It provides a cheap local
smoke mode and a documented full mode; it does not claim to reproduce the
historical v0.1 paper measurements.

## Citation

If you're using `tdhook` in your research, please cite it using the following BibTeX entry:

```
@misc{poupart2025tdhooklightweightframeworkinterpretability,
      title={TDHook: A Lightweight Framework for Interpretability},
      author={Yoann Poupart},
      year={2025},
      eprint={2509.25475},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.25475},
}
```

## License

`tdhook` is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
