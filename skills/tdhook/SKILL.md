---
name: tdhook
description: Provides guidance for interpreting and manipulating neural network internals using tdhook with TensorDict and PyTorch hooks. Use when needing attribution maps, activation analysis, probing, steering, activation patching, or weight-level interventions on PyTorch or TensorDict models.
version: 1.0.0
author: tdhook
license: MIT
tags: [tdhook, Interpretability, Attribution, Activation Analysis, Probing, Steering, TensorDict, PyTorch Hooks]
dependencies: [tdhook, tensordict>=0.3.0, torch>=2.0.0]
---

# tdhook: Interpretability with TensorDict and PyTorch Hooks

tdhook enables interpretability experiments on any PyTorch model via TensorDict I/O and hook-based access to activations. One pattern, many methods: `with Method(...).prepare(model) as hooked: td = hooked(td)`.

**GitHub**: [Xmaster6y/tdhook](https://github.com/Xmaster6y/tdhook)
**Docs**: [tdhook.readthedocs.io](https://tdhook.readthedocs.io)
**Paper**: [TDHook: A Lightweight Framework for Interpretability](https://arxiv.org/abs/2509.25475) (2025)

## Key Value Proposition

**TensorDict-native, hook-based interpretability**: Same prepare/forward pattern for attribution, probing, steering, patching, and weight interventions. Works with nn.Module, TensorDictModule, TorchRL agents, and custom architectures.

```python
# Attribution
with IntegratedGradients(init_attr_targets=init_fn).prepare(model) as hooked:
    td = hooked(TensorDict({"input": x, ("baseline", "input"): baseline}))
    attr = td.get(("attr", "input"))

# Activation patching
with ActivationPatching(["layer.8"], patch_fn=patch_fn).prepare(model) as hooked:
    td = hooked(TensorDict({"input": corrupt, ("patched", "input"): clean}))
    patched_out = td.get(("patched", "output"))
```

## When to Use tdhook

**Use tdhook when you need to:**
- Compute input attributions (gradient, path-integral, LRP) with flexible I/O
- Capture or inspect activations at arbitrary layers
- Patch activations for counterfactual or causal tracing experiments
- Train probes (linear, bilinear) on representations
- Steer model behavior via activation addition or custom interventions
- Apply weight-level changes (pruning, adapters, task vectors)
- Work with TensorDictModule, TorchRL, or HuggingFace models

**Consider alternatives when:**
- You need remote execution on 70B+ models → Use **nnsight** (NDIF)
- You want declarative, shareable intervention configs → Use **pyvene**
- You prefer TransformerLens-style cached activations → **TransformerLens** may suit

## Installation

```bash
pip install tdhook
```

Standard: `tensordict`, `torch`. Optional per-method: `transformers`, `timm`, `torchrl`, `sklearn`, etc.

## Core Concepts

### Prepare–Forward Pattern

All methods use `HookingContextFactory.prepare()` and return a context manager. Forward with a TensorDict; hooks run automatically.

```python
from tensordict import TensorDict

with method.prepare(model) as hooked:
    td = TensorDict({"input": x, ...}, batch_size=...)
    td = hooked(td)
    result = td.get(key)  # method-specific
```

### Low-Level Hooks (HookedModule.run)

For custom logic, use the get/set/save API inside a run context:

```python
with hooked.run(data) as run:
    run.save("layers.5.mlp")           # capture activation
    run.set("layers.5.attn", override)  # override activation
proxy = run.save("...").resolve()       # read after run
```

### TensorDict Keys

| Key pattern | Purpose |
|-------------|---------|
| `"input"`, `"output"` | Default I/O |
| `("baseline", "input")` | Attribution baseline |
| `("positive", "input")`, `("negative", "input")` | Steering extraction |
| `("patched", "input")` | Patching source |
| `("attr", key)` | Attribution output |

## Workflow 1: Input Attribution

Which inputs matter for the model’s output? Use Saliency, IntegratedGradients, GradCAM, LRP.

```python
from tdhook.attribution import Saliency, IntegratedGradients

def init_attr_targets(targets, _):
    return TensorDict(out=targets["output"][..., class_idx], batch_size=targets.batch_size)

# Gradient-based
with Saliency(init_attr_targets=init_attr_targets).prepare(model) as hooked:
    td = hooked(TensorDict({"input": image}, batch_size=...))
    attr = td.get(("attr", "input"))

# Path integral (needs baseline)
with IntegratedGradients(init_attr_targets=init_attr_targets).prepare(model) as hooked:
    td = hooked(TensorDict({"input": image, ("baseline", "input"): baseline}, batch_size=...))
    attr = td.get(("attr", "input"))
```

## Workflow 2: Activation Analysis & Caching

Capture activations at specified layers for analysis (probing, dimension estimation, etc.).

```python
from tdhook.latent.activation_caching import ActivationCaching

with ActivationCaching("transformer\\.h\\.\\d+\\.mlp", relative=False).prepare(model) as hooked:
    hooked(data)
    cache = hooked.hooking_context.cache  # TensorDict of path → tensor
```

Or use the low-level API for one-off capture:

```python
with hooked.run(data) as run:
    h5 = run.save("transformer.h.5.mlp")
act = h5.resolve()
```

## Workflow 3: Activation Patching

Replace activations at modules with values from another run (e.g. clean vs corrupted).

```python
from tdhook.latent.activation_patching import ActivationPatching

def patch_fn(output, output_to_patch, **_):
    output[:, pos, :] = output_to_patch[:, pos, :]
    return output

with ActivationPatching(["layer.8"], patch_fn=patch_fn).prepare(model) as hooked:
    td = TensorDict({"input": corrupt, ("patched", "input"): clean}, batch_size=...)
    td = hooked(td)
    patched_logits = td.get(("patched", "output"))
```

## Workflow 4: Representation Probing

Train classifiers on cached activations (linear or bilinear). Use `Probing` + `ProbeManager` / `BilinearProbeManager` with `additional_keys` (`labels`, `step_type`).

```python
from tdhook.latent.probing import Probing, ProbeManager
from sklearn.linear_model import LogisticRegression

manager = ProbeManager(LogisticRegression, {}, compute_metrics)
with Probing("transformer.h.(0|5|10).mlp$", manager.probe_factory, additional_keys=["labels", "step_type"]).prepare(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    hooked(TensorDict({"input_ids": ids, "labels": labels, "step_type": "fit"}, batch_size=...))
    hooked(TensorDict({"input_ids": test_ids, "labels": test_labels, "step_type": "predict"}, batch_size=...))
# manager.fit_metrics, manager.predict_metrics
```

## Workflow 5: Steering & Activation Addition

Extract a steering vector (positive − negative) and apply it during forward.

```python
from tdhook.latent import ActivationAddition, SteeringVectors

# Extract
with ActivationAddition(["transformer.h.7.mlp"]).prepare(model) as hooked:
    td = hooked(TensorDict({("positive", "input"): pos, ("negative", "input"): neg}, batch_size=1))
    steer = td.get(("steer", "transformer.h.7.mlp"))

# Apply
def steer_fn(module_key, output):
    return output + scale * steer_vec
with SteeringVectors(modules_to_steer=["transformer.h.7.mlp"], steer_fn=steer_fn).prepare(model) as hooked:
    hooked(base_td)
```

## Workflow 6: Weight-Level Interventions

Prune parameters, insert adapters, or merge task vectors.

```python
from tdhook.weights import Pruning, Adapters, TaskVectors

# Pruning
with Pruning(importance_callback=fn, amount_to_prune=0.5).prepare(model) as hooked:
    hooked(inp)

# Adapters
with Adapters(adapters={"layer.5": (adapter, "layer.5", "layer.5")}).prepare(model) as hooked:
    hooked(data)
```

## Common Issues & Solutions

### TensorDict keys
Use tuple keys for nested: `("attr", "input")`, `("baseline", "input")`, `("positive", "input")`. Read with `td.get(("attr", "input"))`.

### HuggingFace models
Override I/O: `in_keys=["input_ids"]`, `out_keys=["logits"]` in `prepare()`.

### Probing lifecycle
For BilinearProbeManager with h1≠h2, call `manager.before_all()` before forwards and `manager.after_all()` after.

### Module path resolution
Paths support indexing (`layers[0]`), slicing (`layers[1:5]`), and special chars (`<block0/module>.layers`). See api.md.

## Comparison with Other Tools

| Feature | tdhook | nnsight | TransformerLens |
|---------|--------|---------|-----------------|
| TensorDict I/O | Yes | No | No |
| TorchRL / TensorDictModule | Yes | No | No |
| Remote (70B+) | No | Yes (NDIF) | No |
| Attribution (IG, Saliency, LRP) | Yes | Via gradients | Limited |
| Probing, steering, patching | Yes | Yes | Yes |
| Weight-level (pruning, adapters) | Yes | No | No |

## Reference Documentation

| File | Contents |
|------|----------|
| [references/README.md](references/README.md) | Overview and quick links |
| [references/api.md](references/api.md) | HookingContextFactory, HookedModule get/set/save, paths |
| [references/methods.md](references/methods.md) | Attribution, latent, weights by category |
| [references/tutorials.md](references/tutorials.md) | Use-case tutorials (VGG attribution, GPT-2 probing, etc.) |

## External Resources

- [Documentation](https://tdhook.readthedocs.io)
- [GitHub](https://github.com/Xmaster6y/tdhook)
- [arXiv paper](https://arxiv.org/abs/2509.25475)
