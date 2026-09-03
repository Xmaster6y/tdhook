# tdhook API Reference

Key classes, usage patterns, and method implementations.

## Method

Base for all method implementations. Subclasses: `IntegratedGradients`, `Saliency`, `Probing`, `ActivationAddition`, etc.

```python
binding = method.bind(module, in_keys=None, out_keys=None)
with binding as hooked_module:
    result = hooked_module(data)
```

- `in_keys` / `out_keys`: Override for non-TensorDictModule (e.g. HuggingFace: `["input_ids"]`, `["logits"]`).

## BoundMethod / CachedBoundMethod

Returned by `bind()` as a context manager. `CachedBoundMethod` adds a `cache` TensorDict and `clear()` for activation capture.

## BoundModule

Wrapper returned inside `with method.bind(model)`. Callable with TensorDict.

```python
td = hooked_module(td)
# Results written into td
```

### HookSession capture and replacement API

Use `HookSession` for low-level capture, replacement, and early stopping. Register operations inside the session before executing the model.

```python
from tdhook.session import HookSession
from tdhook.targets import Target

source = Target("layers.5.attn", "activation", feature_axis=-1, indices=(0,))
destination = Target("layers.5.mlp", "activation", feature_axis=-1, indices=(0,))
with HookSession(model) as session:
    captured = session.capture(source)
    session.replace(destination, captured)
    output = model(inputs)
```

- **`capture(target, direction=..., detach=...)`** – Capture target values for the session lifetime.
- **`replace(target, value, direction=..., transform=...)`** – Apply a static value or route a live capture to a later compatible target.
- **`stop(module_path)`** – Stop after a module runs and expose its exact partial output.

Captured values are ordered in `captured.values`; use `captured.values[-1]` when exactly one observation is expected.

Directions select the hook location:

| Variant | Direction | Use |
|---------|-----------|-----|
| activation output | `fwd` | Forward output |
| activation input | `fwd_pre` | Forward positional input |
| activation args and kwargs | `fwd_pre_kwargs` | `(args, kwargs)` |
| gradient input | `bwd` | Backward gradient input |
| gradient output | `bwd_pre` | Backward gradient output |

```python
gradient = Target("layers.5.mlp", "gradient", feature_axis=-1, indices=(0,))
with HookSession(model) as session:
    captured_gradient = session.capture(gradient, direction="bwd_pre")
    model(inputs).sum().backward()
```

Targets can select structured output paths, feature indices, or repeated call occurrences. Use `occurrences=(0, 2)` to select repeated calls within each root model pass. See Module Path Resolution below for path syntax.

### Binding control

```python
with binding.disable_hooks():
    ...  # Run without method hooks

with binding.disable() as raw_module:
    ...  # Raw module, no hooks
```

## TensorDict Keys

| Key pattern | Method | Purpose |
|-------------|--------|---------|
| `"input"`, `"output"` | Default | Base I/O for nn.Module |
| `("baseline", "input")` | IntegratedGradients | Baseline for path integral |
| `("attr", "input")` | Attribution | Attribution map |
| `("positive", "input")`, `("negative", "input")` | ActivationAddition | Source prompts |
| `("steer", "module.path")` | ActivationAddition | Steering vector output |
| `"labels"`, `"step_type"` | Probing | Passed via additional_keys |

## Module Path Resolution

Submodule paths in targets and `stop()` resolve via `resolve_submodule_path`:

- `layers[0].attention` – indexing
- `layers[-1]`, `layers[1:3]` – slicing
- `<custom/attr>.submodule` – attributes with special chars (e.g. `block0/module`)
- `m1.<0>.layers` – numeric attribute names

Target paths are relative to the model passed to `HookSession`. Probing methods use regex `key_pattern` to match paths (e.g. `"transformer.h.5.mlp$"`).

---

## Method Implementations

High-level modules by category. All extend `Method` and use `bind(module)`.

### Attribution

Explain which inputs or layers contribute. All write to `("attr", key)`.

| Class | Use |
|-------|-----|
| `Saliency` | Gradient w.r.t. input (or latent). Params: `absolute`, `multiply_by_inputs`, `input_modules`, `target_modules` |
| `IntegratedGradients` | Path integral. Requires `("baseline", "input")`. Params: `n_steps`, `method`, `baseline_key` |
| `GuidedBackpropagation` | ReLU-guided gradients. Params: `input_modules`, `use_inputs` |
| `GradCAM` | Channel-weighted spatial. Params: `modules_to_attribute` (path → `DimsConfig`) |
| `LRP` | Layer-wise Relevance Propagation. Params: `rule_mapper`, `init_attr_grads` |
| `ActivationMaximisation` | PGD to maximise target. Writes to `("attr", "input")` |

```python
from tdhook.attribution import Saliency, IntegratedGradients

with Saliency(init_attr_targets=init_fn).bind(model) as hooked:
    attr = hooked(TensorDict({"input": x})).get(("attr", "input"))

with IntegratedGradients(init_attr_targets=init_fn).bind(model) as hooked:
    attr = hooked(TensorDict({"input": x, ("baseline", "input"): baseline})).get(("attr", "input"))
```

### Latent

| Class | Use |
|-------|-----|
| `ActivationAddition` | Extract `positive - negative` at modules. Requires `("positive", "input")`, `("negative", "input")`. Outputs `("steer", module_key)` |
| `SteeringVectors` | Apply `steer_fn(module_key, output)` at modules |
| `ActivationPatching` | Replace activations via `patch_fn(output, output_to_patch, ...)`. Requires `("patched", "input")` |
| `ActivationCaching` | Cache activations at regex-matched modules. `hooked.binding.cache` |
| `Probing` | Train probes via `ProbeManager` / `BilinearProbeManager`. `additional_keys=["labels", "step_type"]` |
| `TwoNnDimensionEstimator`, `LocalKnnDimensionEstimator`, etc. | Intrinsic dimension of `TensorDict({"data": activations})` |

```python
from tdhook.latent import ActivationAddition, SteeringVectors
from tdhook.latent.activation_caching import ActivationCaching
from tdhook.latent.probing import Probing, ProbeManager

# Extract steering vector
with ActivationAddition(["transformer.h.7.mlp"]).bind(model) as hooked:
    steer = hooked(TensorDict({("positive", "input"): pos, ("negative", "input"): neg}, batch_size=1)).get(("steer", "transformer.h.7.mlp"))

# Cache activations
with ActivationCaching(r"transformer\.h\.\d+\.mlp").bind(model) as hooked:
    hooked(data)
    cache = hooked.binding.cache

# Probing (needs ProbeManager, labels, step_type)
manager = ProbeManager(LogisticRegression, {}, compute_metrics)
with Probing("transformer.h.(0|5|10).mlp$", manager.probe_factory, additional_keys=["labels", "step_type"]).bind(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    hooked(train_td)
    hooked(test_td)
```

### Weights

| Class | Use |
|-------|-----|
| `Pruning` | Zero params by `importance_callback`. Params: `amount_to_prune`, `skip_modules`, `modules_to_prune` |
| `Adapters` | Insert modules: `{path: (adapter, source, target)}` |
| `TaskVectors` | `get_task_vector`, `get_forget_vector`, `get_weights`, `compute_alpha` |

```python
from tdhook.weights import Pruning, Adapters, TaskVectors

with Pruning(importance_callback=fn, amount_to_prune=0.5).bind(model) as hooked:
    hooked(inp)

with Adapters(adapters={"layer.5": (adapter, "layer.5", "layer.5")}).bind(model) as hooked:
    hooked(data)
```
