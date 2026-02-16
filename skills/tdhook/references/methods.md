# tdhook Methods

High-level modules by category: **Attribution**, **Latent**, **Weights**. Each section lists modules and their use cases.

---

## Attribution

Explain which inputs or layers contribute to an output. All use `prepare(module)` and write to `("attr", key)` in the output TensorDict.

### Saliency

Gradient of the target w.r.t. input (or latent). Fast baseline.

```python
from tdhook.attribution import Saliency

with Saliency(init_attr_targets=init_attr_targets).prepare(model) as hooked:
    td = hooked(TensorDict({"input": x}, batch_size=...))
    attr = td.get(("attr", "input"))
```

**Params**: `absolute`, `multiply_by_inputs`, `input_modules`, `target_modules`, `use_inputs`, `use_outputs`, `output_grad_callbacks`.

### IntegratedGradients

Path integral from baseline to input. Requires `("baseline", "input")` in TensorDict.

```python
from tdhook.attribution import IntegratedGradients

with IntegratedGradients(init_attr_targets=init_attr_targets).prepare(model) as hooked:
    td = hooked(TensorDict({"input": x, ("baseline", "input"): baseline}, batch_size=...))
    attr = td.get(("attr", "input"))
```

**Params**: `n_steps`, `method`, `baseline_key`, `multiply_by_inputs`, `compute_convergence_delta`. Labels: `init_attr_targets_with_labels`.

### GuidedBackpropagation

ReLU-guided gradients (positive only). **Params**: `input_modules`, `use_inputs`.

### GradCAM

Channel-weighted spatial attribution. **Params**: `modules_to_attribute` (dict of module path → `DimsConfig` for weight/feature pooling).

### LRP

Layer-wise Relevance Propagation. Custom rules (EpsilonPlus, AlphaBeta, etc.) via `rule_mapper`.

```python
from tdhook.attribution import LRP
from tdhook.attribution.lrp_helpers.rules import EpsilonPlus

with LRP(rule_mapper=EpsilonPlus(epsilon=1e-6), init_attr_grads=init_fn).prepare(model) as hooked:
    td = hooked(TensorDict({"input": x}))
    attr = td.get(("attr", "input"))
```

### ActivationMaximisation

Find input that maximises a target via PGD. Writes maximised input to `("attr", "input")`. **Params**: `modules_to_maximise`, `init_attr_targets`, PGD args (`alpha`, `n_steps`, etc.).

---

## Latent

Work with intermediate activations: extract, patch, steer, cache, or probe them.

### ActivationAddition / SteeringVectors

- **ActivationAddition**: Extract `positive - negative` at specified modules. Requires `("positive", "input")` and `("negative", "input")`. Outputs `("steer", module_key)`.
- **SteeringVectors**: Apply custom `steer_fn(module_key, output)` at modules during forward.

```python
from tdhook.latent import ActivationAddition, SteeringVectors

# Extract
with ActivationAddition(["transformer.h.7.mlp"]).prepare(model) as hooked:
    td = hooked(TensorDict({("positive", "input"): pos, ("negative", "input"): neg}, batch_size=1))
    steer = td.get(("steer", "transformer.h.7.mlp"))

# Apply
with SteeringVectors(modules_to_steer=["transformer.h.7.mlp"], steer_fn=steer_fn).prepare(model) as hooked:
    hooked(base_td)
```

### ActivationPatching

Replace activations at modules using a `patch_fn(output, output_to_patch, ...)`. Requires `("patched", "input")` for the patch source. **Params**: `modules_to_patch`, `patch_fn`.

### ActivationCaching

Cache activations at regex-matched modules. Use when you need raw activations for later analysis (no probes).

```python
from tdhook.latent.activation_caching import ActivationCaching

with ActivationCaching("transformer\.h\.\d+\.mlp", relative=False).prepare(model) as hooked:
    hooked(data)
    cache = hooked.hooking_context.cache  # TensorDict of module paths → tensors
```

### Probing

Train classifiers on cached activations. Regex `key_pattern` matches module paths. Requires `ProbeManager` / `BilinearProbeManager` for fit/predict and `additional_keys` (e.g. `"labels"`, `"step_type"`).

```python
from tdhook.latent.probing import Probing, ProbeManager
from sklearn.linear_model import LogisticRegression

manager = ProbeManager(LogisticRegression, {}, compute_metrics)
with Probing("transformer.h.(0|5|10).mlp$", manager.probe_factory, additional_keys=["labels", "step_type"]).prepare(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    hooked(TensorDict({"input_ids": ids, "labels": labels, "step_type": "fit"}, batch_size=...))
    hooked(TensorDict({"input_ids": test_ids, "labels": test_labels, "step_type": "predict"}, batch_size=...))
# manager.fit_metrics, manager.predict_metrics
```

Bilinear: `BilinearProbeManager` with `LowRankBilinearEstimator`, pairs of layers. Call `manager.before_all()` / `manager.after_all()` when `h1 != h2`.

### Dimension Estimation

Estimate intrinsic dimension of activation manifolds. Estimators: `TwoNnDimensionEstimator`, `LocalKnnDimensionEstimator`, `LocalPcaDimensionEstimator`, `CaPcaDimensionEstimator`. Data: `TensorDict({"data": activations}, batch_size=[])`.

```python
from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator, LocalKnnDimensionEstimator

twonn = TwoNnDimensionEstimator(return_xy=True)
td_out = twonn(td.clone())
dim = td_out["dimension"].item()
```

---

## Weights

Modify or analyse model parameters.

### Pruning

Zero out parameters by importance. Uses `importance_callback(parameter, ...)` and `amount_to_prune`. Restores on exit.

```python
from tdhook.weights.pruning import Pruning

pruning = Pruning(importance_callback=importance_fn, amount_to_prune=0.5)
with pruning.prepare(model) as hooked:
    hooked(inp)  # Pruned forward
# Restored
```

**Params**: `skip_modules`, `relative_path`, `modules_to_prune` (per-module amount).

### Adapters

Insert modules at layer boundaries. Dict: `{module_path: (adapter_module, source_module, target_module)}`.

```python
from tdhook.weights.adapters import Adapters

adapters = {"linear2": (DoubleAdapter(), "linear2", "linear2")}
with Adapters(adapters=adapters).prepare(model) as hooked:
    hooked(data)
```

### TaskVectors

Compute and combine task vectors (finetuned - pretrained). `get_task_vector`, `get_forget_vector`, `get_weights`, `compute_alpha`.

```python
from tdhook.weights.task_vectors import TaskVectors

with TaskVectors(alphas=[0.1, 0.5, 1.0], get_test_accuracy=fn, get_control_adequacy=fn).prepare(pretrained) as hooked:
    vector = hooked.get_task_vector(finetuned)
    alpha = hooked.hooking_context.compute_alpha(vector)
    new_weights = hooked.get_weights(vector, forget_vector, alpha=0.1)
```
