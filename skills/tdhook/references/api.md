# tdhook API Reference

Key classes and usage patterns.

## HookingContextFactory

Base for all method implementations. Subclasses: `IntegratedGradients`, `Saliency`, `Probing`, `ActivationAddition`, etc.

```python
method.prepare(module, in_keys=None, out_keys=None, return_context=True)
# Returns HookingContext (context manager) or HookedModule if return_context=False
```

- `in_keys` / `out_keys`: Override for non-TensorDictModule (e.g. HuggingFace: `["input_ids"]`, `["logits"]`).
- `return_context=False`: Returns `HookedModule` directly; context is auto-entered.

## HookingContext / HookingContextWithCache

Returned by `prepare()` (as context manager). `HookingContextWithCache` adds a `cache` TensorDict and `clear()`; used by methods that cache activations (e.g. probing).

## HookedModule

Wrapper returned inside `with method.prepare(model)`. Callable with TensorDict.

```python
td = hooked_module(td)
# Results written into td
```

### Get / Set / Save API

Use `run(data)` for low-level access. Inside the context, register hooks before the forward executes on exit.

```python
with hooked_module.run(data) as run:
    run.save("layers.5.mlp")           # Capture forward output into run.cache
    run.set("layers.5.attn", override)  # Override activation at that module
    run.get("layers.3.mlp", cache_key="custom")  # Cache with custom key
```

- **`save(key)`** – Capture activation; auto cache key (`key_output` or `key_grad_input`).
- **`get(key, cache_key=...)`** – Same as save but explicit cache key; returns `CacheProxy`.
- **`set(key, value)`** – Replace activation. `value` can be a tensor or `CacheProxy.resolve()`.
- **`stop(key)`** – Raise `EarlyStoppingException` at that module (short-circuit).

Direction-specific variants (all take same args as base):

| Variant | Direction | Use |
|---------|-----------|-----|
| `save` / `get` / `set` | `fwd` | Forward output |
| `save_input` / `get_input` / `set_input` | `fwd_pre` | Forward input |
| `save_grad` / `get_grad` / `set_grad` | `bwd` | Gradient input (requires `grad_enabled=True`) |
| `save_grad_output` / `get_grad_output` / `set_grad_output` | `bwd_pre` | Gradient output |

```python
with hooked_module.run(data, grad_enabled=True) as run:
    run.save_input("layers.0")
    run.save_grad("layers.5.mlp")
```

Common params: `callback=fn`, `prepend=False`, `relative=True`. Paths use module resolution (see Module Path Resolution).

**CacheProxy** – Returned by `get`/`save`. Call `proxy.resolve()` after the run to read the cached tensor.

**Run options**: `run(data, cache=None, run_name="run", run_sep=".", grad_enabled=False, run_callback=None)`. Pass `cache=my_td` to write saves into a shared TensorDict; `run_callback` overrides the default `module(data)` execution.

### Context control

```python
with hooked_module.disable_context_hooks():
    ...  # Run without method hooks

with hooked_module.disable_context() as raw_module:
    ...  # Raw module, no hooks

hooked_module.restore()  # When using prepare(return_context=False)
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

Submodule keys in `set`/`get`/`save`/`stop` resolve via `resolve_submodule_path`:

- `layers[0].attention` – indexing
- `layers[-1]`, `layers[1:3]` – slicing
- `<custom/attr>.submodule` – attributes with special chars (e.g. `block0/module`)
- `m1.<0>.layers` – numeric attribute names

Paths default to relative to `td_module`; use `relative=False` for absolute. Probing methods use regex `key_pattern` to match paths (e.g. `"transformer.h.5.mlp$"`).
