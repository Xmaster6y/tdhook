# Contributor-only Source-tree Navigation

This page is for contributors working in a clone of the tdhook repository. It
is not needed to install or use the skill: normal workflows use the importable
`tdhook` package APIs described in [SKILL.md](../SKILL.md) and the other
references bundled alongside it.

## Source Checkout Layout

In a contributor checkout, the package source is under `src/tdhook/`:

```text
src/tdhook/
├── __init__.py           # Public API
├── modules.py            # HookedModule and TensorDict wrappers
├── hooks.py              # Hook factories and low-level handles
├── contexts.py           # HookingContext, HookingContextFactory
├── runtime.py            # Immutable hook programs and bound execution
├── session.py            # Public imperative HookSession lifecycle
├── targets.py            # Serializable activation, gradient, and parameter targets
├── paths.py              # Safe submodule-path resolution
├── metrics.py            # InfidelityMetric, SensitivityMetric
├── sources.py            # Baseline/source utilities
├── _types.py             # Nested-key validation and composition helpers
└── _optional_deps.py     # Lazy imports (sklearn, captum, etc.)
```

## Attribution

```text
attribution/
├── __init__.py           # Saliency, IntegratedGradients, LRP, etc.
├── saliency.py
├── integrated_gradients.py
├── guided_backpropagation.py
├── grad_cam.py
├── activation_maximisation.py
├── lrp.py
├── gradient_helpers/     # Shared gradient/IG logic
└── lrp_helpers/          # LRP rules, mappers
```

## Latent

```text
latent/
├── __init__.py
├── steering_vectors.py   # ActivationAddition, SteeringVectors
├── activation_patching.py
├── activation_caching.py
├── representation_similarity.py
├── probing/              # Probing, ProbeManager, BilinearProbeManager
│   ├── context.py
│   ├── managers.py
│   └── estimators.py
└── dimension_estimation/ # TwoNn, LocalKnn, LocalPca, CaPca
```

## Weights

```text
weights/
├── __init__.py
├── pruning.py
├── adapters.py
└── task_vectors.py
```

## Key Entry Points

| Module | Classes |
|--------|---------|
| `tdhook.attribution` | Saliency, IntegratedGradients, GradCAM, GuidedBackpropagation, LRP, ActivationMaximisation |
| `tdhook.latent` | ActivationAddition, SteeringVectors, ActivationPatching, ActivationCaching |
| `tdhook.latent.probing` | Probing, ProbeManager, BilinearProbeManager |
| `tdhook.latent.dimension_estimation` | TwoNnDimensionEstimator, LocalKnnDimensionEstimator, etc. |
| `tdhook.weights` | Pruning, Adapters, TaskVectors |
