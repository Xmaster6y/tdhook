"""Differentiable, optimization-derived activation interventions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn

from tdhook.hooks import HookDirection
from tdhook.session import HookSession
from tdhook.targets import Target

InterventionStatus = Literal["converged", "stalled", "budget_exhausted", "non_finite"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Serializable optimizer configuration for one intervention value."""

    name: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 1e-2
    kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"unsupported optimizer: {self.name!r}")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if not torch.isfinite(torch.tensor(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(self.kwargs, Mapping):
            raise TypeError("optimizer kwargs must be a mapping")
        try:
            json.dumps(dict(self.kwargs), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("optimizer kwargs must be JSON-serializable") from exc

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "learning_rate": float(self.learning_rate), "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Convergence and stalled-progress criteria.

    Optimization converges once the minimized loss is at or below
    ``objective_threshold``. It stalls after ``patience`` consecutive steps
    improve by less than ``min_delta``. Omitting both criteria consumes the
    full step budget.
    """

    objective_threshold: float | None = None
    min_delta: float = 0.0
    patience: int | None = None

    def __post_init__(self) -> None:
        if self.objective_threshold is not None:
            if isinstance(self.objective_threshold, bool) or not isinstance(self.objective_threshold, (int, float)):
                raise TypeError("objective_threshold must be a number")
            if not torch.isfinite(torch.tensor(self.objective_threshold)):
                raise ValueError("objective_threshold must be finite")
        if isinstance(self.min_delta, bool) or not isinstance(self.min_delta, (int, float)):
            raise TypeError("min_delta must be a number")
        if not torch.isfinite(torch.tensor(self.min_delta)) or self.min_delta < 0:
            raise ValueError("min_delta must be non-negative and finite")
        if self.patience is not None:
            if isinstance(self.patience, bool) or not isinstance(self.patience, int):
                raise TypeError("patience must be an integer")
            if self.patience <= 0:
                raise ValueError("patience must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_threshold": self.objective_threshold,
            "min_delta": float(self.min_delta),
            "patience": self.patience,
        }


@dataclass(frozen=True)
class InterventionObjective:
    """A scalar loss and named scalar terms reported for one step."""

    loss: Tensor
    terms: Mapping[str, Tensor | float | int] = field(default_factory=dict)


InterventionObjectiveFn = Callable[[object, Tensor], Tensor | InterventionObjective]
PreservationRegularizer = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class InterventionSpec:
    """Configuration for one stage of a sequential optimized intervention."""

    target: Target
    objective: InterventionObjectiveFn = field(repr=False, compare=False)
    objective_name: str
    max_steps: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    initial_value: Tensor | None = field(default=None, repr=False, compare=False)
    preservation_regularizer: PreservationRegularizer | None = field(default=None, repr=False, compare=False)
    direction: HookDirection | None = None

    def __post_init__(self) -> None:
        if self.target.kind != "activation":
            raise ValueError("optimized interventions require activation targets")
        if not callable(self.objective):
            raise TypeError("objective must be callable")
        if not isinstance(self.objective_name, str):
            raise TypeError("objective_name must be a string")
        if not self.objective_name:
            raise ValueError("objective_name must be non-empty")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError("max_steps must be an integer")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.initial_value is not None and not isinstance(self.initial_value, Tensor):
            raise TypeError("initial_value must be a tensor")
        if self.preservation_regularizer is not None and not callable(self.preservation_regularizer):
            raise TypeError("preservation_regularizer must be callable")
        if self.direction not in {None, "fwd", "fwd_pre", "fwd_pre_kwargs"}:
            raise ValueError("optimized interventions require a forward hook direction")


@dataclass(frozen=True)
class InterventionStepArtifact:
    """Serializable objective values observed at one optimization step."""

    step: int
    terms: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"step": self.step, "terms": dict(self.terms)}


@dataclass(frozen=True)
class InterventionStageArtifact:
    """Serializable provenance and outcome for one optimized target."""

    target: Target
    objective_name: str
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingConfig
    step_budget: int
    steps_completed: int
    model_pass_budget: int
    model_passes: int
    status: InterventionStatus
    history: tuple[InterventionStepArtifact, ...]
    value_shape: tuple[int, ...]
    value_dtype: str
    value_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "objective_name": self.objective_name,
            "optimizer": self.optimizer.to_dict(),
            "early_stopping": self.early_stopping.to_dict(),
            "step_budget": self.step_budget,
            "steps_completed": self.steps_completed,
            "model_pass_budget": self.model_pass_budget,
            "model_passes": self.model_passes,
            "status": self.status,
            "history": [item.to_dict() for item in self.history],
            "value_shape": list(self.value_shape),
            "value_dtype": self.value_dtype,
            "value_sha256": self.value_sha256,
        }


@dataclass(frozen=True)
class OptimizedIntervention:
    """An optimized value paired with the target where it is applied."""

    target: Target
    value: Tensor = field(repr=False, compare=False)
    direction: HookDirection | None = None


@dataclass(frozen=True)
class OptimizedInterventionResult:
    """Downstream output, optimized values, and serializable run artifacts."""

    output: object = field(repr=False, compare=False)
    interventions: tuple[OptimizedIntervention, ...]
    stages: tuple[InterventionStageArtifact, ...]
    model_pass_budget: int
    model_passes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "model_pass_budget": self.model_pass_budget,
            "model_passes": self.model_passes,
        }


@dataclass
class _ModuleState:
    training: tuple[tuple[nn.Module, bool], ...]
    parameters: tuple[tuple[nn.Parameter, Tensor, bool, Tensor | None], ...]
    buffers: tuple[tuple[Tensor, Tensor], ...]


def optimize_interventions(
    model: nn.Module,
    model_args: Sequence[object],
    specs: Sequence[InterventionSpec],
    *,
    model_kwargs: Mapping[str, object] | None = None,
    frozen_modules: Sequence[nn.Module] = (),
    seed: int | None = None,
) -> OptimizedInterventionResult:
    """Optimize activation values sequentially and run the model with all of them.

    Each stage captures its target with earlier optimized interventions already
    applied, then minimizes its objective while keeping every parameter in
    ``model`` and ``frozen_modules`` frozen. One final model pass applies every
    optimized value and supplies ``result.output``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model_args, (str, bytes)) or not isinstance(model_args, Sequence):
        raise TypeError("model_args must be a sequence")
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence")
    if not specs:
        raise ValueError("specs must contain at least one intervention")
    if any(not isinstance(spec, InterventionSpec) for spec in specs):
        raise TypeError("every spec must be an InterventionSpec")
    if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping")
    if any(not isinstance(module, nn.Module) for module in frozen_modules):
        raise TypeError("frozen_modules must contain torch.nn.Module instances")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer")

    kwargs = dict(model_kwargs or {})
    optimized: list[OptimizedIntervention] = []
    artifacts: list[InterventionStageArtifact] = []
    model_passes = 0
    model_pass_budget = 1 + sum((spec.initial_value is None) + spec.max_steps for spec in specs)

    protected_modules = (model, *frozen_modules)
    input_tensors = _input_tensors((model_args, kwargs))
    with (
        _preserve_modules(protected_modules),
        _preserve_tensor_gradients(input_tensors),
        _preserve_rng(protected_modules, input_tensors) as accelerator_devices,
    ):
        if seed is not None:
            _seed_rng(seed, accelerator_devices)
        for spec in specs:
            initial = spec.initial_value
            if initial is None:
                with HookSession(model) as session:
                    _install_interventions(session, optimized)
                    captured = session.capture(spec.target, direction=spec.direction)
                    model(*model_args, **kwargs)
                model_passes += 1
                if (
                    captured.value is None
                ):  # pragma: no cover - target validation and execution normally guarantee this
                    raise RuntimeError(f"target {spec.target.module_path!r} was not reached")
                initial = captured.value
            else:
                initial = initial.detach().clone()

            value = initial.detach().clone().requires_grad_(True)
            optimizer = _build_optimizer(spec.optimizer, value)
            history: list[InterventionStepArtifact] = []
            best_loss = float("inf")
            stalled_steps = 0
            status: InterventionStatus = "budget_exhausted"

            for step in range(spec.max_steps):
                optimizer.zero_grad()
                with HookSession(model) as session:
                    _install_interventions(session, optimized)
                    session.replace(spec.target, value, direction=spec.direction)
                    output = model(*model_args, **kwargs)
                model_passes += 1
                objective = _objective_value(spec.objective(output, value))
                loss = objective.loss
                terms = dict(objective.terms)
                if spec.preservation_regularizer is not None:
                    if "preservation" in terms:
                        raise ValueError("objective terms reserve the name 'preservation'")
                    preservation = spec.preservation_regularizer(value, initial)
                    _validate_scalar_tensor(preservation, "preservation regularizer")
                    loss = loss + preservation
                    terms["preservation"] = preservation
                terms = {"loss": loss, **terms}
                numeric_terms = _numeric_terms(terms)
                history.append(InterventionStepArtifact(step=step, terms=numeric_terms))
                loss_value = numeric_terms["loss"]
                if not all(torch.isfinite(torch.tensor(item)) for item in numeric_terms.values()):
                    status = "non_finite"
                    break

                stopping = spec.early_stopping
                if stopping.objective_threshold is not None and loss_value <= stopping.objective_threshold:
                    status = "converged"
                    break
                if best_loss - loss_value > stopping.min_delta:
                    best_loss = loss_value
                    stalled_steps = 0
                else:
                    stalled_steps += 1
                    if stopping.patience is not None and stalled_steps >= stopping.patience:
                        status = "stalled"
                        break

                loss.backward()
                if value.grad is None or not torch.isfinite(value.grad).all():
                    status = "non_finite"
                    break
                last_finite_value = value.detach().clone()
                optimizer.step()
                if not torch.isfinite(value).all():
                    with torch.no_grad():
                        value.copy_(last_finite_value)
                    status = "non_finite"
                    break

            final_value = value.detach().clone()
            optimized.append(OptimizedIntervention(spec.target, final_value, spec.direction))
            artifacts.append(
                InterventionStageArtifact(
                    target=spec.target,
                    objective_name=spec.objective_name,
                    optimizer=spec.optimizer,
                    early_stopping=spec.early_stopping,
                    step_budget=spec.max_steps,
                    steps_completed=len(history),
                    model_pass_budget=(spec.initial_value is None) + spec.max_steps,
                    model_passes=1 + len(history) if spec.initial_value is None else len(history),
                    status=status,
                    history=tuple(history),
                    value_shape=tuple(final_value.shape),
                    value_dtype=str(final_value.dtype),
                    value_sha256=_tensor_sha256(final_value),
                )
            )

        with torch.no_grad(), HookSession(model) as session:
            _install_interventions(session, optimized)
            output = _materialize_output(model(*model_args, **kwargs))
        model_passes += 1

    return OptimizedInterventionResult(
        output=output,
        interventions=tuple(optimized),
        stages=tuple(artifacts),
        model_pass_budget=model_pass_budget,
        model_passes=model_passes,
    )


def optimize_intervention(
    model: nn.Module,
    model_args: Sequence[object],
    spec: InterventionSpec,
    **kwargs: object,
) -> OptimizedInterventionResult:
    """Optimize and apply one activation intervention."""

    return optimize_interventions(model, model_args, (spec,), **kwargs)


def _objective_value(value: Tensor | InterventionObjective) -> InterventionObjective:
    if isinstance(value, Tensor):
        result = InterventionObjective(value)
    elif isinstance(value, InterventionObjective):
        result = value
    else:
        raise TypeError("objective must return a tensor or InterventionObjective")
    _validate_scalar_tensor(result.loss, "objective loss")
    if not isinstance(result.terms, Mapping):
        raise TypeError("objective terms must be a mapping")
    if "loss" in result.terms:
        raise ValueError("objective terms reserve the name 'loss'")
    return result


def _validate_scalar_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.numel() != 1:
        raise ValueError(f"{name} must be scalar")


def _numeric_terms(terms: Mapping[str, Tensor | float | int]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for name, value in terms.items():
        if not isinstance(name, str) or not name:
            raise ValueError("objective term names must be non-empty strings")
        if isinstance(value, Tensor):
            _validate_scalar_tensor(value, f"objective term {name!r}")
            numeric[name] = float(value.detach().cpu())
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"objective term {name!r} must be numeric")
        else:
            numeric[name] = float(value)
    return numeric


def _build_optimizer(config: OptimizerConfig, value: Tensor) -> torch.optim.Optimizer:
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    return optimizers[config.name]([value], lr=float(config.learning_rate), **dict(config.kwargs))


def _install_interventions(session: HookSession, interventions: Sequence[OptimizedIntervention]) -> None:
    for intervention in interventions:
        session.replace(intervention.target, intervention.value, direction=intervention.direction)


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return sha256(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _accelerator_devices(modules: Sequence[nn.Module], tensors: Sequence[Tensor]) -> dict[str, list[int]]:
    devices: dict[str, set[int]] = {}
    values = (*tensors, *(item for module in modules for item in (*module.parameters(), *module.buffers())))
    for value in values:
        device_type = value.device.type
        if device_type not in {"cuda", "mps", "xpu"}:
            continue
        index = value.device.index if value.device.index is not None else 0  # pragma: no cover - accelerator-specific
        devices.setdefault(device_type, set()).add(index)  # pragma: no cover - accelerator-specific
    return {device_type: sorted(indices) for device_type, indices in devices.items()}


@contextmanager
def _preserve_rng(modules: Sequence[nn.Module], tensors: Sequence[Tensor]):
    accelerator_devices = _accelerator_devices(modules, tensors)
    with ExitStack() as stack:
        if accelerator_devices:  # pragma: no cover - requires accelerator hardware
            for device_type, devices in accelerator_devices.items():
                stack.enter_context(torch.random.fork_rng(devices=devices, device_type=device_type))
        else:
            stack.enter_context(torch.random.fork_rng(devices=[]))
        yield accelerator_devices


def _seed_rng(seed: int, accelerator_devices: Mapping[str, Sequence[int]]) -> None:
    torch.random.default_generator.manual_seed(seed)
    for device_type, devices in accelerator_devices.items():  # pragma: no cover - requires accelerator hardware
        device_module = getattr(torch, device_type)
        default_generators = getattr(device_module, "default_generators", None)
        if default_generators is not None:
            for device in devices:
                default_generators[device].manual_seed(seed)
        else:
            device_module.manual_seed(seed)


def _input_tensors(values: Sequence[object]) -> tuple[Tensor, ...]:
    tensors: list[Tensor] = []
    stack = list(values)
    seen: set[int] = set()
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Tensor):
            tensors.append(value)
        elif isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            stack.extend(value)
    return tuple(tensors)


@contextmanager
def _preserve_tensor_gradients(tensors: Sequence[Tensor]):
    gradients = tuple(
        (tensor, None if tensor.grad is None else tensor.grad.detach().clone()) for tensor in tensors if tensor.is_leaf
    )
    try:
        yield
    finally:
        for tensor, gradient in gradients:
            tensor.grad = gradient


def _materialize_output(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, TensorDictBase):
        return value.clone()
    if isinstance(value, MutableMapping):
        result = copy(value)
        for key, item in value.items():
            result[key] = _materialize_output(item)
        return result
    if isinstance(value, tuple):
        items = tuple(_materialize_output(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, list):
        return [_materialize_output(item) for item in value]
    return value


@contextmanager
def _preserve_modules(modules: Sequence[nn.Module]):
    states: list[_ModuleState] = []
    seen_parameters: set[int] = set()
    seen_buffers: set[int] = set()
    for module in modules:
        parameters = []
        for parameter in module.parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            gradient = None if parameter.grad is None else parameter.grad.detach().clone()
            parameters.append((parameter, parameter.detach().clone(), parameter.requires_grad, gradient))
            parameter.requires_grad_(False)
        buffers = []
        for buffer in module.buffers():
            if id(buffer) in seen_buffers:
                continue
            seen_buffers.add(id(buffer))
            buffers.append((buffer, buffer.detach().clone()))
        training = tuple((submodule, submodule.training) for submodule in module.modules())
        states.append(_ModuleState(training, tuple(parameters), tuple(buffers)))
    try:
        yield
    finally:
        with torch.no_grad():
            for state in states:
                for submodule, training in state.training:
                    submodule.training = training
                for parameter, value, requires_grad, gradient in state.parameters:
                    parameter.copy_(value)
                    parameter.requires_grad_(requires_grad)
                    parameter.grad = gradient
                for buffer, value in state.buffers:
                    buffer.copy_(value)


__all__ = [
    "EarlyStoppingConfig",
    "InterventionObjective",
    "InterventionSpec",
    "InterventionStageArtifact",
    "InterventionStatus",
    "InterventionStepArtifact",
    "OptimizedIntervention",
    "OptimizedInterventionResult",
    "OptimizerConfig",
    "optimize_intervention",
    "optimize_interventions",
]
