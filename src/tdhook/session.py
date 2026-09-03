"""Interactive capture and intervention through an explicit hook lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Self
import weakref

import torch
from torch import Tensor, nn

from tdhook.hooks import EarlyStoppingException, HookDirection, register_hook_to_module
from tdhook.runtime import CaptureSource, HookProgram, HookProgramBuilder, HookSpec, TargetOccurrenceEvidence
from tdhook.targets import OutputPathComponent, Target


HookOperation = Literal["capture", "replace", "stop"]


@dataclass
class CapturedTarget:
    """Mutable result populated whenever a capture hook observes its target.

    ``value`` retains the most recent observation for compatibility with a
    single model execution. ``values`` preserves every observation in call
    order, so repeated executions remain distinguishable.
    """

    value: Tensor | None = None
    values: list[Tensor] = field(default_factory=list)
    _target: Target | None = field(default=None, repr=False, compare=False)
    _session_token: object | None = field(default=None, repr=False, compare=False)
    _hook_index: int | None = field(default=None, repr=False, compare=False)
    _detach: bool = field(default=True, repr=False, compare=False)
    _execution: int | None = field(default=None, repr=False, compare=False)

    def _record(self, value: Tensor, execution: int | None = None) -> None:
        self.value = value
        self.values.append(value)
        self._execution = execution


@dataclass
class EarlyStopResult:
    """Outcome populated if execution reaches a session stop location.

    ``output`` is the exact output of the module where execution stopped. It
    is not a synthesized output for the model components that did not run.
    ``reached`` distinguishes an unreached stop from a module that returned
    ``None``.
    """

    reached: bool = False
    output: object | None = None


class HookSession:
    """Own an ordered set of temporary capture and replacement operations.

    The session keeps only a weak reference to the caller-owned model. Hooks
    and parameter replacements are active inside the context and are removed
    or restored in reverse registration order on every exit path.

    When ``model`` is a ``DistributedDataParallel`` instance, the session is
    rank-local: it neither communicates nor aggregates results. Create and use
    one session per rank. Targets keep their underlying model paths because
    TDHook resolves DDP's ``module`` wrapper transparently.
    """

    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("HookSession requires a torch.nn.Module")
        self._model_ref = weakref.ref(model)
        self._builder: HookProgramBuilder | None = None
        self._program = HookProgram()
        self._stop_exception: EarlyStoppingException | None = None
        self._session_token: object | None = None
        self._executions = {"activation": 0, "gradient": 0}
        self._tracked_executions: set[str] = set()
        self._occurrence_evidence: tuple[TargetOccurrenceEvidence, ...] = ()

    @property
    def program(self) -> HookProgram:
        """Return an immutable description of the operations installed this run."""

        return self._builder.program if self._builder is not None else self._program

    @property
    def occurrence_evidence(self) -> tuple[TargetOccurrenceEvidence, ...]:
        """Return validated occurrence evidence from completed root model passes."""

        if self._builder is not None:
            return self._builder.occurrence_evidence
        return self._occurrence_evidence

    def __enter__(self) -> Self:
        if self._builder is not None:
            raise RuntimeError("Cannot enter a HookSession twice")
        self._model()
        self._program = HookProgram()
        self._stop_exception = None
        self._session_token = object()
        self._executions = {"activation": 0, "gradient": 0}
        self._tracked_executions.clear()
        self._occurrence_evidence = ()
        self._builder = HookProgramBuilder()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        builder = self._builder
        if builder is None:
            raise RuntimeError("Cannot exit a HookSession that is not active")
        self._builder = None
        bound = builder.build()
        self._program = bound.program
        suppress_stop = exc_value is self._stop_exception
        try:
            bound.remove()
        finally:
            self._occurrence_evidence = bound.occurrence_evidence
            self._stop_exception = None
            self._session_token = None
        return suppress_stop

    def capture(
        self,
        target: Target,
        *,
        direction: HookDirection | None = None,
        prepend: bool = False,
        detach: bool = True,
    ) -> CapturedTarget:
        """Capture ``target`` while this session is active.

        ``direction`` defaults to ``"fwd"`` for activation targets and
        ``"bwd_pre"`` for gradient targets. Forward inputs use ``"fwd_pre"``;
        use ``"fwd_pre_kwargs"`` to expose ``(args, kwargs)`` as the hook value.
        Gradient inputs and outputs use ``"bwd"`` and ``"bwd_pre"``
        respectively. By default captures are detached clones. Set
        ``detach=False`` when a later attribution objective must backpropagate
        from the captured activation; the result then retains its autograd
        history and is only valid for the lifetime of the surrounding graph.
        ``Target.occurrences`` captures one or more ordered, zero-based module
        calls in each root-model execution. Validated observations are exposed through
        :attr:`occurrence_evidence`.
        """

        model, builder = self._active_state()
        module = target.validate(model)
        self._validate_stop_compatibility(builder, target)
        if not isinstance(detach, bool):
            raise TypeError("detach must be a bool")
        direction = self._direction(target, direction)
        spec = HookSpec(target.module_path, "capture", direction, prepend, target)
        hook_index = len(builder.program.hooks)
        captured = CapturedTarget(
            _target=target,
            _session_token=self._session_token,
            _hook_index=hook_index,
            _detach=detach,
        )

        if target.kind == "parameter":
            parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
            captured._record(self._captured_value(target._select(parameter), detach=detach))
            builder.record(spec)
        else:

            def forward_hook(_module: nn.Module, _args: tuple[object, ...], value: object):
                captured._record(
                    self._captured_value(target.select_output(value), detach=detach),
                    self._executions["activation"],
                )

            def forward_pre_hook(_module: nn.Module, args: tuple[object, ...]):
                captured._record(
                    self._captured_value(target.select_output(args), detach=detach),
                    self._executions["activation"],
                )

            def forward_pre_kwargs_hook(
                _module: nn.Module,
                args: tuple[object, ...],
                kwargs: dict[str, object],
            ):
                captured._record(
                    self._captured_value(target.select_output((args, kwargs)), detach=detach),
                    self._executions["activation"],
                )

            def backward_hook(
                _module: nn.Module,
                grad_input: tuple[Tensor | None, ...],
                _grad_output: tuple[Tensor | None, ...],
            ):
                captured._record(
                    self._captured_value(target.select_output(grad_input), detach=detach),
                    self._executions["gradient"],
                )

            def backward_pre_hook(_module: nn.Module, values: tuple[Tensor | None, ...]):
                captured._record(
                    self._captured_value(target.select_output(values), detach=detach),
                    self._executions["gradient"],
                )

            hooks = {
                "fwd": forward_hook,
                "fwd_pre": forward_pre_hook,
                "fwd_pre_kwargs": forward_pre_kwargs_hook,
                "bwd": backward_hook,
                "bwd_pre": backward_pre_hook,
            }
            hook = hooks.get(direction)
            if hook is None:  # pragma: no cover - guarded by _direction
                raise RuntimeError(f"unsupported capture direction: {direction!r}")
            builder.register_target(model, hook, spec)

        return captured

    @staticmethod
    def _captured_value(value: Tensor, *, detach: bool) -> Tensor:
        return value.detach().clone() if detach else value

    def replace(
        self,
        target: Target,
        value: Tensor | float | int | CapturedTarget,
        *,
        direction: HookDirection | None = None,
        prepend: bool = False,
        transform: Callable[[Tensor], Tensor | float | int] | None = None,
    ) -> None:
        """Replace ``target`` until the session exits.

        Hook directions have the same target semantics as :meth:`capture`.
        Pass a :class:`CapturedTarget` from this session to route each newly
        observed live value to a later compatible target in the same model
        execution. ``transform`` is applied to that value immediately before
        replacement. Whether the routed value retains its graph is controlled
        by the source capture's ``detach`` argument. ``Target.occurrences``
        selects one or more ordered, zero-based calls per root-model execution.
        """

        model, builder = self._active_state()
        module = target.validate(model)
        self._validate_stop_compatibility(builder, target)
        direction = self._direction(target, direction)
        source = self._validate_live_source(target, value, transform)
        source_spec = None
        if source is not None:
            self._ensure_execution_tracking(model, builder, target.kind)
            source_spec = CaptureSource(source._hook_index, source._detach)  # type: ignore[arg-type]
        spec = HookSpec(target.module_path, "replace", direction, prepend, target, source_spec)

        if target.kind == "parameter":
            parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
            original = target._select(parameter).detach().clone()
            try:
                with torch.no_grad():
                    target._assign(parameter, value)
            except BaseException:
                self._restore_parameter(target, parameter, original)
                raise
            builder.record(spec, lambda: self._restore_live_parameter(target, original))
        else:

            def replacement_value() -> Tensor | float | int:
                if source is None:
                    return value  # type: ignore[return-value]
                execution = self._executions[target.kind]
                if execution == 0 or source.value is None or source._execution != execution:
                    raise RuntimeError("live replacement reached its target before a fresh source capture")
                replacement = source.value
                return transform(replacement) if transform is not None else replacement

            def forward_hook(_module: nn.Module, _args: tuple[object, ...], output: object):
                return target.replace_output(output, replacement_value())

            def forward_pre_hook(_module: nn.Module, args: tuple[object, ...]):
                return target.replace_output(args, replacement_value())

            def forward_pre_kwargs_hook(
                _module: nn.Module,
                args: tuple[object, ...],
                kwargs: dict[str, object],
            ):
                return target.replace_output((args, kwargs), replacement_value())

            def backward_hook(
                _module: nn.Module,
                grad_input: tuple[Tensor | None, ...],
                _grad_output: tuple[Tensor | None, ...],
            ):
                return target.replace_output(grad_input, replacement_value())

            def backward_pre_hook(_module: nn.Module, values: tuple[Tensor | None, ...]):
                return target.replace_output(values, replacement_value())

            hooks = {
                "fwd": forward_hook,
                "fwd_pre": forward_pre_hook,
                "fwd_pre_kwargs": forward_pre_kwargs_hook,
                "bwd": backward_hook,
                "bwd_pre": backward_pre_hook,
            }
            hook = hooks.get(direction)
            if hook is None:  # pragma: no cover - guarded by _direction
                raise RuntimeError(f"unsupported replacement direction: {direction!r}")
            builder.register_target(model, hook, spec)

    def _validate_live_source(
        self,
        target: Target,
        value: Tensor | float | int | CapturedTarget,
        transform: Callable[[Tensor], Tensor | float | int] | None,
    ) -> CapturedTarget | None:
        if not isinstance(value, CapturedTarget):
            if transform is not None:
                raise ValueError("transform is only valid for a live CapturedTarget replacement")
            return None
        if value._session_token is None or value._session_token is not self._session_token:
            raise ValueError("live replacements require a capture from the same active HookSession")
        if value._target is None or value._hook_index is None:
            raise ValueError("live replacements require a capture created by HookSession.capture")
        if target.kind == "parameter" or value._target.kind not in {"activation", "gradient"}:
            raise ValueError("live replacements support only activation or gradient targets")
        if value._target.kind != target.kind:
            raise ValueError("live capture and replacement targets must have the same kind")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable")
        return value

    def _ensure_execution_tracking(
        self,
        model: nn.Module,
        builder: HookProgramBuilder,
        kind: str,
    ) -> None:
        if kind in self._tracked_executions:
            return
        direction: HookDirection = "fwd_pre" if kind == "activation" else "bwd_pre"

        def begin_execution(_module: nn.Module, _values: object) -> None:
            self._executions[kind] += 1

        handle = register_hook_to_module(model, begin_execution, direction, prepend=True)
        builder.add_cleanup(handle.remove)
        self._tracked_executions.add(kind)

    def stop(self, module_path: str, *, prepend: bool = False) -> EarlyStopResult:
        """Stop forward execution after ``module_path`` produces its output.

        Reaching the location exits the active session context without
        exposing TDHook's internal control-flow exception. Earlier captures
        and this module's exact output remain available through their result
        objects; no final model output is synthesized. The partial output
        preserves its autograd history, but session-managed gradient capture
        and replacement cannot be combined with early stopping because those
        hooks are cleaned up when the context exits.
        """

        model, builder = self._active_state()
        if not isinstance(module_path, str):
            raise TypeError("module_path must be a string")
        if any(spec.direction in {"bwd", "bwd_pre"} for spec in builder.program.hooks):
            raise ValueError("HookSession early stopping cannot be combined with gradient operations")
        result = EarlyStopResult()
        spec = HookSpec(module_path, "stop", "fwd", prepend)

        def stopping_hook(_module: nn.Module, _args: tuple[object, ...], output: object):
            result.reached = True
            result.output = output
            builder.mark_stopped(module_path)
            exception = EarlyStoppingException(module_path)
            self._stop_exception = exception
            raise exception

        builder.register_path(model, stopping_hook, spec)
        return result

    @staticmethod
    def _validate_stop_compatibility(builder: HookProgramBuilder, target: Target) -> None:
        if target.kind == "gradient" and any(spec.operation == "stop" for spec in builder.program.hooks):
            raise ValueError("HookSession early stopping cannot be combined with gradient operations")

    def _active_state(self) -> tuple[nn.Module, HookProgramBuilder]:
        if self._builder is None:
            raise RuntimeError("HookSession operations require an active context")
        return self._model(), self._builder

    def _model(self) -> nn.Module:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("The model bound to this HookSession no longer exists")
        return model

    @staticmethod
    def _direction(target: Target, direction: HookDirection | None = None) -> HookDirection | None:
        allowed: dict[str, set[HookDirection | None]] = {
            "activation": {"fwd", "fwd_pre", "fwd_pre_kwargs"},
            "gradient": {"bwd", "bwd_pre"},
            "parameter": {None},
        }
        if direction is not None:
            if direction not in allowed[target.kind]:
                raise ValueError(f"direction {direction!r} is not valid for a {target.kind} target")
            return direction
        if target.kind == "activation":
            return "fwd"
        if target.kind == "gradient":
            return "bwd_pre"
        return None

    @staticmethod
    def _restore_parameter(target: Target, parameter: Tensor, original: Tensor) -> None:
        with torch.no_grad():
            target._assign(parameter, original)

    def _restore_live_parameter(self, target: Target, original: Tensor) -> None:
        module = target.validate(self._model())
        parameter = module.get_parameter(target.parameter)  # type: ignore[arg-type]
        self._restore_parameter(target, parameter, original)

    @staticmethod
    def _resolved_output_path(value: object, path: tuple[OutputPathComponent, ...]) -> tuple[OutputPathComponent, ...]:
        return Target._resolved_output_path(value, path)

    @classmethod
    def _hook_tensor(cls, value: object, path: tuple[OutputPathComponent, ...] = ()) -> Tensor:
        return Target._output_tensor(value, path)

    @classmethod
    def _replace_hook_tensor(
        cls,
        value: object,
        path: tuple[OutputPathComponent, ...],
        replacement: Tensor,
    ) -> object:
        return Target._replace_output_tensor(value, path, replacement)


__all__ = [
    "CapturedTarget",
    "CaptureSource",
    "EarlyStopResult",
    "HookOperation",
    "HookProgram",
    "HookSession",
    "HookSpec",
]
