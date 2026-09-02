import pytest
import torch
from torch import nn

from tdhook.latent import ActivationCaching
from tdhook.runtime import CaptureSource, HookProgram, HookProgramBuilder, HookSpec, temporary_module_state
from tdhook.session import HookSession
from tdhook.targets import Target


def test_interactive_and_prepared_capture_share_one_program_model(default_test_model):
    inputs = torch.randn(2, 10)
    target = Target("linear2", "activation", -1, (0,))

    with HookSession(default_test_model) as session:
        session.capture(target)
        default_test_model(inputs)

    factory = ActivationCaching("linear2")
    with factory.prepare(default_test_model) as method:
        method(inputs)
    prepared_program = method.hooking_context.program

    assert prepared_program is not None
    assert [(spec.module_path, spec.operation, spec.direction, spec.prepend) for spec in session.program.hooks] == [
        (spec.module_path, spec.operation, spec.direction, spec.prepend) for spec in prepared_program.hooks
    ]


def test_program_builder_installs_and_removes_hooks_in_reverse_order():
    module = nn.Identity()
    events = []

    def first(_module, _args, output):
        events.append("first run")
        return output

    def second(_module, _args, output):
        events.append("second run")
        return output

    builder = HookProgramBuilder()
    builder.register(module, first, HookSpec("", "observe", "fwd"))
    builder.register(module, second, HookSpec("", "observe", "fwd"))
    builder.record(HookSpec("", "temporary state", None), lambda: events.append("state restored"))
    bound = builder.build()

    module(1)
    bound.remove()

    assert bound.program == HookProgram(
        (
            HookSpec("", "observe", "fwd"),
            HookSpec("", "observe", "fwd"),
            HookSpec("", "temporary state", None),
        )
    )
    assert events == ["first run", "second run", "state restored"]
    assert len(module._forward_hooks) == 0


def test_program_builder_rolls_back_partial_registration_and_preserves_first_error():
    events = []

    def first_cleanup():
        events.append("first")
        raise RuntimeError("first failure")

    def second_cleanup():
        events.append("second")
        raise RuntimeError("second failure")

    builder = HookProgramBuilder()
    builder.record(HookSpec("", "first", None), first_cleanup)
    builder.record(HookSpec("", "second", None), second_cleanup)

    with pytest.raises(RuntimeError, match="second failure"):
        builder.remove()
    assert events == ["second", "first"]
    builder.remove()
    assert events == ["second", "first"]


def test_program_builder_rejects_invalid_specs_and_reuse():
    with pytest.raises(TypeError, match="module_path"):
        HookSpec(None, "capture", "fwd")
    with pytest.raises(TypeError, match="operation"):
        HookSpec("", 1, "fwd")
    with pytest.raises(ValueError, match="operation"):
        HookSpec("", "", "fwd")
    with pytest.raises(ValueError, match="direction"):
        HookSpec("", "capture", "sideways")
    with pytest.raises(TypeError, match="prepend"):
        HookSpec("", "capture", "fwd", prepend=1)
    with pytest.raises(TypeError, match="target"):
        HookSpec("", "capture", "fwd", target=object())
    with pytest.raises(TypeError, match="source"):
        HookSpec("", "replace", "fwd", source=object())
    with pytest.raises(TypeError, match="hook_index"):
        CaptureSource(True, detach=True)
    with pytest.raises(ValueError, match="hook_index"):
        CaptureSource(-1, detach=True)
    with pytest.raises(TypeError, match="detach"):
        CaptureSource(0, detach=1)
    with pytest.raises(ValueError, match="earlier hook"):
        HookProgram((HookSpec("", "replace", "fwd", source=CaptureSource(0, detach=True)),))
    with pytest.raises(ValueError, match="capture hook"):
        HookProgram(
            (
                HookSpec("", "observe", "fwd"),
                HookSpec("", "replace", "fwd", source=CaptureSource(0, detach=True)),
            )
        )

    builder = HookProgramBuilder()
    with pytest.raises(ValueError, match="require a direction"):
        builder.register(nn.Identity(), lambda *_: None, HookSpec("", "capture", None))
    with pytest.raises(ValueError, match="must have the signature"):
        builder.register(nn.Identity(), lambda _module: None, HookSpec("", "capture", "fwd"))

    builder.build()
    with pytest.raises(RuntimeError, match="already built"):
        builder.record(HookSpec("", "capture", None))
    with pytest.raises(RuntimeError, match="already built"):
        builder.remove()

    builder = HookProgramBuilder()
    with pytest.raises(TypeError, match="cleanup"):
        builder.record(HookSpec("", "capture", None), cleanup=1)
    with pytest.raises(TypeError, match="cleanup"):
        builder.add_cleanup(1)

    builder = HookProgramBuilder()
    with pytest.raises(TypeError, match="module_path"):
        builder.mark_stopped(None)
    builder.mark_stopped("layer")
    assert builder.program.stopped_at == "layer"


def test_temporary_module_state_rejects_hook_specs():
    with pytest.raises(ValueError, match="directionless"):
        with temporary_module_state(nn.Identity(), None, HookSpec("", "state", "fwd")):
            pass


def test_program_builder_context_rolls_back_when_registration_fails():
    module = nn.Identity()
    with pytest.raises(RuntimeError, match="boom"):
        with HookProgramBuilder() as builder:
            builder.register(module, lambda _module, _args, output: output, HookSpec("", "capture", "fwd"))
            raise RuntimeError("boom")

    assert len(module._forward_hooks) == 0


def test_program_builder_registers_exact_paths_and_rolls_back_resolution_failure():
    root = nn.Module()
    root.layers = nn.Sequential(nn.Identity())

    with pytest.raises(ValueError, match="missing"):
        with HookProgramBuilder() as builder:
            builder.register_path(
                root,
                lambda _module, _args, output: output,
                HookSpec("[0]", "capture", "fwd"),
                relative_path="layers",
            )
            builder.register_path(
                root,
                lambda _module, _args, output: output,
                HookSpec("missing", "capture", "fwd"),
            )

    assert len(root.layers[0]._forward_hooks) == 0


def test_program_builder_rejects_nonmodules_and_warns_for_module_lists():
    root = nn.Module()
    root.value = object()
    root.layers = nn.ModuleList([nn.Identity()])
    builder = HookProgramBuilder()

    with pytest.raises(TypeError, match="torch.nn.Module"):
        builder.register_path(
            root,
            lambda _module, _args, output: output,
            HookSpec("value", "capture", "fwd"),
        )
    with pytest.warns(UserWarning, match="ModuleList"):
        builder.register_path(
            root,
            lambda _module, _args, output: output,
            HookSpec("layers", "capture", "fwd"),
        )

    builder.remove()
