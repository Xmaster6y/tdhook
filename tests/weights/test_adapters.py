from __future__ import annotations

import torch
import pytest
from tensordict import TensorDict

from tdhook.runtime import CaptureSource, HookProgram, HookSpec
from tdhook.weights.adapters import Adapters


class _DoubleAdapter(torch.nn.Module):
    def forward(self, x, **_):
        return x * 2


class TestAdapters:
    def _test_adapter_behavior(self, default_test_model, adapter_source, adapter_target):
        data = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
        baseline_out = default_test_model(data["input"]).detach().clone()

        adapters = {"linear2": (_DoubleAdapter(), adapter_source, adapter_target)}
        ctx_factory = Adapters(adapters=adapters)

        with ctx_factory.prepare(default_test_model) as hooked:
            patched_data = hooked(data.clone())
            patched_out = patched_data["output"]
            assert not torch.allclose(baseline_out, patched_out)

        expected_specs = []
        if adapter_source != adapter_target:
            expected_specs.append(HookSpec(adapter_source, "capture", "fwd"))
        expected_specs.append(
            HookSpec(
                adapter_target,
                "replace",
                "fwd",
                source=CaptureSource(0, detach=False) if adapter_source != adapter_target else None,
            )
        )
        assert hooked.hooking_context.program == HookProgram(tuple(expected_specs))

        restored_out = default_test_model(data["input"])
        assert torch.allclose(baseline_out, restored_out)

    def test_adapter_modifies_output(self, default_test_model):
        self._test_adapter_behavior(default_test_model, "linear2", "linear2")

    def test_adapter_crosslayer(self, default_test_model):
        self._test_adapter_behavior(default_test_model, "linear1", "linear2")

    def test_adapter_registration_failure_removes_source_hook(self, default_test_model):
        adapters = {"broken": (_DoubleAdapter(), "linear1", "missing")}

        with pytest.raises(ValueError, match="missing"):
            with Adapters(adapters).prepare(default_test_model):
                pass

        assert not default_test_model.linear1._forward_hooks

    def test_crosslayer_adapter_rejects_a_stale_capture(self):
        class ConditionalSource(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.source = torch.nn.Identity()
                self.destination = torch.nn.Identity()

            def forward(self, value):
                if value.sum() > 0:
                    value = self.source(value)
                return self.destination(value)

        model = ConditionalSource()
        method = Adapters({"conditional": (_DoubleAdapter(), "source", "destination")})

        with method.prepare(model) as prepared:
            prepared(TensorDict({"input": torch.ones(1, 2)}, batch_size=[1]))
            with pytest.raises(RuntimeError, match="fresh source capture"):
                prepared(TensorDict({"input": -torch.ones(1, 2)}, batch_size=[1]))
