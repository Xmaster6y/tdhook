from __future__ import annotations

import torch
from tensordict import TensorDict

from tdhook.weights.adapters import Adapters


class _DoubleAdapter(torch.nn.Module):
    def forward(self, x, **_):
        return x * 2


class TestAdapters:
    def test_adapter_modifies_output(self, default_test_model):
        data = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
        baseline_out = default_test_model(data["input"]).detach().clone()

        adapters = {"linear2": (_DoubleAdapter(), "linear2", "linear2")}
        ctx_factory = Adapters(adapters=adapters)

        with ctx_factory.prepare(default_test_model) as hooked:
            patched_data = hooked(data.clone())
            patched_out = patched_data["output"]
            assert not torch.allclose(baseline_out, patched_out)

        restored_out = default_test_model(data["input"])
        assert torch.allclose(baseline_out, restored_out)

    def test_adapter_crosslayer(self, default_test_model):
        data = TensorDict({"input": torch.randn(4, 10)}, batch_size=4)
        baseline_out = default_test_model(data["input"]).detach().clone()

        adapters = {"linear2": (_DoubleAdapter(), "linear1", "linear2")}
        ctx_factory = Adapters(adapters=adapters)

        with ctx_factory.prepare(default_test_model) as hooked:
            patched_data = hooked(data.clone())
            patched_out = patched_data["output"]
            assert not torch.allclose(baseline_out, patched_out)

        restored_out = default_test_model(data["input"])
        assert torch.allclose(baseline_out, restored_out)
