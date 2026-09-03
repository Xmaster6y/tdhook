"""
Tests for the activation caching functionality.
"""

import pytest
import torch
from tensordict import MemoryMappedTensor, TensorDict
from tensordict.nn import TensorDictModule

from tdhook.latent.activation_caching import ActivationCaching, ActivationCachingModule
from tdhook.modules import get_best_device
from tdhook.runtime import HookProgram, HookSpec
from tdhook.targets import Target


class TestActivationCaching:
    """Test the ActivationCaching class."""

    def test_activation_caching_context_creation(self, default_test_model):
        """Test creating a ActivationCaching."""

        context = ActivationCaching(r"td_module\.module\.linear2", relative=False)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "td_module.module.linear2" in hooked_module.hooking_context.cache

    def test_activation_caching_context_creation_relative(self, default_test_model):
        """Test creating a ActivationCaching with relative naming."""

        context = ActivationCaching("linear2", relative=True)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(inputs)
        assert "linear2" in hooked_module.hooking_context.cache
        assert hooked_module.hooking_context.program == HookProgram((HookSpec("linear2", "capture", "fwd"),))

    def test_tensordict_execution_publishes_a_native_cache_output(self, default_test_model):
        context = ActivationCaching("linear2", cache_key=("activations", "cache"))
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with context.prepare(default_test_model) as hooked_module:
            result = hooked_module(data)

        assert hooked_module.in_keys == ["input"]
        assert hooked_module.out_keys == ["output", ("activations", "cache")]
        assert result["activations", "cache"]["linear2"].shape == (2, 20)

    def test_target_selection_is_cached_and_reported(self, default_test_model):
        target = Target("linear2", "activation", -1, (0, 2))

        with ActivationCaching(target).prepare(default_test_model) as hooked_module:
            result = hooked_module(TensorDict({"input": torch.randn(2, 10)}, batch_size=[2]))

        assert result["cache", "linear2"].shape == (2, 2)
        assert hooked_module.hooking_context.program == HookProgram(
            (HookSpec("linear2", "capture", "fwd", target=target),)
        )

    def test_target_occurrence_selects_one_repeated_module_call(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Identity()

            def forward(self, value):
                return self.shared(value + 1) + self.shared(value + 2)

        target = Target("shared", "activation", -1, (0,), occurrence=0)
        data = TensorDict({"input": torch.tensor([[1.0, 2.0]])}, batch_size=[1])

        with ActivationCaching(target).prepare(Model()) as prepared:
            result = prepared(data)

        torch.testing.assert_close(result["cache", "shared"], torch.tensor([[2.0]]))
        assert prepared.hooking_context.occurrence_evidence[0].target_path == "shared"
        assert prepared.hooking_context.occurrence_evidence[0].selected_indices == (0,)
        assert prepared.hooking_context.occurrence_evidence[0].observed_indices == (0, 1)

    def test_target_uses_the_shared_module_path_grammar(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.ReLU()])

            def forward(self, value):
                for layer in self.layers:
                    value = layer(value)
                return value

        target = Target("layers[-1]", "activation", -1, (0,))
        with ActivationCaching(target).prepare(Model()) as hooked_module:
            result = hooked_module(TensorDict({"input": torch.randn(2, 3)}, batch_size=[2]))

        assert result["cache", "layers[-1]"].shape == (2, 1)
        assert hooked_module.hooking_context.program == HookProgram(
            (HookSpec("layers[-1]", "capture", "fwd", target=target),)
        )

    def test_target_caching_rejects_ambiguous_direction_and_path_modes(self):
        activation = Target("linear", "activation", -1, (0,))
        gradient = Target("linear", "gradient", -1, (0,))

        with pytest.raises(ValueError, match="activation Target"):
            ActivationCaching(gradient)
        with pytest.raises(ValueError, match="relative to the caller-owned model"):
            ActivationCaching(activation, relative=False)
        with pytest.raises(ValueError, match="forward direction"):
            ActivationCaching(activation, directions=["bwd"])

    def test_published_cache_is_not_cleared_by_a_later_execution(self, default_test_model):
        context = ActivationCaching("linear2")
        first = TensorDict({"input": torch.ones(2, 10)}, batch_size=[2])
        second = TensorDict({"input": torch.zeros(2, 10)}, batch_size=[2])

        with context.prepare(default_test_model) as hooked_module:
            hooked_module(first)
        first_values = first["cache"]["linear2"].clone()
        with context.prepare(default_test_model) as hooked_module:
            hooked_module(second)

        torch.testing.assert_close(first["cache"]["linear2"], first_values)

    def test_cache_output_can_be_disabled_for_context_only_use(self, default_test_model):
        context = ActivationCaching("linear2", cache_key=None)
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with context.prepare(default_test_model) as hooked_module:
            hooked_module(data)

        assert hooked_module.out_keys == ["output"]
        assert "cache" not in data

    def test_cache_output_key_is_validated_and_cannot_replace_a_model_output(self):
        with pytest.raises(TypeError, match="cache_key"):
            ActivationCaching("", cache_key=object())

        model = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=["cache"])
        with pytest.raises(ValueError, match="collides"):
            with ActivationCaching("", cache_key="cache").prepare(model):
                pass

        nested_output = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=[("result", "value")])
        for cache_key in ("result", ("result", "value", "cache")):
            with pytest.raises(ValueError, match="collides"):
                with ActivationCaching("", cache_key=cache_key).prepare(nested_output):
                    pass

    def test_cache_publication_requires_context_and_key_pattern_is_mutable(self):
        td_module = TensorDictModule(torch.nn.Identity(), in_keys=["input"], out_keys=["output"])
        prepared = ActivationCachingModule(td_module, cache_key="cache")
        with pytest.raises(RuntimeError, match="prepared hooking context"):
            prepared.finalize_tensordict(TensorDict({"output": torch.ones(2, 3)}, batch_size=[2]))

        method = ActivationCaching("linear1")
        assert method.key_pattern == "linear1"
        method.key_pattern = "linear2"
        assert method.key_pattern == "linear2"

    def test_different_device_cache(self, default_test_model):
        """Test creating a ActivationCaching with cache on a different device."""

        device = get_best_device()
        cache = TensorDict(device=device)
        context = ActivationCaching("linear2", relative=True, cache=cache)

        inputs = torch.randn(2, 10)
        with context.prepare(default_test_model) as hooked_module:
            output = hooked_module(inputs)
        assert output.device.type == "cpu"
        assert hooked_module.hooking_context.cache["linear2"].device.type == device.type

    @pytest.mark.parametrize("run_in_workflow", [False, True])
    def test_preallocated_memmap_cache_is_published_without_materializing(
        self, default_test_model, tmp_path, run_in_workflow
    ):
        path = tmp_path / "activation-cache"
        cache = TensorDict({("fwd", "linear2"): torch.zeros(2, 20)}, batch_size=[]).memmap(path)
        context = ActivationCaching(
            "linear2",
            cache=cache,
            clear_cache=False,
            use_nested_keys=True,
            cache_key=("artifacts", "activations"),
        )
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        if run_in_workflow:
            from tdhook.workflow import Workflow

            result = Workflow(context)(default_test_model, data)
        else:
            with context.prepare(default_test_model) as hooked_module:
                result = hooked_module(data)

        published = result["artifacts", "activations", "fwd", "linear2"]
        assert isinstance(published, MemoryMappedTensor)
        assert published.data_ptr() == cache["fwd", "linear2"].data_ptr()
        assert torch.count_nonzero(published)
        reloaded = TensorDict.load_memmap(path)
        torch.testing.assert_close(reloaded["fwd", "linear2"], published)

    def test_memmap_cache_requires_explicit_lifetime_and_preallocated_keys(self, default_test_model, tmp_path):
        path = tmp_path / "activation-cache"
        cache = TensorDict({"other": torch.zeros(2, 20)}, batch_size=[]).memmap(path)

        with pytest.raises(ValueError, match="clear_cache=False"):
            ActivationCaching("linear2", cache=cache)

        context = ActivationCaching("linear2", cache=cache, clear_cache=False)
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])
        with (
            context.prepare(default_test_model) as hooked_module,
            pytest.raises(RuntimeError, match="preallocated entry.*linear2"),
        ):
            hooked_module(data)

    def test_cache_locked_after_configuration_is_rejected_at_context_entry(self, default_test_model, tmp_path):
        cache = TensorDict({"linear2": torch.zeros(2, 20)}, batch_size=[])
        context = ActivationCaching("linear2", cache=cache)
        cache.memmap_(tmp_path / "activation-cache")

        with pytest.raises(ValueError, match="clear_cache=False"):
            with context.prepare(default_test_model):
                pass

    def test_locked_cache_reports_missing_nested_parent(self, default_test_model, tmp_path):
        cache = TensorDict({"other": torch.zeros(2, 20)}, batch_size=[]).memmap(tmp_path / "activation-cache")
        context = ActivationCaching(
            "linear2",
            cache=cache,
            clear_cache=False,
            use_nested_keys=True,
        )
        data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])

        with (
            context.prepare(default_test_model) as hooked_module,
            pytest.raises(RuntimeError, match=r"preallocated entry.*fwd.*linear2"),
        ):
            hooked_module(data)
