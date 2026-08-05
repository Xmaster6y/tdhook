"""TensorDict-native concept selection and conditioned attribution."""

from __future__ import annotations

from contextvars import ContextVar
from copy import copy
from typing import Literal
from weakref import WeakKeyDictionary

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from tdhook._types import is_nested_key, join_keys
from tdhook.attribution import LRP
from tdhook.contexts import HookingContextFactory
from tdhook.execution import ExecutionSpec
from tdhook.modules import HookedModule
from tdhook.runtime import BoundHookProgram


class ConceptSelection(TensorDictModuleBase):
    """Select one channel from labelled concept-example relevances."""

    def __init__(
        self,
        relevance_key: NestedKey,
        *,
        labels_key: NestedKey = "concept_labels",
        out_key: NestedKey = ("metrics", "concept_selection"),
        direction: Literal["positive", "negative"] = "positive",
    ) -> None:
        super().__init__()
        if direction not in {"positive", "negative"}:
            raise ValueError("direction must be 'positive' or 'negative'")
        if not all(is_nested_key(key) for key in (relevance_key, labels_key, out_key)):
            raise TypeError("Concept selection keys must be TensorDict nested keys")
        self.relevance_key = relevance_key
        self.labels_key = labels_key
        self.out_key = out_key
        self.direction = direction
        self.in_keys = [relevance_key, labels_key]
        self.out_keys = [out_key]

    def forward(self, artifacts: TensorDictBase) -> TensorDictBase:
        relevances = artifacts.get(self.relevance_key)
        labels = artifacts.get(self.labels_key)
        if not isinstance(relevances, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Concept selection requires tensor relevances and labels")
        batch_dims = artifacts.batch_dims
        if relevances.ndim < batch_dims + 1:
            raise ValueError("Concept relevances must have TensorDict batch and channel dimensions")
        if relevances.shape[batch_dims] == 0:
            raise ValueError("Concept relevances must have a non-empty channel dimension")
        if tuple(labels.shape) != tuple(artifacts.batch_size):
            raise ValueError("Concept labels must match the TensorDict batch shape")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Concept labels must be binary (0 or 1)")
        positive = labels.reshape(-1).to(dtype=torch.bool, device=relevances.device)
        if not positive.any() or positive.all():
            raise ValueError("Concept examples must contain both positive and negative labels")
        values = relevances.reshape(labels.numel(), *relevances.shape[batch_dims:])
        # RelMax ranks the magnitude of signed relevance summed over spatial
        # dimensions; applying abs first would remove meaningful cancellation.
        if values.ndim > 2:
            values = values.sum(dim=tuple(range(2, values.ndim)))
        values = values.abs()
        positive_mean = values[positive].mean(dim=0)
        negative_mean = values[~positive].mean(dim=0)
        scores = positive_mean - negative_mean
        channel = scores.argmax() if self.direction == "positive" else scores.argmin()
        direction = 1 if self.direction == "positive" else -1
        batch_size = artifacts.batch_size
        batch_shape = tuple(batch_size)
        artifacts.set(
            self.out_key,
            TensorDict(
                {
                    "positive_mean": positive_mean.expand(*batch_shape, *positive_mean.shape),
                    "negative_mean": negative_mean.expand(*batch_shape, *negative_mean.shape),
                    "scores": scores.expand(*batch_shape, *scores.shape),
                    "channel": channel.to(dtype=torch.long).expand(*batch_shape),
                    "direction": torch.tensor(direction, device=scores.device).expand(*batch_shape),
                    "score": scores[channel].expand(*batch_shape),
                },
                batch_size=batch_size,
            ),
        )
        return artifacts


def concept_channel_gradient_callback(channel: int, direction: int, *, channel_axis: int = -1):
    """Build the LRP output-gradient callback for a selected feature channel."""
    if channel < 0:
        raise ValueError("Concept channel must be non-negative")
    if direction not in {-1, 1}:
        raise ValueError("Concept direction must be either -1 or 1")

    def callback(grad_output, **kwargs):
        gradient = grad_output[0]
        if gradient.ndim < 2:
            raise ValueError(f"Concept channel {channel} is invalid for gradient shape {tuple(gradient.shape)}")
        axis = channel_axis % gradient.ndim
        if channel >= gradient.shape[axis]:
            raise ValueError(f"Concept channel {channel} is invalid for gradient shape {tuple(gradient.shape)}")
        mask = torch.zeros_like(gradient)
        mask.select(axis, channel).fill_(direction)
        return (gradient * mask,)

    return callback


def _uniform_selection(selection: TensorDictBase) -> tuple[int, int]:
    if "channel" not in selection or "direction" not in selection:
        raise ValueError("Concept selection must provide 'channel' and 'direction'")
    channel = selection.get("channel")
    direction = selection.get("direction")
    if not isinstance(channel, torch.Tensor) or not isinstance(direction, torch.Tensor):
        raise TypeError("Concept selection channel and direction must be tensors")
    if channel.numel() == 0 or direction.numel() == 0:
        raise ValueError("Concept selection channel and direction cannot be empty")
    if (
        channel.dtype == torch.bool
        or direction.dtype == torch.bool
        or channel.is_floating_point()
        or direction.is_floating_point()
    ):
        raise TypeError("Concept selection channel and direction must use integer tensor dtypes")
    if not torch.equal(channel, channel.flatten()[0].expand_as(channel)) or not torch.equal(
        direction, direction.flatten()[0].expand_as(direction)
    ):
        raise ValueError("Concept selection must be identical for every example in the batch")
    selected_channel = int(channel.flatten()[0].item())
    selected_direction = int(direction.flatten()[0].item())
    if selected_channel < 0:
        raise ValueError("Concept selection channel must be non-negative")
    if selected_direction not in {-1, 1}:
        raise ValueError("Concept selection direction must be either -1 or 1")
    return selected_channel, selected_direction


class ChannelConditionedLRP(HookingContextFactory):
    """Run LRP using a channel selection read from the execution TensorDict.

    The configured method has no side cache shared between passes. Its prepared
    TensorDict module reads ``selection_key`` before the model call; the bound
    backward hook then uses that value for this execution only.
    """

    def __init__(
        self,
        base: LRP,
        *,
        condition_module: str,
        gradient_channel_axis: int = -1,
        selection_key: NestedKey = ("metrics", "concept_selection"),
        attribution_key: NestedKey = ("attributions", "conditioned"),
    ) -> None:
        super().__init__()
        if not isinstance(base, LRP):
            raise TypeError("base must be an LRP method")
        if not condition_module:
            raise ValueError("condition_module must be non-empty")
        if not is_nested_key(selection_key) or not is_nested_key(attribution_key):
            raise TypeError("selection_key and attribution_key must be TensorDict nested keys")
        if condition_module in base._output_grad_callbacks:
            raise ValueError(f"LRP method already has a gradient callback for {condition_module!r}")
        self.base = base
        self.condition_module = condition_module
        self.gradient_channel_axis = gradient_channel_axis
        self.selection_key = selection_key
        self.attribution_key = attribution_key
        self._prepared: WeakKeyDictionary[TensorDictModuleBase, LRP] = WeakKeyDictionary()
        self._hooked_module_kwargs = dict(base._hooked_module_kwargs)

    @property
    def execution_spec(self) -> ExecutionSpec:
        return self.base.execution_spec

    def _prepare_module(self, module, in_keys, out_keys, extra_relative_path):
        bound = copy(self.base)
        bound._additional_init_keys = [
            *bound._additional_init_keys,
            join_keys(self.selection_key, "channel"),
            join_keys(self.selection_key, "direction"),
        ]
        bound._attr_key = self.attribution_key
        original_init = bound._init_attr_inputs
        selected: ContextVar[tuple[int, int] | None] = ContextVar(
            f"conditioned_lrp_selection_{id(bound)}",
            default=None,
        )

        def initialise(inputs: TensorDict, additional: TensorDict) -> TensorDict:
            channel, direction = _uniform_selection(additional.get(self.selection_key))
            selected.set((channel, direction))
            return original_init(inputs, additional) if original_init is not None else inputs

        def condition(grad_output, **kwargs):
            selection = selected.get()
            if selection is None:
                raise RuntimeError("Concept selection was not loaded before conditioned attribution")
            channel, direction = selection
            return concept_channel_gradient_callback(channel, direction, channel_axis=self.gradient_channel_axis)(
                grad_output, **kwargs
            )

        bound._init_attr_inputs = initialise
        bound._output_grad_callbacks = {**bound._output_grad_callbacks, self.condition_module: condition}
        prepared = bound._prepare_module(module, in_keys, out_keys, extra_relative_path)
        self._prepared[prepared] = bound
        return prepared

    def _hook_module(self, module: HookedModule) -> BoundHookProgram:
        return self._prepared[module.td_module]._hook_module(module)

    def _restore_module(self, module, in_keys, out_keys, extra_relative_path):
        return module


__all__ = ["ChannelConditionedLRP", "ConceptSelection", "concept_channel_gradient_callback"]
