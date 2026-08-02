"""Declared stages for concept-conditioned attribution workflows.

The stages here keep concept selection as a TensorDict artifact instead of a
Python value captured by a notebook callback.  This gives the pipeline a
visible boundary between collecting concept evidence and running the
conditioned attribution pass.
"""

from __future__ import annotations

from copy import copy
from typing import Literal

import torch
from torch import nn
from tensordict import TensorDict, TensorDictBase

from tdhook.artifacts import ArtifactContract
from tdhook.attribution import LRP
from tdhook.pipeline import PipelineKey, Stage
from tdhook.stages import AttributionStage


class ConceptSelectionStage(Stage):
    """Select one channel from labelled concept-example relevances.

    The stage publishes positive and negative means, their difference, and a
    scalar ``channel``/``direction`` selection under ``selection_key``.  A
    replacement selection policy only needs to provide the same artifact
    schema, so downstream conditioned attribution remains unchanged.
    """

    def __init__(
        self,
        name: str,
        *,
        relevance_key: PipelineKey = ("attributions", "concept_examples"),
        labels_key: PipelineKey = ("inputs", "concept_labels"),
        selection_key: PipelineKey = ("metrics", "concept_selection"),
        direction: Literal["positive", "negative"] = "positive",
    ) -> None:
        if direction not in {"positive", "negative"}:
            raise ValueError("direction must be 'positive' or 'negative'")
        super().__init__(
            name,
            artifact_contract=ArtifactContract(
                requires={"relevances": relevance_key, "labels": labels_key},
                provides={"selection": selection_key},
            ),
            effects=("concept_selection",),
            method_id="concept-channel-selection",
        )
        self.relevance_key = relevance_key
        self.labels_key = labels_key
        self.selection_key = selection_key
        self.direction = direction

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        relevances = artifacts.get(self.relevance_key)
        labels = artifacts.get(self.labels_key)
        if not isinstance(relevances, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Concept selection requires tensor relevances and labels")
        if relevances.ndim < 2:
            raise ValueError("Concept relevances must have batch and channel dimensions")
        if labels.ndim != 1 or labels.shape[0] != relevances.shape[0]:
            raise ValueError("Concept labels must be one-dimensional and match the relevance batch")
        positive = labels.to(dtype=torch.bool, device=relevances.device)
        if not positive.any() or positive.all():
            raise ValueError("Concept examples must contain both positive and negative labels")
        values = relevances.abs()
        positive_mean = values[positive].mean(dim=(0, *range(2, values.ndim)))
        negative_mean = values[~positive].mean(dim=(0, *range(2, values.ndim)))
        scores = positive_mean - negative_mean
        channel = scores.argmax() if self.direction == "positive" else scores.argmin()
        direction = 1 if self.direction == "positive" else -1
        # TensorDict nested values inherit the pipeline batch size.  Repeat
        # this global selection for each example rather than relying on a
        # hidden side cache with an incompatible scalar batch shape.
        batch_size = artifacts.batch_size
        batch_shape = tuple(batch_size)
        artifacts.set(
            self.selection_key,
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
        axis = channel_axis % gradient.ndim
        if gradient.ndim < 2 or channel >= gradient.shape[axis]:
            raise ValueError(f"Concept channel {channel} is invalid for gradient shape {tuple(gradient.shape)}")
        mask = torch.zeros_like(gradient)
        mask.select(axis, channel).fill_(direction)
        return (gradient * mask,)

    return callback


class ChannelConditionedLRPStage(Stage):
    """Run an LRP pass conditioned on a selected channel artifact.

    ``base_factory`` retains all normal LRP settings (rules, targets, and
    input modules).  For each execution the stage copies it and binds the
    declared selection artifact to ``condition_module``.  No notebook-level
    context or cache state is transferred between the two model passes.
    """

    def __init__(
        self,
        name: str,
        base_factory: LRP,
        *,
        condition_module: str,
        gradient_channel_axis: int = -1,
        input_key: PipelineKey = ("inputs", "input"),
        selection_key: PipelineKey = ("metrics", "concept_selection"),
        attribution_key: PipelineKey = ("attributions", "conditioned"),
        legacy_attribution_key: PipelineKey = ("attr", "input"),
    ) -> None:
        if not condition_module:
            raise ValueError("condition_module must be non-empty")
        super().__init__(
            name,
            artifact_contract=ArtifactContract(
                requires={"input": input_key, "selection": selection_key},
                provides={"attributions": attribution_key},
            ),
            effects=("model_execution", "gradient", "concept_conditioning"),
            method_id="LRP[concept-conditioned]",
            model_passes=1,
            gradient_mode="required",
            device_batch_constraints=("inputs and selected channel must share the attribution device",),
        )
        self.base_factory = base_factory
        self.condition_module = condition_module
        self.gradient_channel_axis = gradient_channel_axis
        self.input_key = input_key
        self.selection_key = selection_key
        self.attribution_key = attribution_key
        self.legacy_attribution_key = legacy_attribution_key

    def run(self, model: nn.Module, artifacts: TensorDictBase) -> TensorDictBase:
        selection = artifacts.get(self.selection_key)
        if not isinstance(selection, TensorDictBase) or "channel" not in selection or "direction" not in selection:
            raise ValueError("Concept selection must provide scalar 'channel' and 'direction' artifacts")
        channel = selection.get("channel")
        direction = selection.get("direction")
        if not isinstance(channel, torch.Tensor) or not isinstance(direction, torch.Tensor):
            raise TypeError("Concept selection channel and direction must be tensors")
        if channel.numel() == 0 or direction.numel() == 0:
            raise ValueError("Concept selection channel and direction cannot be empty")
        if not torch.equal(channel, channel.flatten()[0].expand_as(channel)) or not torch.equal(
            direction, direction.flatten()[0].expand_as(direction)
        ):
            raise ValueError("Concept selection must be identical for every example in the batch")
        factory = copy(self.base_factory)
        callbacks = dict(factory._output_grad_callbacks)
        if self.condition_module in callbacks:
            raise ValueError(f"LRP factory already has a gradient callback for {self.condition_module!r}")
        callbacks[self.condition_module] = concept_channel_gradient_callback(
            int(channel.flatten()[0].item()),
            int(direction.flatten()[0].item()),
            channel_axis=self.gradient_channel_axis,
        )
        factory._output_grad_callbacks = callbacks
        return AttributionStage(
            self.name,
            factory,
            input_key=self.input_key,
            attribution_key=self.attribution_key,
            legacy_attribution_key=self.legacy_attribution_key,
        ).run(model, artifacts)
