"""Model-evaluated attribution metrics with TensorDict-native inputs and outputs."""

from __future__ import annotations

from typing import List

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from tdhook._types import join_keys


def _feature_reduce(value: torch.Tensor, batch_dims: int, *, mean: bool = False) -> torch.Tensor:
    dims = tuple(range(batch_dims, value.ndim))
    if not dims:
        return value
    return value.mean(dim=dims) if mean else value.sum(dim=dims)


def _require_tensor(data: TensorDictBase, key: NestedKey, *, role: str) -> torch.Tensor:
    value = data.get(key, None)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"{role} tensor is missing at key {key!r}")
    return value


class InfidelityMetric:
    """Measure attribution infidelity using repeated perturbed model evaluations.

    ``original_data`` must already contain attributions and the corresponding
    unperturbed model output. Each perturbation performs exactly one call to
    the prepared TensorDict module.
    """

    def __init__(
        self,
        n_perturb_samples: int = 10,
        *,
        attribution_key: NestedKey = "attr",
        output_key: NestedKey = ("_mod_out", "output"),
    ):
        if not isinstance(n_perturb_samples, int) or isinstance(n_perturb_samples, bool):
            raise TypeError("n_perturb_samples must be an int")
        if n_perturb_samples <= 0:
            raise ValueError("n_perturb_samples must be positive")
        self.n_perturb_samples = n_perturb_samples
        self.attribution_key = attribution_key
        self.output_key = output_key

    def additional_model_passes(self, module: TensorDictModuleBase) -> int:
        """Return the exact number of prepared-module calls for ``module``."""

        return self.n_perturb_samples * len(module.in_keys)

    def __call__(
        self,
        module: TensorDictModuleBase,
        original_data: TensorDictBase,
    ) -> TensorDict:
        """Return one infidelity tensor per native model input key."""

        original_output = _require_tensor(original_data, self.output_key, role="Model output")
        results = TensorDict(batch_size=original_data.batch_size, device=original_data.device)
        batch_dims = original_data.ndim

        for key in module.in_keys:
            original_input = _require_tensor(original_data, key, role="Model input")
            original_attr = _require_tensor(
                original_data,
                join_keys(self.attribution_key, key),
                role="Attribution",
            )
            perturbation_scores = []
            output_changes = []

            for _ in range(self.n_perturb_samples):
                perturbed_data = self._perturb_data(original_data, [key])
                perturbed_result = module(perturbed_data)
                perturbed_input = _require_tensor(perturbed_result, key, role="Perturbed model input")
                perturbed_output = _require_tensor(perturbed_result, self.output_key, role="Perturbed model output")
                perturbation = original_input - perturbed_input
                perturbation_scores.append(_feature_reduce(original_attr * perturbation, batch_dims))
                output_changes.append(_feature_reduce(original_output - perturbed_output, batch_dims))

            perturbation_scores_tensor = torch.stack(perturbation_scores, dim=-1)
            output_changes_tensor = torch.stack(output_changes, dim=-1)
            results.set(key, ((perturbation_scores_tensor - output_changes_tensor) ** 2).mean(dim=-1))

        return results

    @staticmethod
    @torch.no_grad()
    def _perturb_data(data: TensorDictBase, in_keys: List[NestedKey]) -> TensorDictBase:
        perturbed_data = data.clone()
        for key in in_keys:
            value = _require_tensor(perturbed_data, key, role="Model input")
            if not value.is_floating_point():
                raise TypeError(f"Model input at key {key!r} must be floating point")
            perturbed_data.set(key, value + torch.randn_like(value) * 0.01)
        return perturbed_data


class SensitivityMetric:
    """Measure relative attribution change after one perturbed model evaluation."""

    def __init__(
        self,
        perturb_radius: float = 0.02,
        *,
        attribution_key: NestedKey = "attr",
    ):
        if perturb_radius < 0:
            raise ValueError("perturb_radius must be non-negative")
        self.perturb_radius = perturb_radius
        self.attribution_key = attribution_key

    def additional_model_passes(self, module: TensorDictModuleBase) -> int:
        """Return the exact number of prepared-module calls for ``module``."""

        return 1

    def __call__(
        self,
        module: TensorDictModuleBase,
        original_data: TensorDictBase,
    ) -> TensorDict:
        """Return one sensitivity tensor per native model input key."""

        perturbed_data = self._perturb_data(original_data, module.in_keys)
        perturbed_result = module(perturbed_data)
        results = TensorDict(batch_size=original_data.batch_size, device=original_data.device)
        batch_dims = original_data.ndim

        for key in module.in_keys:
            attr_key = join_keys(self.attribution_key, key)
            original_attr = _require_tensor(original_data, attr_key, role="Attribution")
            perturbed_attr = _require_tensor(perturbed_result, attr_key, role="Perturbed attribution")
            explanation_diff = (original_attr - perturbed_attr).abs()
            original_magnitude = _feature_reduce(original_attr.abs(), batch_dims, mean=True)
            explanation_diff_mean = _feature_reduce(explanation_diff, batch_dims, mean=True)
            results.set(
                key,
                torch.where(
                    original_magnitude == 0,
                    explanation_diff_mean,
                    explanation_diff_mean / original_magnitude,
                ),
            )

        return results

    @torch.no_grad()
    def _perturb_data(self, data: TensorDictBase, in_keys: List[NestedKey]) -> TensorDictBase:
        perturbed_data = data.clone()
        for key in in_keys:
            value = _require_tensor(perturbed_data, key, role="Model input")
            if not value.is_floating_point():
                raise TypeError(f"Model input at key {key!r} must be floating point")
            noise = torch.empty_like(value).uniform_(-self.perturb_radius, self.perturb_radius)
            perturbed_data.set(key, value + noise)
        return perturbed_data


__all__ = ["InfidelityMetric", "SensitivityMetric"]
