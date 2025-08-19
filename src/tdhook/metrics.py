"""
Metrics
"""

import torch
from tensordict import TensorDict
from typing import List

from tdhook.module import HookedModule


# TODO: test against captum
# TODO: fix this
class InfidelityMetric:
    def __init__(self, n_perturb_samples: int = 10):
        self.n_perturb_samples = n_perturb_samples

    def __call__(
        self,
        module: HookedModule,
        original_data: TensorDict,
    ) -> TensorDict:
        """Compute infidelity as the difference between attribution-weighted perturbations and model output changes."""

        infidelities = TensorDict(batch_size=original_data.batch_size)
        n_batch_dims = len(original_data.batch_size)

        for key in module.in_keys:
            # Get original attribution
            original_attr = original_data.get(("attr", key))

            # Generate multiple perturbations
            perturbation_scores = []
            output_changes = []

            for _ in range(self.n_perturb_samples):
                # Generate perturbed data
                perturbed_data = self._perturb_data(original_data, [key])

                # Get perturbed attribution
                module(perturbed_data)

                # Calculate perturbation (difference between original and perturbed input)
                perturbation = original_data[key] - perturbed_data[key]

                # Attribution-weighted perturbation
                attr_weighted_perturb = (original_attr * perturbation).sum(
                    dim=tuple(range(n_batch_dims, original_attr.dim()))
                )

                # Model output change
                original_output = module(original_data)[("_mod_out", "output")]
                perturbed_output = module(perturbed_data)[("_mod_out", "output")]
                output_change = (original_output - perturbed_output).sum(
                    dim=tuple(range(n_batch_dims, original_output.dim()))
                )

                perturbation_scores.append(attr_weighted_perturb)
                output_changes.append(output_change)

            # Stack results
            perturbation_scores = torch.stack(perturbation_scores, dim=-1)  # [batch, n_samples]
            output_changes = torch.stack(output_changes, dim=-1)  # [batch, n_samples]

            # Compute infidelity as MSE between attribution-weighted perturbations and output changes
            infidelity = ((perturbation_scores - output_changes) ** 2).mean(dim=-1)
            infidelities[key] = infidelity

        return infidelities

    @torch.no_grad()
    def _perturb_data(self, data: TensorDict, in_keys: List[str]) -> TensorDict:
        """Add random noise to create perturbations."""
        perturbed_data = data.clone()
        for key in in_keys:
            value = perturbed_data[key]
            if isinstance(value, torch.Tensor):
                # Generate random noise for perturbation
                noise = torch.randn_like(value) * 0.01  # Small noise
                perturbed_data[key] = value + noise
        return perturbed_data


class SensitivityMetric:
    def __init__(self, perturb_radius: float = 0.02):
        self.perturb_radius = perturb_radius

    def __call__(
        self,
        module: HookedModule,
        original_data: TensorDict,
    ) -> TensorDict:
        """Compute sensitivity as the relative change in explanation when input is perturbed."""

        perturbed_data = self._perturb_data(original_data, module.in_keys)
        module(perturbed_data)

        sensitivities = TensorDict(batch_size=original_data.batch_size)
        n_batch_dims = len(original_data.batch_size)

        for key in module.in_keys:
            original_attr = original_data.get(("attr", key))
            perturbed_attr = perturbed_data.get(("attr", key))
            explanation_diff = (original_attr - perturbed_attr).abs()

            # Calculate mean over all dimensions except the batch dimensions
            # batch dimensions are the first `batch_size` dimensions
            non_batch_dims = tuple(range(n_batch_dims, original_attr.dim()))
            original_magnitude = original_attr.abs().mean(dim=non_batch_dims)
            explanation_diff_mean = explanation_diff.mean(dim=non_batch_dims)

            # Avoid division by zero
            sensitivities[key] = torch.where(
                original_magnitude == 0, explanation_diff_mean, explanation_diff_mean / original_magnitude
            )

        return sensitivities

    @torch.no_grad()
    def _perturb_data(self, data: TensorDict, in_keys: List[str]) -> TensorDict:
        """Add random noise within the perturbation radius."""
        perturbed_data = data.clone()
        for key in in_keys:
            value = perturbed_data[key]
            if isinstance(value, torch.Tensor):
                noise = (
                    torch.FloatTensor(value.size())
                    .uniform_(-self.perturb_radius, self.perturb_radius)  # TODO: replace with actual radius dist
                    .to(value.device)
                )
                perturbed_data[key] = value + noise
        return perturbed_data
