"""Shared utilities for intrinsic dimension estimation."""

import torch


def sorted_neighbors(data: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sorted distances and indices of neighbors for each point.

    Returns (sorted_dist, indices) each of shape (N, N). Each row is sorted ascending.
    Self and distances <= eps are inf, with their indices appearing last in each row.
    """
    dist = torch.cdist(data, data, p=2)
    dist = dist.clone()
    dist.fill_diagonal_(float("inf"))
    dist = torch.where(dist > eps, dist, float("inf"))
    sorted_dist, indices = torch.sort(dist, dim=1)
    return sorted_dist, indices
