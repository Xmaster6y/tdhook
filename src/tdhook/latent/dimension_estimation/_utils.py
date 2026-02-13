"""Shared utilities for intrinsic dimension estimation."""

import torch


def sorted_neighbor_distances(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute sorted distances to neighbors for each point.

    Returns (N, N) where each row is sorted ascending. Self and distances <= eps are inf.
    """
    dist = torch.cdist(data, data, p=2)
    dist = dist.clone()
    dist.fill_diagonal_(float("inf"))
    dist = torch.where(dist > eps, dist, float("inf"))
    sorted_dist, _ = torch.sort(dist, dim=1)
    return sorted_dist
