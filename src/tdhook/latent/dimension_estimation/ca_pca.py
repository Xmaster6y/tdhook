from textwrap import indent
from typing import Literal, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from tdhook._optional_deps import requires_sklearn

from ._utils import sorted_neighbors
from .local_knn import _resolve_k


class CaPcaDimensionEstimator(TensorDictModuleBase):
    """
    Curvature-adjusted intrinsic dimension estimation via local PCA :cite:`gilbert2023capca`.

    Extends local PCA by calibrating to a quadratic embedding instead of a flat unit ball,
    accounting for manifold curvature. For each point, uses its k+1 nearest neighbors,
    forms the local covariance, and selects dimension by comparing curvature-corrected
    eigenvalues to the expected spectrum of a d-dimensional ball.

    Reads a data tensor from the input TensorDict. Expects (N, D) or (..., N, D).
    Outputs per-point dimension estimates of shape (..., N).
    """

    def __init__(
        self,
        k: Union[int, Literal["auto"]] = "auto",
        in_key: str = "data",
        out_key: str = "dimension",
        eps: float = 1e-5,
    ):
        super().__init__()
        if k != "auto":
            if not isinstance(k, int):
                raise TypeError("k must be an int or 'auto'")
            if k < 1:
                raise ValueError("k must be at least 1")
        self.k = k
        self.in_key = in_key
        self.out_key = out_key
        self.eps = eps
        self.in_keys = [in_key]
        self.out_keys = [out_key]

    @requires_sklearn
    def forward(self, td: TensorDict) -> TensorDict:
        from sklearn.decomposition import PCA

        data = td[self.in_key]
        N = data.shape[-2]
        k = _resolve_k(self.k, N)
        if N < k + 2:
            raise ValueError(f"At least k+2 points required for CA-PCA (k={k}), got {N}")
        batch_shape = data.shape[:-2]
        flat = data.reshape(-1, data.shape[-2], data.shape[-1])
        device = data.device
        dtype = data.dtype
        dims = []
        for i in range(flat.shape[0]):
            d_i = _ca_pca(flat[i], k=k, eps=self.eps, pca_cls=PCA)
            dims.append(d_i)
        td[self.out_key] = torch.stack(dims).reshape(*batch_shape, N).to(device=device, dtype=dtype)
        return td

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys},\nk={self.k},\neps={self.eps}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _ca_pca(data: torch.Tensor, k: int, eps: float, pca_cls: type) -> torch.Tensor:
    """Compute per-point dimension via CA-PCA. data: (N, D). Returns (N,) dimension estimates."""
    sorted_dist, indices = sorted_neighbors(data, eps)
    N, D = data.shape
    dims = []
    for i in range(N):
        dist_k = sorted_dist[i, k - 1]
        dist_kp1 = sorted_dist[i, k]
        r = (dist_k + dist_kp1) / 2.0
        if r <= 0 or not np.isfinite(float(r)):
            dims.append(float("nan"))
            continue
        neighbor_idx = indices[i, :k]
        neighborhood = data[neighbor_idx].cpu().double().numpy()
        if neighborhood.shape[0] < 2:
            dims.append(float("nan"))
            continue
        pca = pca_cls(n_components=None).fit(neighborhood)
        eigvals = pca.explained_variance_
        lambda_hat = np.zeros(D, dtype=np.float64)
        n_eig = min(len(eigvals), D)
        lambda_hat[:n_eig] = eigvals[:n_eig] / (r**2)
        d_est = _dim_from_ca_pca(lambda_hat, D)
        dims.append(float(d_est))
    return torch.tensor(dims, device=data.device, dtype=torch.float32)


def _dim_from_ca_pca(lambda_hat: np.ndarray, D: int) -> int:
    """Select dimension via curvature-corrected eigenvalue matching."""
    best_d = 1
    best_score = np.inf
    for d in range(1, D + 1):
        tail_sum = lambda_hat[d:].sum()
        coef = (3 * d + 4) / (d * (d + 4)) if d > 0 else 0.0
        lambda_d = np.zeros(D)
        lambda_d[:d] = lambda_hat[:d] + coef * tail_sum
        lambda_d[d:] = 0.0
        target = np.zeros(D)
        target[:d] = 1.0 / (d + 2)
        target[d:] = 0.0
        score = float(np.linalg.norm(target - lambda_d)) + 2.0 * tail_sum
        if score < best_score:
            best_score = score
            best_d = d
    return best_d
