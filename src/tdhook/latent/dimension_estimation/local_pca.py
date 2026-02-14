"""Local PCA dimension estimation via eigenvalues of local covariance."""

from textwrap import indent
from typing import Literal, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from tdhook._optional_deps import requires_sklearn

from ._utils import sorted_neighbors
from .local_knn import _resolve_k


class LocalPcaDimensionEstimator(TensorDictModuleBase):
    """
    Local intrinsic dimension estimation via PCA on k-NN neighborhoods :cite:`fukunaga1971algorithm`.

    For each point, extracts its k+1 nearest neighbors (self + k neighbors), fits PCA,
    and estimates dimension from eigenvalues using a configurable criterion (maxgap or ratio).

    Reads a data tensor from the input TensorDict. Expects (N, D) or (..., N, D).
    Outputs per-point dimension estimates of shape (..., N).
    """

    def __init__(
        self,
        k: Union[int, Literal["auto"]] = "auto",
        criterion: Literal["maxgap", "ratio"] = "maxgap",
        alpha: float = 0.05,
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
        self.criterion = criterion
        self.alpha = alpha
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
        if N < k + 1:
            raise ValueError(f"At least k+1 points required for local PCA (k={k}), got {N}")
        batch_shape = data.shape[:-2]
        flat = data.reshape(-1, data.shape[-2], data.shape[-1])
        device = data.device
        dtype = data.dtype
        dims = []
        for i in range(flat.shape[0]):
            d_i = _local_pca(
                flat[i],
                k=k,
                eps=self.eps,
                criterion=self.criterion,
                alpha=self.alpha,
                pca_cls=PCA,
            )
            dims.append(d_i)
        td[self.out_key] = torch.stack(dims).reshape(*batch_shape, N).to(device=device, dtype=dtype)
        return td

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys},\nk={self.k},\n"
            f"criterion={self.criterion!r},\nalpha={self.alpha},\neps={self.eps}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _local_pca(
    data: torch.Tensor,
    k: int,
    eps: float,
    criterion: Literal["maxgap", "ratio"],
    alpha: float,
    pca_cls: type,
) -> torch.Tensor:
    """Compute per-point local dimension via PCA. data: (N, D). Returns (N,) dimension estimates."""
    sorted_dist, indices = sorted_neighbors(data, eps)
    N, D = data.shape
    dims = []
    for i in range(N):
        valid_mask = torch.isfinite(sorted_dist[i])
        valid_indices = indices[i][valid_mask]
        neighbor_idx = valid_indices[:k]
        if len(neighbor_idx) < k:
            dims.append(float("nan"))
            continue
        neighborhood = torch.cat([data[i : i + 1], data[neighbor_idx]], dim=0)
        X = neighborhood.detach().cpu().double().numpy()
        if X.shape[0] < 2:
            dims.append(float("nan"))
            continue
        pca = pca_cls(n_components=None).fit(X)
        lambda_ = pca.explained_variance_
        if len(lambda_) == 0:
            dims.append(1.0)
            continue
        if criterion == "maxgap":
            d = float(_dim_from_eigenvalues_maxgap(lambda_))
        elif criterion == "ratio":
            d = float(_dim_from_eigenvalues_ratio(lambda_, alpha))
        else:
            raise ValueError(f"Unknown criterion: {criterion!r}")
        dims.append(d)
    return torch.tensor(dims, device=data.device, dtype=torch.float32)


def _dim_from_eigenvalues_maxgap(lambda_: np.ndarray) -> int:
    """Estimate dimension from eigenvalues using the maximum gap criterion :cite:`bruske1998intrinsic`.

    de = argmax(lambda[i]/lambda[i+1]) + 1 (1-based dimension).
    """
    if len(lambda_) < 2:
        return 1
    gaps = lambda_[:-1] / (lambda_[1:] + 1e-15)
    return int(np.argmax(gaps) + 1)


def _dim_from_eigenvalues_ratio(lambda_: np.ndarray, alpha: float) -> int:
    """Estimate dimension using ratio criterion :cite:`fukunaga1971algorithm`.

    Count eigenvalues above alpha * lambda[0]. Clamped to at least 1.
    """
    if len(lambda_) == 0:
        return 1
    threshold = alpha * lambda_[0]
    de = int(np.sum(lambda_ > threshold))
    return max(1, de)
