from textwrap import indent
from typing import Literal, Union

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from ._utils import sorted_neighbor_distances


def _resolve_k(k: Union[int, Literal["auto"]], n: int) -> int:
    """Resolve k to an integer. If 'auto', use int(n**0.5), clamped to valid range."""
    if k == "auto":
        return max(2, min(int(n**0.5), (n - 1) // 2))
    return k


class LocalKnnDimensionEstimator(TensorDictModuleBase):
    """
    Local intrinsic dimension estimation via k-NN distances :cite:`farahmand2007manifold`.

    For each point x, d(x) = ln(2) / ln(R2k/Rk), where Rk and R2k are distances
    to the k-th and 2k-th nearest neighbors respectively.

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
            if k <= 1:
                raise ValueError("k must be greater than 1")
        self.k = k
        self.in_key = in_key
        self.out_key = out_key
        self.eps = eps
        self.in_keys = [in_key]
        self.out_keys = [out_key]

    def forward(self, td: TensorDict) -> TensorDict:
        data = td[self.in_key]
        N = data.shape[-2]
        k = _resolve_k(self.k, N)
        if N < 2 * k + 1:
            raise ValueError(f"At least 2k+1 points required for local KNN (k={k}), got {N}")
        batch_shape = data.shape[:-2]
        flat = data.reshape(-1, data.shape[-2], data.shape[-1])
        dims = []
        for i in range(flat.shape[0]):
            d_i = _local_knn(flat[i], k=k, eps=self.eps)
            dims.append(d_i)
        td[self.out_key] = torch.stack(dims).reshape(*batch_shape, N)
        return td

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys},\nk={self.k},\neps={self.eps}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _local_knn(data: torch.Tensor, k: int, eps: float) -> torch.Tensor:
    """Compute per-point local dimension. data: (N, D). Returns (N,) dimension estimates."""
    sorted_dist = sorted_neighbor_distances(data, eps)
    rk = sorted_dist[:, k - 1]
    r2k = sorted_dist[:, 2 * k - 1]

    valid = torch.isfinite(rk) & torch.isfinite(r2k) & (r2k > rk)
    ratio = r2k / rk
    log_ratio = torch.log(ratio)

    d = torch.full_like(rk, float("nan"))
    d[valid] = torch.log(torch.tensor(2.0, device=data.device, dtype=data.dtype)) / log_ratio[valid]
    d[~valid] = float("nan")

    return d.float()
