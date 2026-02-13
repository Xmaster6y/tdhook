"""Local intrinsic dimension estimation via k-nearest neighbors."""

from textwrap import indent

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from ._utils import sorted_neighbor_distances


class LocalKnnDimensionEstimator(TensorDictModuleBase):
    """
    Local intrinsic dimension estimation via k-NN distances.

    For each point x, d(x) = ln(2) / ln(R2k/Rk), where Rk and R2k are distances
    to the k-th and 2k-th nearest neighbors respectively.

    Reads a data tensor from the input TensorDict. Expects (N, D) or (..., N, D).
    Outputs per-point dimension estimates of shape (..., N).
    """

    def __init__(
        self,
        k: int,
        in_key: str = "data",
        out_key: str = "dimension",
        eps: float = 1e-5,
    ):
        super().__init__()
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.in_key = in_key
        self.out_key = out_key
        self.eps = eps
        self.in_keys = [in_key]
        self.out_keys = [out_key]

    def forward(self, td: TensorDict) -> TensorDict:
        data = td[self.in_key]
        N = data.shape[-2]
        if N < 2 * self.k + 1:
            raise ValueError(f"At least 2k+1 points required for local KNN (k={self.k}), got {N}")
        batch_shape = data.shape[:-2]
        flat = data.reshape(-1, data.shape[-2], data.shape[-1])
        dims = []
        for i in range(flat.shape[0]):
            d_i = _local_knn(flat[i], k=self.k, eps=self.eps)
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
