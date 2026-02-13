from textwrap import indent

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from ._utils import sorted_neighbor_distances


class TwoNnDimensionEstimator(TensorDictModuleBase):
    """
    Intrinsic dimension estimation via the Two NN algorithm :cite:`facco_estimating_2017`.

    Reads a data tensor from the input TensorDict. Expects (N, D) or (..., N, D).
    For (..., N, D), flattens all leading dims, computes one dimension per dataset,
    stacks and reshapes to preserve the original batch shape (excluding last two dims).
    """

    def __init__(
        self,
        in_key: str = "data",
        out_key: str = "dimension",
        return_xy: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.return_xy = return_xy
        self.eps = eps
        self.in_keys = [in_key]
        self.out_keys = [out_key, f"{out_key}_x", f"{out_key}_y"] if return_xy else [out_key]

    def forward(self, td: TensorDict) -> TensorDict:
        data = td[self.in_key]
        if data.shape[-2] < 3:
            raise ValueError("At least 3 points required for Two NN dimension estimation")
        batch_shape = data.shape[:-2]
        flat = data.reshape(-1, data.shape[-2], data.shape[-1])
        dims = []
        xs, ys = [], []
        for i in range(flat.shape[0]):
            d_i, x_i, y_i = _twonn(flat[i], eps=self.eps)
            dims.append(d_i)
            if self.return_xy:
                xs.append(x_i)
                ys.append(y_i)
        td[self.out_key] = torch.stack(dims).reshape(batch_shape)
        if self.return_xy:
            max_len = data.shape[-2] - 1
            x_padded = torch.stack(
                [torch.nn.functional.pad(x_i, (0, max_len - len(x_i)), value=float("nan")) for x_i in xs]
            )
            y_padded = torch.stack(
                [torch.nn.functional.pad(y_i, (0, max_len - len(y_i)), value=float("nan")) for y_i in ys]
            )
            td[f"{self.out_key}_x"] = x_padded.reshape(*batch_shape, max_len)
            td[f"{self.out_key}_y"] = y_padded.reshape(*batch_shape, max_len)
        return td

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys},\neps={self.eps}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _twonn(data: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Two NN intrinsic dimension. data: (N, D). Returns (d, x, y).

    Distances <= eps are treated as duplicates (excluded from nearest-neighbor selection).
    """
    sorted_dist = sorted_neighbor_distances(data, eps)
    r1 = sorted_dist[:, 0]
    r2 = sorted_dist[:, 1]

    valid = torch.isfinite(r1) & torch.isfinite(r2)
    N_valid = valid.sum().item()
    if N_valid < 3:
        raise ValueError("At least 3 valid points required for Two NN dimension estimation")

    mu = (r2 / r1)[valid]
    sorted_indices = torch.argsort(mu)
    mu_sorted = mu[sorted_indices]

    # Empirical CDF F = rank/N; exclude F=1 to avoid log(0), so len(x) = N_valid - 1
    rank_1based = torch.arange(1, N_valid, device=data.device, dtype=data.dtype)
    one_minus_F = 1 - rank_1based / N_valid
    x = torch.log(mu_sorted[:-1])
    y = -torch.log(one_minus_F)

    x_col = x.unsqueeze(1)
    d = torch.linalg.lstsq(x_col, y.unsqueeze(1), rcond=None, driver=None).solution.squeeze()
    d = d.float()

    return d, x, y
