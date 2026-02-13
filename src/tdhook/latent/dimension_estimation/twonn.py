from textwrap import indent

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase


class TwoNnDimensionEstimator(TensorDictModuleBase):
    """
    Intrinsic dimension estimation via the Two NN algorithm :cite:`facco_estimating_2017`.

    Reads a data tensor from the input TensorDict, flattens it to 2D (samples × features),
    and writes the estimated intrinsic dimension to the output key.
    """

    def __init__(
        self,
        in_key: str = "data",
        out_key: str = "dimension",
        return_xy: bool = False,
    ):
        super().__init__()
        self._in_key = in_key
        self._out_key = out_key
        self._return_xy = return_xy
        self.in_keys = [in_key]
        if return_xy:
            self.out_keys = [out_key, f"{out_key}_x", f"{out_key}_y"]
        else:
            self.out_keys = [out_key]

    def forward(self, td: TensorDict) -> TensorDict:
        data = td[self._in_key]
        data = data.flatten(0, -2)
        if data.shape[0] < 3:
            raise ValueError("At least 3 points required for Two NN dimension estimation")
        d, x, y = _twonn(data)
        td[self._out_key] = d.reshape(())
        if self._return_xy:
            td[f"{self._out_key}_x"] = x
            td[f"{self._out_key}_y"] = y
        return td

    def __repr__(self):
        fields = indent(
            f"in_key={self._in_key!r},\nout_key={self._out_key!r},\nreturn_xy={self._return_xy}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _twonn(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Two NN intrinsic dimension. data: (N, D). Returns (d, x, y)."""
    dist = torch.cdist(data, data, p=2)
    dist = dist.clone()
    dist.fill_diagonal_(float("inf"))  # exclude self from nearest neighbors
    sorted_dist, _ = torch.sort(dist, dim=1)
    r1 = sorted_dist[:, 0]
    r2 = sorted_dist[:, 1]

    # Filter out points where r1 or r2 is 0 (duplicates)
    valid = (r1 > 0) & (r2 > 0)
    if valid.sum() < 3:
        raise ValueError("Too many duplicate or degenerate points for Two NN dimension estimation")

    mu = (r2 / r1)[valid]
    N_valid = mu.shape[0]

    sorted_indices = torch.argsort(mu)
    mu_sorted = mu[sorted_indices]

    # Empirical CDF F = rank/N; exclude F=1 to avoid log(0)
    rank_1based = torch.arange(1, N_valid, device=data.device, dtype=data.dtype)
    one_minus_F = 1 - rank_1based / N_valid
    x = torch.log(mu_sorted[:-1])
    y = -torch.log(one_minus_F)
    if len(x) < 3:
        raise ValueError("Insufficient valid points for linear fit")

    x_col = x.unsqueeze(1)
    d = torch.linalg.lstsq(x_col, y.unsqueeze(1), rcond=None, driver=None).solution.squeeze()
    d = d.float()

    return d, x, y
