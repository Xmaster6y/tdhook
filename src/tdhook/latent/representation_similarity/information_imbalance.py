from textwrap import indent

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase


class InformationImbalanceEstimator(TensorDictModuleBase):
    """
    Information Imbalance between two representations.

    Reads two data tensors from the input TensorDict. Expects `(N, D)` or
    `(..., N, D)` for both tensors, with shared batch shape and sample count.
    Outputs both directional imbalances per batch item: A->B and B->A.

    This implementation uses the nearest-neighbor definition:
    for each point i, select j such that r^A_ij = 1 and average r^B_ij with
    normalization 2 / N, yielding values close to 0 for strong
    neighborhood predictability and close to 1 for uninformative mappings.
    """

    def __init__(
        self,
        in_key_a: str = "data_a",
        in_key_b: str = "data_b",
        out_key_a_to_b: str = "information_imbalance_a_to_b",
        out_key_b_to_a: str = "information_imbalance_b_to_a",
        p: float = 2.0,
    ):
        super().__init__()

        self.in_key_a = in_key_a
        self.in_key_b = in_key_b
        self.out_key_a_to_b = out_key_a_to_b
        self.out_key_b_to_a = out_key_b_to_a
        self.p = p
        self.in_keys = [in_key_a, in_key_b]
        self.out_keys = [out_key_a_to_b, out_key_b_to_a]

    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.in_key_a]
        y = td[self.in_key_b]
        _validate_inputs(x, y)

        batch_shape = x.shape[:-2]
        n = x.shape[-2]
        flat_x = x.reshape(-1, n, x.shape[-1])
        flat_y = y.reshape(-1, n, y.shape[-1])
        if flat_x.shape[0] == 0:
            empty = torch.empty(batch_shape, dtype=torch.float32, device=x.device)
            td[self.out_key_a_to_b] = empty
            td[self.out_key_b_to_a] = empty.clone()
            return td

        values_a_to_b = []
        values_b_to_a = []
        for i in range(flat_x.shape[0]):
            a_to_b, b_to_a = _information_imbalance(flat_x[i], flat_y[i], p=self.p)
            values_a_to_b.append(a_to_b)
            values_b_to_a.append(b_to_a)

        td[self.out_key_a_to_b] = torch.stack(values_a_to_b).reshape(batch_shape)
        td[self.out_key_b_to_a] = torch.stack(values_b_to_a).reshape(batch_shape)
        return td

    def __repr__(self):
        fields = indent(
            (f"in_keys={self.in_keys},\nout_keys={self.out_keys},\np={self.p}"),
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _validate_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("Information Imbalance expects tensors with shape (N, D) or (..., N, D)")
    if x.shape[:-2] != y.shape[:-2]:
        raise ValueError(f"Expected matching batch shapes, got {x.shape[:-2]} and {y.shape[:-2]}")
    if x.shape[-2] != y.shape[-2]:
        raise ValueError(f"Expected matching sample counts, got {x.shape[-2]} and {y.shape[-2]}")
    if x.device != y.device:
        raise ValueError(f"Expected both tensors on the same device, got {x.device} and {y.device}")

    n = x.shape[-2]
    if n < 2:
        raise ValueError(f"Expected at least 2 samples, got {n}")


def _compute_ranks_from_dist(dist: torch.Tensor) -> torch.Tensor:
    """
    Convert pairwise distances (N, N) to rank matrix (N, N).

    Convention: diagonal entries (self) have rank 0 and non-diagonal entries are
    ranked from 1 to N-1 according to ascending distance for each row.
    """
    n = dist.shape[0]
    working = dist.clone()
    working.fill_diagonal_(float("-inf"))
    order = torch.argsort(working, dim=1)

    rank_values = torch.arange(n, device=dist.device, dtype=torch.int64).expand(n, -1)
    ranks = torch.empty_like(order, dtype=torch.int64)
    ranks.scatter_(1, order, rank_values)
    return ranks


def _as_distance_input_dtype(x: torch.Tensor, y: torch.Tensor) -> torch.dtype:
    dtype = torch.promote_types(x.dtype, y.dtype)
    if not torch.empty((), dtype=dtype).is_floating_point():
        return torch.float32
    return dtype


def _directional_imbalance(
    source_ranks: torch.Tensor,
    target_ranks: torch.Tensor,
) -> torch.Tensor:
    n = source_ranks.shape[0]
    source_mask = source_ranks == 1
    selected_target_ranks = target_ranks[source_mask].to(dtype=torch.float32)
    avg_rank = selected_target_ranks.mean()
    return (2.0 / n) * avg_rank


def _information_imbalance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = _as_distance_input_dtype(x, y)
    x = x.to(dtype=dtype)
    y = y.to(dtype=dtype)

    dist_a = torch.cdist(x, x, p=p)
    dist_b = torch.cdist(y, y, p=p)

    ranks_a = _compute_ranks_from_dist(dist_a)
    ranks_b = _compute_ranks_from_dist(dist_b)

    a_to_b = _directional_imbalance(ranks_a, ranks_b)
    b_to_a = _directional_imbalance(ranks_b, ranks_a)
    return a_to_b, b_to_a
