from textwrap import indent

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase


class LinearCkaEstimator(TensorDictModuleBase):
    """
    Linear centered kernel alignment (CKA) between two representations.

    Reads two data tensors from the input TensorDict. Expects `(N, D)` or
    `(..., N, D)` for both tensors, with shared batch shape and sample count.
    Outputs one scalar similarity value per batch item.
    """

    def __init__(
        self,
        in_key_a: str = "data_a",
        in_key_b: str = "data_b",
        out_key: str = "cka",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.in_key_a = in_key_a
        self.in_key_b = in_key_b
        self.out_key = out_key
        self.eps = eps
        self.in_keys = [in_key_a, in_key_b]
        self.out_keys = [out_key]

    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.in_key_a]
        y = td[self.in_key_b]
        _validate_inputs(x, y)

        batch_shape = x.shape[:-2]
        n = x.shape[-2]
        flat_x = x.reshape(-1, n, x.shape[-1])
        flat_y = y.reshape(-1, n, y.shape[-1])
        cka_values = [_linear_cka(flat_x[i], flat_y[i], eps=self.eps) for i in range(flat_x.shape[0])]
        td[self.out_key] = torch.stack(cka_values).reshape(batch_shape)
        return td

    def __repr__(self):
        fields = indent(
            f"in_keys={self.in_keys},\nout_keys={self.out_keys},\neps={self.eps}",
            4 * " ",
        )
        return f"{type(self).__name__}(\n{fields})"


def _validate_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("Linear CKA expects tensors with shape (N, D) or (..., N, D)")
    if x.shape[:-2] != y.shape[:-2]:
        raise ValueError(f"Expected matching batch shapes, got {x.shape[:-2]} and {y.shape[:-2]}")
    if x.shape[-2] != y.shape[-2]:
        raise ValueError(f"Expected matching sample counts, got {x.shape[-2]} and {y.shape[-2]}")
    if x.device != y.device:
        raise ValueError(f"Expected both tensors on the same device, got {x.device} and {y.device}")


def _linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = torch.promote_types(x.dtype, y.dtype)
    if not torch.empty((), dtype=dtype).is_floating_point():
        dtype = torch.float32

    x = x.to(dtype=dtype)
    y = y.to(dtype=dtype)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    cross_cov = x.transpose(-1, -2) @ y
    x_cov = x.transpose(-1, -2) @ x
    y_cov = y.transpose(-1, -2) @ y

    numerator = torch.sum(cross_cov.square())
    x_norm = torch.sum(x_cov.square())
    y_norm = torch.sum(y_cov.square())
    denominator = torch.sqrt(x_norm * y_norm)

    nan = torch.full((), float("nan"), dtype=dtype, device=x.device)
    if not torch.isfinite(denominator) or denominator <= eps:
        return nan

    value = numerator / denominator
    return value.float() if torch.isfinite(value) else nan
