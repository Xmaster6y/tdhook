"""
Tests for representation similarity estimators.
"""

import pytest
import torch
from tensordict import TensorDict

from tdhook.latent.representation_similarity import CkaEstimator


def make_td(x, y, in_key_a="data_a", in_key_b="data_b", batch_size=None):
    if batch_size is None:
        batch_size = [] if x.ndim == 2 else x.shape[:-2]
    return TensorDict({in_key_a: x, in_key_b: y}, batch_size=batch_size)


def make_random_pair(n=64, d_x=10, d_y=7):
    return torch.randn(n, d_x), torch.randn(n, d_y)


@pytest.fixture
def run_estimator():
    torch.manual_seed(42)

    def _run(x, y, in_key_a="data_a", in_key_b="data_b", batch_size=None, **estimator_kwargs):
        td = make_td(x, y, in_key_a=in_key_a, in_key_b=in_key_b, batch_size=batch_size)
        return CkaEstimator(in_key_a=in_key_a, in_key_b=in_key_b, **estimator_kwargs)(td)

    return _run


class TestCkaEstimator:
    def test_default_keys(self, run_estimator):
        x, y = make_random_pair()

        result = run_estimator(x, y)

        assert "cka" in result
        assert result["cka"].ndim == 0
        assert result["cka"].dtype in (torch.float32, torch.float64)
        assert torch.isfinite(result["cka"])

    def test_custom_keys(self, run_estimator):
        x = torch.randn(48, 8)
        y = torch.randn(48, 6)

        result = run_estimator(x, y, in_key_a="linear1", in_key_b="linear2", out_key="similarity")

        assert "linear1" in result
        assert "linear2" in result
        assert "similarity" in result
        assert result["similarity"].ndim == 0

    def test_identical_views_are_one(self, run_estimator):
        x = torch.randn(128, 16)

        result = run_estimator(x, x.clone())

        assert torch.isclose(result["cka"], torch.tensor(1.0), atol=1e-6)

    def test_invariant_to_isotropic_scaling(self, run_estimator):
        x = torch.randn(128, 16)
        y = 7.5 * x

        result = run_estimator(x, y)

        assert torch.isclose(result["cka"], torch.tensor(1.0), atol=1e-5)

    def test_invariant_to_orthogonal_rotation(self, run_estimator):
        x = torch.randn(128, 12)
        q, _ = torch.linalg.qr(torch.randn(12, 12))
        y = x @ q

        result = run_estimator(x, y)

        assert torch.isclose(result["cka"], torch.tensor(1.0), atol=1e-5)

    def test_independent_random_views_have_low_cka(self, run_estimator):
        x = torch.randn(512, 32)
        y = torch.randn(512, 24)

        result = run_estimator(x, y)

        assert result["cka"].item() < 0.2

    @pytest.mark.parametrize(
        ("x_shape", "y_shape"),
        [
            ((1, 10, 8), (1, 10, 6)),
            ((5, 10, 8), (5, 10, 6)),
            ((2, 3, 10, 8), (2, 3, 10, 6)),
        ],
        ids=["1x10", "5x10", "2x3x10"],
    )
    def test_batch_shape_preservation(self, run_estimator, x_shape, y_shape):
        x = torch.randn(*x_shape)
        y = torch.randn(*y_shape)
        batch_size = x_shape[:-2]

        result = run_estimator(x, y, batch_size=batch_size)

        assert result["cka"].shape == batch_size

    def test_empty_flattened_batch_returns_empty_output(self, run_estimator):
        x = torch.randn(2, 0, 10, 8)
        y = torch.randn(2, 0, 10, 6)

        result = run_estimator(x, y, batch_size=[2, 0])

        assert result["cka"].shape == (2, 0)
        assert result["cka"].dtype == torch.float32
        assert result["cka"].numel() == 0

    def test_mismatched_sample_counts_raise(self, run_estimator):
        with pytest.raises(ValueError, match="matching sample counts"):
            run_estimator(torch.randn(32, 8), torch.randn(31, 6))

    def test_mismatched_batch_shapes_raise(self, run_estimator):
        with pytest.raises(ValueError, match="matching batch shapes"):
            run_estimator(torch.randn(2, 3, 16, 8), torch.randn(2, 4, 16, 6), batch_size=[2])

    def test_invalid_rank_raises(self, run_estimator):
        with pytest.raises(ValueError, match=r"shape \(N, D\) or \(\.\.\., N, D\)"):
            run_estimator(torch.randn(32), torch.randn(32))

    def test_mismatched_devices_raise(self, run_estimator):
        x = torch.randn(32, 8)
        y = torch.randn(32, 6, device="meta")

        with pytest.raises(ValueError, match="same device"):
            run_estimator(x, y)

    def test_constant_representation_returns_nan(self, run_estimator):
        x = torch.ones(64, 8)
        y = torch.randn(64, 6)

        result = run_estimator(x, y)

        assert torch.isnan(result["cka"])

    def test_integer_inputs_are_promoted_to_float32(self, run_estimator):
        x = torch.arange(512, dtype=torch.int64).reshape(128, 4)

        result = run_estimator(x, x.clone())

        assert result["cka"].dtype == torch.float32
        assert torch.isclose(result["cka"], torch.tensor(1.0, dtype=torch.float32), atol=1e-6)

    def test_determinism(self, run_estimator):
        x = torch.randn(96, 10)
        y = torch.randn(96, 7)

        r1 = run_estimator(x.clone(), y.clone())["cka"]
        r2 = run_estimator(x.clone(), y.clone())["cka"]

        assert torch.allclose(r1, r2, equal_nan=True)

    def test_repr(self):
        est = CkaEstimator()
        r = repr(est)

        assert "CkaEstimator" in r
        assert "in_keys=['data_a', 'data_b']" in r
        assert "out_keys=['cka']" in r
        assert "kernel='linear'" in r
        assert "eps=" in r

    def test_unknown_kernel_raises(self):
        with pytest.raises(NotImplementedError, match="Only 'linear' is implemented"):
            CkaEstimator(kernel="rbf")
