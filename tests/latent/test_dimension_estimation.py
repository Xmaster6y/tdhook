"""
Tests for intrinsic dimension estimation.
"""

import pytest
import torch
from sklearn.metrics import r2_score
from tensordict import TensorDict

from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator


@pytest.fixture
def run_estimator():
    torch.manual_seed(42)

    def _run(data, in_key="data", batch_size=None, **estimator_kwargs):
        if batch_size is None:
            batch_size = [] if data.ndim == 2 else data.shape[:-2]
        td = TensorDict({in_key: data}, batch_size=batch_size)
        return TwoNnDimensionEstimator(in_key=in_key, **estimator_kwargs)(td)

    return _run


class TestTwoNnDimensionEstimator:
    """Test the TwoNnDimensionEstimator class."""

    def test_default_keys(self, run_estimator):
        """Test with default in_key and out_key."""
        data = torch.randn(100, 10)
        result = run_estimator(data)
        assert "dimension" in result
        assert result["dimension"].ndim == 0
        assert result["dimension"].dtype in (torch.float32, torch.float64)
        assert result["dimension"].item() > 0

    def test_custom_keys(self, run_estimator):
        """Test with custom in_key and out_key."""
        data = torch.randn(50, 8)
        result = run_estimator(data, in_key="linear2", out_key="intrinsic_dim")
        assert "intrinsic_dim" in result
        assert "linear2" in result
        assert result["intrinsic_dim"].ndim == 0

    def test_return_xy(self, run_estimator):
        """Test that return_xy writes _x and _y keys."""
        data = torch.randn(50, 5)
        result = run_estimator(data, return_xy=True)
        assert "dimension" in result
        assert "dimension_x" in result
        assert "dimension_y" in result
        x, y = result["dimension_x"], result["dimension_y"]
        assert x.shape == y.shape
        assert len(x) >= 3
        d = result["dimension"].item()
        # y ≈ d * x (through origin), so slope of regression is d
        assert d > 0

    def test_known_dimension_2d(self, run_estimator):
        """Test on 2D manifold embedded in higher space."""
        data = torch.randn(100, 10)
        data[:, 2:] = 0
        result = run_estimator(data)
        d = result["dimension"].item()
        assert 1.5 < d < 2.5

    def test_known_dimension_circle(self, run_estimator):
        """Test on 1D manifold (circle) embedded in 2D."""
        theta = torch.rand(100) * 2 * torch.pi
        data = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        result = run_estimator(data, return_xy=True)
        d = result["dimension"].item()
        x, y = result["dimension_x"].numpy(), result["dimension_y"].numpy()
        r2 = r2_score(y, d * x)
        assert 0.5 < d < 1.5
        assert r2 > 0.9

    @pytest.mark.parametrize("return_xy", [True, False])
    @pytest.mark.parametrize(
        "shape",
        [(1, 5, 8), (10, 5, 8), (10, 10, 4), (2, 3, 5, 8)],
        ids=["1x5x8", "10x5x8", "10x10x4", "2x3x5x8"],
    )
    def test_batch_size_preservation(self, run_estimator, shape, return_xy):
        """Test that (..., N, D) preserves batch shape, including size-1 batch axes."""
        data = torch.randn(*shape)
        batch_size = shape[:-2]
        result = run_estimator(data, batch_size=batch_size, return_xy=return_xy)
        assert "dimension" in result
        assert result["dimension"].shape == batch_size
        assert (result["dimension"] > 0).all()
        if return_xy:
            assert result["dimension_x"].shape[: len(batch_size)] == batch_size
            assert result["dimension_y"].shape[: len(batch_size)] == batch_size

    def test_too_few_points_raises(self, run_estimator):
        """Test that fewer than 3 points raises."""
        with pytest.raises(ValueError, match="At least 3 points"):
            run_estimator(torch.randn(2, 5))

    def test_determinism(self, run_estimator):
        """Test that same input yields same output."""
        torch.manual_seed(123)
        data = torch.randn(80, 6)
        r1 = run_estimator(data.clone())["dimension"].item()
        r2 = run_estimator(data.clone())["dimension"].item()
        assert r1 == r2

    def test_repr(self):
        """Test __repr__ includes class name, in_keys, out_keys, and eps."""
        est = TwoNnDimensionEstimator()
        r = repr(est)
        assert "TwoNnDimensionEstimator" in r
        assert "in_keys=['data']" in r
        assert "out_keys=['dimension']" in r
        assert "eps=" in r
