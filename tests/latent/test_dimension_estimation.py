"""
Tests for intrinsic dimension estimation.
"""

import pytest
import torch
from tensordict import TensorDict

from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator


class TestTwoNnDimensionEstimator:
    """Test the TwoNnDimensionEstimator class."""

    def test_default_keys(self):
        """Test with default in_key and out_key."""
        torch.manual_seed(42)
        data = torch.randn(100, 10)
        td = TensorDict({"data": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator()
        result = estimator(td)
        assert "dimension" in result
        assert result["dimension"].ndim == 0
        assert result["dimension"].dtype in (torch.float32, torch.float64)
        assert result["dimension"].item() > 0

    def test_custom_keys(self):
        """Test with custom in_key and out_key."""
        torch.manual_seed(42)
        data = torch.randn(50, 8)
        td = TensorDict({"linear2": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator(in_key="linear2", out_key="intrinsic_dim")
        result = estimator(td)
        assert "intrinsic_dim" in result
        assert "linear2" in result
        assert result["intrinsic_dim"].ndim == 0

    def test_return_xy(self):
        """Test that return_xy writes _x and _y keys."""
        torch.manual_seed(42)
        data = torch.randn(50, 5)
        td = TensorDict({"data": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator(return_xy=True)
        result = estimator(td)
        assert "dimension" in result
        assert "dimension_x" in result
        assert "dimension_y" in result
        x, y = result["dimension_x"], result["dimension_y"]
        assert x.shape == y.shape
        assert len(x) >= 3
        d = result["dimension"].item()
        # y ≈ d * x (through origin), so slope of regression is d
        assert d > 0

    def test_known_dimension_2d(self):
        """Test on 2D manifold embedded in higher space."""
        torch.manual_seed(42)
        # Points on a 2D plane: only first 2 coords vary
        data = torch.randn(100, 10)
        data[:, 2:] = 0
        td = TensorDict({"data": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator()
        result = estimator(td)
        d = result["dimension"].item()
        # Should be close to 2 (with tolerance for finite sample)
        assert 1.0 < d < 5.0

    def test_known_dimension_3d(self):
        """Test on 3D manifold embedded in higher space."""
        torch.manual_seed(42)
        data = torch.randn(150, 20)
        data[:, 3:] = 0
        td = TensorDict({"data": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator()
        result = estimator(td)
        d = result["dimension"].item()
        assert 1.5 < d < 6.0

    def test_flattened_input(self):
        """Test that batched/sequential input is flattened correctly."""
        torch.manual_seed(42)
        # (batch, seq, features) -> flatten to (batch*seq, features)
        data = torch.randn(10, 5, 8)  # 50 points
        td = TensorDict({"data": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator()
        result = estimator(td)
        assert "dimension" in result
        assert result["dimension"].ndim == 0

    def test_too_few_points_raises(self):
        """Test that fewer than 3 points raises."""
        estimator = TwoNnDimensionEstimator()
        td = TensorDict({"data": torch.randn(2, 5)}, batch_size=[])
        with pytest.raises(ValueError, match="At least 3 points"):
            estimator(td)

    def test_determinism(self):
        """Test that same input yields same output."""
        torch.manual_seed(123)
        data = torch.randn(80, 6)
        td = TensorDict({"data": data}, batch_size=[])
        estimator = TwoNnDimensionEstimator()
        r1 = estimator(td.clone())["dimension"].item()
        r2 = estimator(td.clone())["dimension"].item()
        assert r1 == r2
