"""
Tests for intrinsic dimension estimation.
"""

import importlib.util
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score
from tensordict import TensorDict

from tdhook.latent.dimension_estimation import (
    CaPcaDimensionEstimator,
    LocalKnnDimensionEstimator,
    LocalPcaDimensionEstimator,
    TwoNnDimensionEstimator,
)
from tdhook.latent.dimension_estimation.local_pca import (
    _dim_from_eigenvalues_maxgap,
    _dim_from_eigenvalues_ratio,
    _local_pca,
)


@pytest.fixture
def plane_data():
    """2D manifold embedded in 10D (last 8 dims zero)."""
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    data[:, 2:] = 0
    return data


@pytest.fixture
def circle_data():
    """1D manifold (circle) embedded in 2D."""
    torch.manual_seed(42)
    theta = torch.rand(100) * 2 * torch.pi
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


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

    def test_known_dimension_2d(self, run_estimator, plane_data):
        """Test on 2D manifold embedded in higher space."""
        result = run_estimator(plane_data)
        d = result["dimension"].item()
        assert 1.5 < d < 2.5

    def test_known_dimension_circle(self, run_estimator, circle_data):
        """Test on 1D manifold (circle) embedded in 2D."""
        result = run_estimator(circle_data, return_xy=True)
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

    def test_duplicate_points_raises(self, run_estimator):
        """Test that all-duplicate points raises."""
        with pytest.raises(ValueError, match="At least 3 valid points"):
            run_estimator(torch.ones(5, 3))

    def test_determinism(self, run_estimator):
        """Test that same input yields same output."""
        torch.manual_seed(42)
        data = torch.randn(80, 6)
        r1 = run_estimator(data.clone())["dimension"]
        r2 = run_estimator(data.clone())["dimension"]
        assert torch.allclose(r1, r2)

    def test_repr(self):
        """Test __repr__ includes class name, in_keys, out_keys, and eps."""
        est = TwoNnDimensionEstimator()
        r = repr(est)
        assert "TwoNnDimensionEstimator" in r
        assert "in_keys=['data']" in r
        assert "out_keys=['dimension']" in r
        assert "eps=" in r


@pytest.fixture
def run_local_knn_estimator():
    torch.manual_seed(42)

    def _run(data, k=2, in_key="data", batch_size=None, **estimator_kwargs):
        if batch_size is None:
            batch_size = [] if data.ndim == 2 else data.shape[:-2]
        td = TensorDict({in_key: data}, batch_size=batch_size)
        return LocalKnnDimensionEstimator(k=k, in_key=in_key, **estimator_kwargs)(td)

    return _run


class TestLocalKnnDimensionEstimator:
    """Test the LocalKnnDimensionEstimator class."""

    def test_default_keys(self, run_local_knn_estimator):
        """Test with default in_key and out_key."""
        data = torch.randn(50, 10)
        result = run_local_knn_estimator(data, k=2)
        assert "dimension" in result
        assert result["dimension"].shape == (50,)
        assert result["dimension"].dtype in (torch.float32, torch.float64)
        valid = torch.isfinite(result["dimension"])
        assert valid.sum() > 0
        assert (result["dimension"][valid] > 0).all()

    def test_custom_keys(self, run_local_knn_estimator):
        """Test with custom in_key and out_key."""
        data = torch.randn(50, 8)
        result = run_local_knn_estimator(data, k=2, in_key="linear2", out_key="intrinsic_dim")
        assert "intrinsic_dim" in result
        assert "linear2" in result
        assert result["intrinsic_dim"].shape == (50,)

    def test_output_shape(self, run_local_knn_estimator):
        """Test output shape (N,) for (N, D) input."""
        data = torch.randn(100, 5)
        result = run_local_knn_estimator(data, k=3)
        assert result["dimension"].shape == (100,)

    def test_known_dimension_2d(self, run_local_knn_estimator, plane_data):
        """Test on 2D manifold embedded in higher space."""
        result = run_local_knn_estimator(plane_data, k=5)
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 1.0 < mean_d < 5.0

    def test_known_dimension_circle(self, run_local_knn_estimator, circle_data):
        """Test on 1D manifold (circle) embedded in 2D."""
        result = run_local_knn_estimator(circle_data, k=5)
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 0.3 < mean_d < 2.5

    @pytest.mark.parametrize(
        "shape",
        [(1, 10, 8), (5, 10, 8), (2, 3, 10, 4)],
        ids=["1x10x8", "5x10x8", "2x3x10x4"],
    )
    def test_batch_shape_preservation(self, run_local_knn_estimator, shape):
        """Test that (..., N, D) preserves batch shape, output is (..., N)."""
        data = torch.randn(*shape)
        batch_size = shape[:-2]
        N = shape[-2]
        result = run_local_knn_estimator(data, k=2, batch_size=batch_size)
        assert result["dimension"].shape == (*batch_size, N)

    def test_too_few_points_raises(self, run_local_knn_estimator):
        """Test that N < 2k+1 raises."""
        with pytest.raises(ValueError, match="At least 2k\\+1 points"):
            run_local_knn_estimator(torch.randn(4, 5), k=2)  # 4 < 2*2+1

    def test_k_validation(self):
        """Test that invalid k raises (k <= 1 or wrong type)."""
        with pytest.raises(ValueError, match="k must be greater than 1"):
            LocalKnnDimensionEstimator(k=1)
        with pytest.raises(TypeError, match="k must be an int or 'auto'"):
            LocalKnnDimensionEstimator(k=2.5)

    def test_determinism(self, run_local_knn_estimator):
        """Test that same input yields same output."""
        torch.manual_seed(42)
        data = torch.randn(80, 6)
        r1 = run_local_knn_estimator(data.clone(), k=2)["dimension"]
        r2 = run_local_knn_estimator(data.clone(), k=2)["dimension"]
        assert torch.allclose(r1, r2, equal_nan=True)

    def test_k_auto(self, run_local_knn_estimator, plane_data):
        """Test k='auto' uses n**0.5."""
        result = run_local_knn_estimator(plane_data, k="auto")
        assert "dimension" in result
        assert result["dimension"].shape == (100,)

    def test_repr(self):
        """Test __repr__ includes class name, in_keys, out_keys, k, and eps."""
        est = LocalKnnDimensionEstimator(k=3)
        r = repr(est)
        assert "LocalKnnDimensionEstimator" in r
        assert "in_keys=['data']" in r
        assert "out_keys=['dimension']" in r
        assert "k=3" in r
        assert "eps=" in r


@pytest.fixture
def run_local_pca_estimator():
    torch.manual_seed(42)

    def _run(data, k=5, in_key="data", batch_size=None, **estimator_kwargs):
        if batch_size is None:
            batch_size = [] if data.ndim == 2 else data.shape[:-2]
        td = TensorDict({in_key: data}, batch_size=batch_size)
        return LocalPcaDimensionEstimator(k=k, in_key=in_key, **estimator_kwargs)(td)

    return _run


class TestLocalPcaDimensionEstimator:
    """Test the LocalPcaDimensionEstimator class."""

    def test_default_keys(self, run_local_pca_estimator):
        """Test with default in_key and out_key."""
        data = torch.randn(50, 10)
        result = run_local_pca_estimator(data, k=5)
        assert "dimension" in result
        assert result["dimension"].shape == (50,)
        assert result["dimension"].dtype in (torch.float32, torch.float64)
        valid = torch.isfinite(result["dimension"])
        assert valid.sum() > 0
        assert (result["dimension"][valid] >= 1).all()

    def test_custom_keys(self, run_local_pca_estimator):
        """Test with custom in_key and out_key."""
        data = torch.randn(50, 8)
        result = run_local_pca_estimator(data, k=5, in_key="linear2", out_key="intrinsic_dim")
        assert "intrinsic_dim" in result
        assert "linear2" in result
        assert result["intrinsic_dim"].shape == (50,)

    def test_output_shape(self, run_local_pca_estimator):
        """Test output shape (N,) for (N, D) input."""
        data = torch.randn(100, 5)
        result = run_local_pca_estimator(data, k=5)
        assert result["dimension"].shape == (100,)

    def test_known_dimension_2d_maxgap(self, run_local_pca_estimator, plane_data):
        """Test on 2D manifold with maxgap criterion."""
        result = run_local_pca_estimator(plane_data, k=5, criterion="maxgap")
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 1.0 < mean_d < 5.0

    def test_known_dimension_2d_ratio(self, run_local_pca_estimator, plane_data):
        """Test on 2D manifold with ratio criterion."""
        result = run_local_pca_estimator(plane_data, k=5, criterion="ratio")
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 1.0 < mean_d < 5.0

    def test_known_dimension_circle(self, run_local_pca_estimator, circle_data):
        """Test on 1D manifold (circle) embedded in 2D."""
        result = run_local_pca_estimator(circle_data, k=5)
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 0.5 < mean_d < 3.0

    @pytest.mark.parametrize(
        "shape",
        [(1, 10, 8), (5, 10, 8), (2, 3, 10, 4)],
        ids=["1x10x8", "5x10x8", "2x3x10x4"],
    )
    def test_batch_shape_preservation(self, run_local_pca_estimator, shape):
        """Test that (..., N, D) preserves batch shape, output is (..., N)."""
        data = torch.randn(*shape)
        batch_size = shape[:-2]
        N = shape[-2]
        result = run_local_pca_estimator(data, k=5, batch_size=batch_size)
        assert result["dimension"].shape == (*batch_size, N)

    def test_too_few_points_raises(self, run_local_pca_estimator):
        """Test that N < k+1 raises."""
        with pytest.raises(ValueError, match="At least k\\+1 points"):
            run_local_pca_estimator(torch.randn(5, 5), k=5)  # 5 < 5+1

    def test_k_validation(self):
        """Test that invalid k raises (k < 1 or wrong type)."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            LocalPcaDimensionEstimator(k=0)
        with pytest.raises(TypeError, match="k must be an int or 'auto'"):
            LocalPcaDimensionEstimator(k=2.5)

    def test_determinism(self, run_local_pca_estimator):
        """Test that same input yields same output."""
        torch.manual_seed(42)
        data = torch.randn(80, 6)
        r1 = run_local_pca_estimator(data.clone(), k=5)["dimension"]
        r2 = run_local_pca_estimator(data.clone(), k=5)["dimension"]
        assert torch.allclose(r1, r2, equal_nan=True)

    def test_k_auto(self, run_local_pca_estimator, plane_data):
        """Test k='auto' uses n**0.5."""
        result = run_local_pca_estimator(plane_data, k="auto")
        assert "dimension" in result
        assert result["dimension"].shape == (100,)

    def test_repr(self):
        """Test __repr__ includes class name, in_keys, out_keys, k, criterion, and eps."""
        est = LocalPcaDimensionEstimator(k=3, criterion="maxgap")
        r = repr(est)
        assert "LocalPcaDimensionEstimator" in r
        assert "in_keys=['data']" in r
        assert "out_keys=['dimension']" in r
        assert "k=3" in r
        assert "criterion='maxgap'" in r
        assert "eps=" in r

    def test_sklearn_missing_raises(self):
        """Test that ImportError is raised when sklearn is not installed."""
        with patch.object(importlib.util, "find_spec", return_value=None):
            est = LocalPcaDimensionEstimator(k=5)
            td = TensorDict({"data": torch.randn(20, 5)}, batch_size=[])
            with pytest.raises(ImportError, match="scikit-learn"):
                est(td)

    def test_unknown_criterion_raises(self, run_local_pca_estimator):
        """Test that unknown criterion raises ValueError."""
        data = torch.randn(20, 5)
        est = LocalPcaDimensionEstimator(k=5, criterion="invalid")
        td = TensorDict({"data": data}, batch_size=[])
        with pytest.raises(ValueError, match="Unknown criterion"):
            est(td)

    def test_maxgap_single_eigenvalue(self, run_local_pca_estimator):
        """Test maxgap with 1D data (single eigenvalue) returns 1."""
        data = torch.randn(15, 1)
        result = run_local_pca_estimator(data, k=2, criterion="maxgap")
        assert (result["dimension"] >= 1).all()
        assert result["dimension"].shape == (15,)

    def test_ratio_empty_eigenvalues(self):
        """Test ratio criterion with empty eigenvalues returns 1."""
        assert _dim_from_eigenvalues_ratio(np.array([]), alpha=0.05) == 1

    def test_maxgap_few_eigenvalues(self):
        """Test maxgap with single eigenvalue returns 1."""
        assert _dim_from_eigenvalues_maxgap(np.array([1.0])) == 1
        assert _dim_from_eigenvalues_maxgap(np.array([])) == 1

    def test_constant_data_handled(self, run_local_pca_estimator):
        """Test constant data (zero variance) does not crash."""
        data = torch.ones(10, 5)
        result = run_local_pca_estimator(data, k=2)
        assert result["dimension"].shape == (10,)
        assert torch.isfinite(result["dimension"]).all()

    def test_few_neighborhood_points_returns_nan(self):
        """Test that k=0 (single-point neighborhood) returns nan."""
        from sklearn.decomposition import PCA

        data = torch.randn(5, 3)
        result = _local_pca(data, k=0, eps=1e-5, criterion="maxgap", alpha=0.05, pca_cls=PCA)
        assert result.shape == (5,)
        assert torch.isnan(result).all()

    def test_empty_eigenvalues_returns_one(self):
        """Test that PCA returning empty eigenvalues yields 1.0."""

        class MockPCA:
            def __init__(self, n_components=None):
                pass

            def fit(self, X):
                self.explained_variance_ = np.array([])
                return self

        data = torch.randn(5, 3)
        result = _local_pca(data, k=2, eps=1e-5, criterion="maxgap", alpha=0.05, pca_cls=MockPCA)
        assert result.shape == (5,)
        assert (result == 1.0).all()


@pytest.fixture
def run_ca_pca_estimator():
    torch.manual_seed(42)

    def _run(data, k=5, in_key="data", batch_size=None, **estimator_kwargs):
        if batch_size is None:
            batch_size = [] if data.ndim == 2 else data.shape[:-2]
        td = TensorDict({in_key: data}, batch_size=batch_size)
        return CaPcaDimensionEstimator(k=k, in_key=in_key, **estimator_kwargs)(td)

    return _run


class TestCaPcaDimensionEstimator:
    """Test the CaPcaDimensionEstimator class."""

    def test_default_keys(self, run_ca_pca_estimator):
        """Test with default in_key and out_key."""
        data = torch.randn(50, 10)
        result = run_ca_pca_estimator(data, k=5)
        assert "dimension" in result
        assert result["dimension"].shape == (50,)
        assert result["dimension"].dtype in (torch.float32, torch.float64)
        valid = torch.isfinite(result["dimension"])
        assert valid.sum() > 0
        assert (result["dimension"][valid] >= 1).all()

    def test_custom_keys(self, run_ca_pca_estimator):
        """Test with custom in_key and out_key."""
        data = torch.randn(50, 8)
        result = run_ca_pca_estimator(data, k=5, in_key="linear2", out_key="intrinsic_dim")
        assert "intrinsic_dim" in result
        assert "linear2" in result
        assert result["intrinsic_dim"].shape == (50,)

    def test_output_shape(self, run_ca_pca_estimator):
        """Test output shape (N,) for (N, D) input."""
        data = torch.randn(100, 5)
        result = run_ca_pca_estimator(data, k=5)
        assert result["dimension"].shape == (100,)

    def test_known_dimension_2d(self, run_ca_pca_estimator, plane_data):
        """Test on 2D manifold embedded in higher space."""
        result = run_ca_pca_estimator(plane_data, k=5)
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 1.0 < mean_d < 5.0

    def test_known_dimension_circle(self, run_ca_pca_estimator, circle_data):
        """Test on 1D manifold (circle) embedded in 2D."""
        result = run_ca_pca_estimator(circle_data, k=5)
        d = result["dimension"]
        valid = torch.isfinite(d)
        mean_d = d[valid].mean().item()
        assert 0.5 < mean_d < 3.0

    @pytest.mark.parametrize(
        "shape",
        [(1, 10, 8), (5, 10, 8), (2, 3, 10, 4)],
        ids=["1x10x8", "5x10x8", "2x3x10x4"],
    )
    def test_batch_shape_preservation(self, run_ca_pca_estimator, shape):
        """Test that (..., N, D) preserves batch shape, output is (..., N)."""
        data = torch.randn(*shape)
        batch_size = shape[:-2]
        N = shape[-2]
        result = run_ca_pca_estimator(data, k=5, batch_size=batch_size)
        assert result["dimension"].shape == (*batch_size, N)

    def test_too_few_points_raises(self, run_ca_pca_estimator):
        """Test that N < k+2 raises."""
        with pytest.raises(ValueError, match="At least k\\+2 points"):
            run_ca_pca_estimator(torch.randn(6, 5), k=5)  # 6 < 5+2

    def test_k_validation(self):
        """Test that invalid k raises (k < 1 or wrong type)."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            CaPcaDimensionEstimator(k=0)
        with pytest.raises(TypeError, match="k must be an int or 'auto'"):
            CaPcaDimensionEstimator(k=2.5)

    def test_determinism(self, run_ca_pca_estimator):
        """Test that same input yields same output."""
        torch.manual_seed(42)
        data = torch.randn(80, 6)
        r1 = run_ca_pca_estimator(data.clone(), k=5)["dimension"]
        r2 = run_ca_pca_estimator(data.clone(), k=5)["dimension"]
        assert torch.allclose(r1, r2, equal_nan=True)

    def test_k_auto(self, run_ca_pca_estimator, plane_data):
        """Test k='auto' uses n**0.5."""
        result = run_ca_pca_estimator(plane_data, k="auto")
        assert "dimension" in result
        assert result["dimension"].shape == (100,)

    def test_repr(self):
        """Test __repr__ includes class name, in_keys, out_keys, k, and eps."""
        est = CaPcaDimensionEstimator(k=3)
        r = repr(est)
        assert "CaPcaDimensionEstimator" in r
        assert "in_keys=['data']" in r
        assert "out_keys=['dimension']" in r
        assert "k=3" in r
        assert "eps=" in r

    def test_sklearn_missing_raises(self):
        """Test that ImportError is raised when sklearn is not installed."""
        with patch.object(importlib.util, "find_spec", return_value=None):
            est = CaPcaDimensionEstimator(k=5)
            td = TensorDict({"data": torch.randn(20, 5)}, batch_size=[])
            with pytest.raises(ImportError, match="scikit-learn"):
                est(td)

    def test_constant_data_handled(self, run_ca_pca_estimator):
        """Test constant data (all duplicates) returns nan without crashing."""
        data = torch.ones(10, 5)
        result = run_ca_pca_estimator(data, k=5)
        assert result["dimension"].shape == (10,)
        assert torch.isnan(result["dimension"]).all()

    def test_k1_returns_nan(self, run_ca_pca_estimator):
        """Test k=1 (single-point neighborhood) returns nan."""
        data = torch.randn(10, 5)
        result = run_ca_pca_estimator(data, k=1)
        assert result["dimension"].shape == (10,)
        assert torch.isnan(result["dimension"]).all()
