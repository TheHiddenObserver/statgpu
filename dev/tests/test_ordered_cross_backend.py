"""Cross-backend precision regression tests for Ordered models.

Tests OrderedLogitRegression and OrderedProbitRegression across
numpy (CPU), CuPy (GPU), and PyTorch (GPU) backends.
"""
import numpy as np
import pytest

from statgpu._config import set_device, Device
from statgpu.linear_model import OrderedLogitRegression, OrderedProbitRegression


def _has_cuda():
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except Exception:
        return False


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _make_data(n=200, p=10, k=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = rng.normal(0.3, 0.5, size=p).astype(np.float64)
    eta = X @ beta + rng.normal(0, 0.5, size=n)
    y = np.digitize(eta, bins=np.percentile(eta, np.linspace(0, 100, k)[1:-1]))
    return X, y.astype(np.int32)


class TestOrderedLogit:
    """OrderedLogitRegression cross-backend regression tests."""

    def test_cpu_fits_without_crash(self):
        X, y = _make_data()
        model = OrderedLogitRegression(n_categories=3, device="cpu", max_iter=200)
        model.fit(X, y)

        assert model.coef_ is not None
        assert model.coef_.shape == (X.shape[1],)
        assert model.thresholds_ is not None
        assert model.n_iter_ is not None
        assert np.all(np.isfinite(model.coef_))
        assert model.n_iter_ > 0 and model.n_iter_ <= 200
        # thresholds_ has -inf/+inf boundaries; interior values must be finite
        thresh = model.thresholds_
        assert thresh[0] < -1e100 and thresh[-1] > 1e100
        assert np.all(np.isfinite(thresh[1:-1]))

    def test_cpu_reproducible(self):
        X, y = _make_data()
        m1 = OrderedLogitRegression(n_categories=3, device="cpu", max_iter=200)
        m1.fit(X, y)
        m2 = OrderedLogitRegression(n_categories=3, device="cpu", max_iter=200)
        m2.fit(X, y)

        assert np.allclose(m1.coef_, m2.coef_, rtol=1e-12, atol=1e-12)
        assert np.allclose(m1.thresholds_, m2.thresholds_, rtol=1e-12, atol=1e-12)

    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
    def test_cpu_vs_cupy(self):
        X, y = _make_data()

        m_cpu = OrderedLogitRegression(n_categories=3, device="cpu", max_iter=200)
        m_cpu.fit(X, y)

        m_gpu = OrderedLogitRegression(n_categories=3, device="cuda", max_iter=200)
        m_gpu.fit(X, y)

        # Coefficients: tight tolerance
        assert np.allclose(m_cpu.coef_, m_gpu.coef_, rtol=1e-2, atol=1e-2)

        # Thresholds: looser (optimization paths can vary)
        assert np.allclose(m_cpu.thresholds_[1:-1], m_gpu.thresholds_[1:-1],
                           rtol=0.5, atol=0.5)

        # Prediction parity: same class assignments
        pred_cpu = m_cpu.predict(X)
        pred_gpu = m_gpu.predict(X)
        assert np.array_equal(pred_cpu, pred_gpu)

    @pytest.mark.skipif(not _has_torch_cuda(), reason="CUDA not available for torch")
    def test_cpu_vs_torch(self):
        X, y = _make_data()

        m_cpu = OrderedLogitRegression(n_categories=3, device="cpu", max_iter=200)
        m_cpu.fit(X, y)

        m_torch = OrderedLogitRegression(n_categories=3, device="torch", max_iter=200)
        m_torch.fit(X, y)

        # Coefficients: tight tolerance
        assert np.allclose(m_cpu.coef_, m_torch.coef_, rtol=1e-2, atol=1e-2)

        # Thresholds: looser (optimization paths can vary)
        assert np.allclose(m_cpu.thresholds_[1:-1], m_torch.thresholds_[1:-1],
                           rtol=0.5, atol=0.5)

        # Prediction parity: same class assignments
        pred_cpu = m_cpu.predict(X)
        pred_torch = m_torch.predict(X)
        assert np.array_equal(pred_cpu, pred_torch)

    @pytest.mark.skipif(not _has_torch_cuda(), reason="CUDA not available for torch")
    def test_torch_n_iter_not_max(self):
        X, y = _make_data()

        m_torch = OrderedLogitRegression(n_categories=3, device="torch", max_iter=200)
        m_torch.fit(X, y)

        assert m_torch.n_iter_ < m_torch.max_iter, (
            f"n_iter_={m_torch.n_iter_} equals max_iter={m_torch.max_iter} — "
            f"likely reporting bogus value"
        )


class TestOrderedProbit:
    """OrderedProbitRegression cross-backend regression tests."""

    def test_cpu_fits_without_crash(self):
        X, y = _make_data()
        model = OrderedProbitRegression(n_categories=3, device="cpu", max_iter=200)
        model.fit(X, y)

        assert model.coef_ is not None
        assert model.coef_.shape == (X.shape[1],)
        assert model.thresholds_ is not None
        assert model.n_iter_ is not None
        assert np.all(np.isfinite(model.coef_))
        assert model.n_iter_ > 0 and model.n_iter_ <= 200
        # thresholds_ has -inf/+inf boundaries; interior values must be finite
        thresh = model.thresholds_
        assert thresh[0] < -1e100 and thresh[-1] > 1e100
        assert np.all(np.isfinite(thresh[1:-1]))

    def test_cpu_reproducible(self):
        X, y = _make_data()
        m1 = OrderedProbitRegression(n_categories=3, device="cpu", max_iter=200)
        m1.fit(X, y)
        m2 = OrderedProbitRegression(n_categories=3, device="cpu", max_iter=200)
        m2.fit(X, y)

        assert np.allclose(m1.coef_, m2.coef_, rtol=1e-12, atol=1e-12)
        assert np.allclose(m1.thresholds_, m2.thresholds_, rtol=1e-12, atol=1e-12)

    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
    def test_cpu_vs_cupy(self):
        X, y = _make_data()

        m_cpu = OrderedProbitRegression(n_categories=3, device="cpu", max_iter=200)
        m_cpu.fit(X, y)

        m_gpu = OrderedProbitRegression(n_categories=3, device="cuda", max_iter=200)
        m_gpu.fit(X, y)

        # Coefficients: tight tolerance
        assert np.allclose(m_cpu.coef_, m_gpu.coef_, rtol=1e-2, atol=1e-2)

        # Thresholds: looser (optimization paths can vary)
        assert np.allclose(m_cpu.thresholds_[1:-1], m_gpu.thresholds_[1:-1],
                           rtol=0.5, atol=0.5)

        # Prediction parity: same class assignments
        pred_cpu = m_cpu.predict(X)
        pred_gpu = m_gpu.predict(X)
        assert np.array_equal(pred_cpu, pred_gpu)

    @pytest.mark.skipif(not _has_torch_cuda(), reason="CUDA not available for torch")
    def test_cpu_vs_torch(self):
        X, y = _make_data()

        m_cpu = OrderedProbitRegression(n_categories=3, device="cpu", max_iter=200)
        m_cpu.fit(X, y)

        m_torch = OrderedProbitRegression(n_categories=3, device="torch", max_iter=200)
        m_torch.fit(X, y)

        # Coefficients: tight tolerance
        assert np.allclose(m_cpu.coef_, m_torch.coef_, rtol=1e-2, atol=1e-2)

        # Thresholds: looser (optimization paths can vary)
        assert np.allclose(m_cpu.thresholds_[1:-1], m_torch.thresholds_[1:-1],
                           rtol=0.5, atol=0.5)

        # Prediction parity: same class assignments
        pred_cpu = m_cpu.predict(X)
        pred_torch = m_torch.predict(X)
        assert np.array_equal(pred_cpu, pred_torch)

    @pytest.mark.skipif(not _has_torch_cuda(), reason="CUDA not available for torch")
    def test_torch_n_iter_not_max(self):
        X, y = _make_data()

        m_torch = OrderedProbitRegression(n_categories=3, device="torch", max_iter=200)
        m_torch.fit(X, y)

        assert m_torch.n_iter_ < m_torch.max_iter, (
            f"n_iter_={m_torch.n_iter_} equals max_iter={m_torch.max_iter} — "
            f"likely reporting bogus value"
        )
