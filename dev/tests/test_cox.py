"""Tests for Cox Proportional Hazards model."""

import numpy as np
import pytest

from statgpu.survival import CoxPH
from statgpu._config import set_device, Device


def _make_survival_data(n_samples=200, n_features=5, seed=42):
    """Generate synthetic survival dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.array([0.5, -0.3, 0.8, -0.2, 0.4], dtype=np.float64)[:n_features]
    hazard = np.exp(X @ beta)
    time = rng.exponential(1.0 / hazard)
    event = rng.binomial(1, 0.7, size=n_samples).astype(np.int32)
    return X.astype(np.float64), time.astype(np.float64), event


class TestCoxPH:
    """CPU tests for CoxPH."""

    @pytest.mark.parametrize("ties", ["breslow", "efron"])
    def test_basic_fit_cpu(self, ties):
        """Model fits and exposes key outputs on CPU."""
        set_device("cpu")
        X, time, event = _make_survival_data()

        model = CoxPH(ties=ties, device="cpu", max_iter=50)
        model.fit(X, time, event)

        assert model.coef_ is not None
        assert model.hazard_ratios_ is not None
        assert model._iterations <= 50
        assert model._log_likelihood is not None
        assert np.isfinite(model._log_likelihood)

    def test_compute_inference_false_cpu(self):
        """Inference outputs can be disabled cleanly."""
        X, time, event = _make_survival_data(seed=7)
        model = CoxPH(device="cpu", compute_inference=False, max_iter=50)
        model.fit(X, time, event)

        assert model._bse is None
        assert model._conf_int is None
        assert model._baseline_hazard is None
        assert np.isfinite(model._log_likelihood)

    def test_explicit_torch_without_cuda_raises(self, monkeypatch):
        """device='torch' should require Torch CUDA, not fall back to Torch CPU."""
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        X, time, event = _make_survival_data(n_samples=40, n_features=3, seed=12)

        model = CoxPH(device="torch", compute_inference=False, max_iter=1)

        with pytest.raises(RuntimeError, match="device='torch'"):
            model.fit(X, time, event)

    def test_coxph_exposes_torch_cleanup_hook(self):
        """CoxPH follows the shared GPU cleanup hook contract."""
        model = CoxPH(gpu_memory_cleanup=True)

        assert callable(model._cleanup_torch_memory)
        assert callable(model.__del__)

    @pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
    def test_cov_type_inference_cpu(self, cov_type):
        """CoxPH supports nonrobust/hc0/hc1 covariance on CPU."""
        set_device("cpu")
        X, time, event = _make_survival_data(n_samples=220, n_features=5, seed=21)
        model = CoxPH(device="cpu", cov_type=cov_type, compute_inference=True, max_iter=60)
        model.fit(X, time, event)
        assert model._bse is not None
        assert model._conf_int is not None
        assert np.all(np.isfinite(model._bse))
        assert np.all(np.isfinite(model._pvalues))

    def test_cov_type_cluster_cpu(self):
        """Cluster-robust covariance works when cluster ids are provided."""
        set_device("cpu")
        X, time, event = _make_survival_data(n_samples=240, n_features=5, seed=31)
        cluster = np.repeat(np.arange(40), 6)
        model = CoxPH(device="cpu", cov_type="cluster", compute_inference=True, max_iter=60)
        model.fit(X, time, event, cluster=cluster)
        assert model._bse is not None
        assert np.all(np.isfinite(model._bse))

    def test_entry_supported_cpu(self):
        """entry should be supported in CPU fit path."""
        set_device("cpu")
        X, time, event = _make_survival_data(n_samples=120, n_features=4, seed=77)
        entry = np.zeros_like(time, dtype=np.float64)
        entry[:40] = np.minimum(time[:40] * 0.3, time[:40] * 0.95)
        model = CoxPH(device="cpu", max_iter=20)
        model.fit(X, time, event, entry=entry)
        assert model.coef_ is not None
        assert model._log_likelihood is not None


class TestGPU:
    """GPU-specific tests for CoxPH (run only when CUDA available)."""

    @pytest.mark.skipif(
        not CoxPH(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    @pytest.mark.parametrize("ties", ["breslow", "efron"])
    def test_gpu_matches_cpu(self, ties):
        """GPU and CPU coefficients/loglikelihood are numerically close."""
        np.random.seed(42)
        X, time, event = _make_survival_data(n_samples=160, n_features=5, seed=42)

        # CPU reference
        model_cpu = CoxPH(
            ties=ties,
            device="cpu",
            max_iter=60,
            compute_inference=False,
        )
        model_cpu.fit(X, time, event)

        # GPU run
        model_gpu = CoxPH(
            ties=ties,
            device="cuda",
            max_iter=60,
            compute_inference=False,
        )
        model_gpu.fit(X, time, event)

        assert np.allclose(model_cpu.coef_, model_gpu.coef_, rtol=5e-3, atol=1e-4)
        assert np.allclose(
            model_cpu._log_likelihood,
            model_gpu._log_likelihood,
            rtol=5e-3,
            atol=1e-4,
        )

    @pytest.mark.skipif(
        not CoxPH(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_cov_type_inference_gpu(self):
        """CoxPH cov_type runs in CUDA fit path with inference enabled."""
        X, time, event = _make_survival_data(n_samples=180, n_features=5, seed=52)
        model = CoxPH(device="cuda", cov_type="hc1", compute_inference=True, max_iter=50)
        model.fit(X, time, event)
        assert model._bse is not None
        assert np.all(np.isfinite(model._bse))

    @pytest.mark.skipif(
        not CoxPH(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_entry_supported_gpu(self):
        """entry should be supported for CUDA calls via CPU fallback path."""
        X, time, event = _make_survival_data(n_samples=128, n_features=4, seed=88)
        entry = np.zeros_like(time, dtype=np.float64)
        entry[:30] = np.minimum(time[:30] * 0.25, time[:30] * 0.95)
        model = CoxPH(device="cuda", max_iter=20)
        model.fit(X, time, event, entry=entry)
        assert model.coef_ is not None
