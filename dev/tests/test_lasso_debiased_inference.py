"""
Tests for debiased Lasso inference (Javanmard-Montanari / Zhang-Zhang).

Covers:
  - CPU path: shapes, signs, coverage, comparison with OLS on well-specified data
  - GPU path: CPU vs GPU consistency
  - summary() output
"""

import numpy as np
import pytest

from statgpu.linear_model import Lasso
from statgpu._config import cuda_available


def _make_sparse_regression(n=400, p=50, s=5, seed=42):
    """Generate sparse regression data with known true coefficients."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    support = rng.choice(p, size=s, replace=False)
    beta[support] = rng.normal(loc=0, scale=3.0, size=s)
    y = X @ beta + rng.normal(scale=1.0, size=n)
    return X, y, beta, support


class TestDebiasedInferenceCPU:
    """Tests for debiased Lasso inference on CPU."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y, self.beta, self.support = _make_sparse_regression(
            n=300, p=40, s=5, seed=123
        )

    def test_basic_shapes(self):
        m = Lasso(
            alpha=0.1, inference_method="debiased", device="cpu",
            compute_inference=True, fit_intercept=True,
            max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)

        n_params = 1 + self.X.shape[1]  # intercept + features
        assert m._bse is not None
        assert m._bse.shape == (n_params,)
        assert m._pvalues.shape == (n_params,)
        assert m._conf_int.shape == (n_params, 2)
        assert m._tvalues.shape == (n_params,)

    def test_shapes_no_intercept(self):
        m = Lasso(
            alpha=0.1, inference_method="debiased", device="cpu",
            compute_inference=True, fit_intercept=False,
            max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)

        n_params = self.X.shape[1]
        assert m._bse.shape == (n_params,)
        assert m._pvalues.shape == (n_params,)
        assert m._conf_int.shape == (n_params, 2)

    def test_bse_positive(self):
        m = Lasso(
            alpha=0.05, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        assert np.all(m._bse > 0), "All standard errors must be positive"

    def test_pvalues_in_range(self):
        m = Lasso(
            alpha=0.05, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        assert np.all(m._pvalues >= 0)
        assert np.all(m._pvalues <= 1)

    def test_ci_contains_estimate(self):
        m = Lasso(
            alpha=0.05, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        params = m._params
        for i in range(len(params)):
            assert m._conf_int[i, 0] <= params[i] <= m._conf_int[i, 1], (
                f"CI for param {i} does not contain the estimate"
            )

    def test_ci_lower_lt_upper(self):
        m = Lasso(
            alpha=0.05, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        assert np.all(m._conf_int[:, 0] < m._conf_int[:, 1])

    def test_large_coef_small_pvalue(self):
        """True nonzero coefficients should tend to have small p-values."""
        X, y, beta, support = _make_sparse_regression(n=500, p=30, s=3, seed=77)
        m = Lasso(
            alpha=0.02, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=1000, tol=1e-6, cpu_solver="fista",
        )
        m.fit(X, y)

        offset = 1 if m.fit_intercept else 0
        for j in support:
            idx = offset + j
            if abs(beta[j]) > 1.5:
                assert m._pvalues[idx] < 0.2, (
                    f"Large true coef beta[{j}]={beta[j]:.2f} should have small p-value, "
                    f"got {m._pvalues[idx]:.4f}"
                )

    def test_coverage_rate(self):
        """Across multiple seeds, the 95% CI should cover ~90-100% of true params."""
        coverage_counts = 0
        total = 0
        for seed in range(10):
            X, y, beta, _ = _make_sparse_regression(n=400, p=20, s=3, seed=seed + 200)
            m = Lasso(
                alpha=0.05, inference_method="debiased", device="cpu",
                compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
            )
            m.fit(X, y)
            offset = 1 if m.fit_intercept else 0
            for j in range(X.shape[1]):
                idx = offset + j
                if m._conf_int[idx, 0] <= beta[j] <= m._conf_int[idx, 1]:
                    coverage_counts += 1
                total += 1

        coverage_rate = coverage_counts / total
        assert coverage_rate >= 0.80, (
            f"Coverage rate {coverage_rate:.2f} is too low (expected >= 0.80 for 95% CI)"
        )

    def test_summary_runs(self, capsys):
        m = Lasso(
            alpha=0.1, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        m.summary()
        captured = capsys.readouterr()
        assert "Debiased Lasso" in captured.out
        assert "P>|z|" in captured.out

    def test_f_pvalue_infinite_fvalue_returns_zero(self):
        """Perfect-fit style edge case should map to near-zero F-test p-value."""
        m = Lasso(alpha=0.1, device="cpu")
        m.coef_ = np.array([1.0, -0.5], dtype=float)
        m._df_resid = 8
        # Keep the manually constructed state internally consistent:
        # 10 observations with 2 coefficients implies 8 residual dof.
        m._y = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float
        )
        m._resid = np.zeros_like(m._y)

        assert m.fvalue == np.inf
        assert m.f_pvalue == 0.0

    def test_debiased_vs_ols_low_dim(self):
        """In low-dimensional regime (n >> p, sparse), debiased should be close to OLS."""
        X, y, beta, _ = _make_sparse_regression(n=1000, p=10, s=3, seed=55)
        m_db = Lasso(
            alpha=0.01, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=1000, tol=1e-6, cpu_solver="fista",
        )
        m_db.fit(X, y)

        m_ols = Lasso(
            alpha=0.01, inference_method="cpu_ols_inference", device="cpu",
            compute_inference=True, max_iter=1000, tol=1e-6, cpu_solver="fista",
        )
        m_ols.fit(X, y)

        assert np.allclose(m_db._bse, m_ols._bse, rtol=0.5, atol=0.05), (
            "In low-dim regime, debiased SE should be within 50% of OLS SE"
        )


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestDebiasedInferenceGPU:
    """Tests for debiased Lasso inference on GPU."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y, self.beta, self.support = _make_sparse_regression(
            n=300, p=40, s=5, seed=456
        )

    def test_gpu_shapes(self):
        m = Lasso(
            alpha=0.1, inference_method="debiased", device="cuda",
            compute_inference=True, max_iter=500, tol=1e-5,
            solver="fista",
        )
        m.fit(self.X, self.y)

        n_params = 1 + self.X.shape[1]
        assert m._bse is not None
        assert m._bse.shape == (n_params,)
        assert m._pvalues.shape == (n_params,)
        assert m._conf_int.shape == (n_params, 2)

    def test_gpu_cpu_consistency(self):
        """GPU and CPU debiased inference should produce close results."""
        m_cpu = Lasso(
            alpha=0.1, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m_cpu.fit(self.X, self.y)

        m_gpu = Lasso(
            alpha=0.1, inference_method="debiased", device="cuda",
            compute_inference=True, max_iter=500, tol=1e-5,
            solver="fista",
        )
        m_gpu.fit(self.X, self.y)

        assert np.allclose(m_cpu.coef_, m_gpu.coef_, rtol=1e-3, atol=1e-4), (
            "CPU and GPU Lasso coefficients should be close"
        )
        assert np.allclose(m_cpu._bse, m_gpu._bse, rtol=0.1, atol=0.01), (
            "CPU and GPU debiased SE should be close"
        )
        assert np.allclose(m_cpu._pvalues, m_gpu._pvalues, rtol=0.2, atol=0.05), (
            "CPU and GPU debiased p-values should be close"
        )

    def test_gpu_pvalues_in_range(self):
        m = Lasso(
            alpha=0.05, inference_method="debiased", device="cuda",
            compute_inference=True, max_iter=500, tol=1e-5,
            solver="fista",
        )
        m.fit(self.X, self.y)
        assert np.all(m._pvalues >= 0)
        assert np.all(m._pvalues <= 1)

    def test_gpu_bse_positive(self):
        m = Lasso(
            alpha=0.05, inference_method="debiased", device="cuda",
            compute_inference=True, max_iter=500, tol=1e-5,
            solver="fista",
        )
        m.fit(self.X, self.y)
        assert np.all(m._bse > 0)
