"""
Tests for debiased Lasso inference (Javanmard-Montanari / Zhang-Zhang).

Covers:
  - CPU path: shapes, signs, coverage, comparison with OLS on well-specified data
  - GPU path: CPU vs GPU consistency
  - summary() output
"""

import numpy as np
import pytest

from statgpu.inference import DebiasedInferenceResult, ParameterInferenceResult
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
        assert isinstance(m._inference_result, DebiasedInferenceResult)
        assert m._inference_result.precision_method == "nodewise_lasso"
        assert m._inference_result.statistic_name == "z"
        assert np.allclose(m._inference_result.statistic, m._tvalues)
        assert np.allclose(m._zvalues, m._tvalues)
        assert m._inference_result.metadata["precision_cache_hit"] in (True, False)

    def test_compute_inference_false_no_result(self):
        m = Lasso(
            alpha=0.1, inference_method="cpu_ols_inference", device="cpu",
            compute_inference=False, max_iter=100, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        assert m._bse is None
        assert m._pvalues is None
        assert m._conf_int is None
        assert m._inference_result is None

    def test_cpu_ols_result_container(self):
        m = Lasso(
            alpha=0.1, inference_method="cpu_ols_inference", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        assert isinstance(m._inference_result, ParameterInferenceResult)
        assert not isinstance(m._inference_result, DebiasedInferenceResult)
        assert m._inference_result.method == "post_selection_ols"
        assert m._inference_result.statistic_name == "t"
        assert m._inference_result.metadata["heuristic_post_selection"] is True
        assert m._inference_result.metadata["backend_path"] == "cpu_ols"
        assert np.allclose(m._inference_result.bse, m._bse)
        assert np.allclose(m._inference_result.statistic, m._tvalues)

    def test_cpu_ols_refit_clears_stale_zvalues(self):
        m = Lasso(
            alpha=0.1, inference_method="debiased", device="cpu",
            compute_inference=True, max_iter=500, tol=1e-5, cpu_solver="fista",
        )
        m.fit(self.X, self.y)
        assert m._zvalues is not None
        m.inference_method = "cpu_ols_inference"
        m.fit(self.X, self.y)
        assert m._zvalues is None
        assert isinstance(m._inference_result, ParameterInferenceResult)
        assert m._inference_result.statistic_name == "t"

    def test_bootstrap_result_container_metadata(self):
        m = Lasso(
            alpha=0.1, inference_method="bootstrap", device="cpu",
            compute_inference=True, max_iter=200, tol=1e-5, cpu_solver="fista",
            n_bootstrap=12, bootstrap_random_state=123,
        )
        m.fit(self.X[:120], self.y[:120])
        assert isinstance(m._inference_result, ParameterInferenceResult)
        assert m._inference_result.method == "residual_bootstrap"
        assert m._inference_result.distribution == "bootstrap_percentile"
        assert m._inference_result.metadata["n_bootstrap"] == 12
        assert m._inference_result.metadata["random_state"] == 123
        assert "samples" not in m._inference_result.to_dict()

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

    def test_f_pvalue_zero_predictors_returns_none(self):
        """Zero-predictor edge case should not map infinite F to near-zero p-value."""
        m = Lasso(alpha=0.1, device="cpu")
        m.coef_ = np.array([], dtype=float)
        m._df_resid = 8
        m._y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        m._resid = np.array([0.1, -0.1, 0.1, -0.1], dtype=float)

        assert m.fvalue == np.inf
        assert m.f_pvalue is None

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

    def test_simultaneous_ci_shape_and_width(self):
        m = Lasso(
            alpha=0.05,
            inference_method="debiased",
            device="cpu",
            compute_inference=True,
            max_iter=500,
            tol=1e-5,
            cpu_solver="fista",
            enable_simultaneous_inference=True,
            simultaneous_method="maxz_bootstrap",
            simultaneous_n_bootstrap=128,
            simultaneous_random_state=123,
            simultaneous_include_intercept=False,
        )
        m.fit(self.X, self.y)
        assert m._conf_int_simultaneous is not None
        assert m._conf_int_simultaneous.shape == m._conf_int.shape
        assert isinstance(m._inference_result, DebiasedInferenceResult)
        assert np.allclose(
            m._inference_result.simultaneous_conf_int,
            m._conf_int_simultaneous,
        )
        assert m._inference_result.simultaneous_method == "maxz_bootstrap"
        assert m._inference_result.simultaneous_n_bootstrap == 128
        assert np.isclose(
            m._inference_result.simultaneous_critical_value,
            m._simultaneous_critical_value,
        )
        # Feature intervals should be weakly wider than marginal intervals.
        idx = np.arange(1, m._conf_int.shape[0])
        width_m = m._conf_int[idx, 1] - m._conf_int[idx, 0]
        width_s = m._conf_int_simultaneous[idx, 1] - m._conf_int_simultaneous[idx, 0]
        assert np.all(width_s >= width_m - 1e-12)

    def test_simultaneous_ci_reproducible(self):
        kwargs = dict(
            alpha=0.05,
            inference_method="debiased",
            device="cpu",
            compute_inference=True,
            max_iter=500,
            tol=1e-5,
            cpu_solver="fista",
            enable_simultaneous_inference=True,
            simultaneous_method="maxz_bootstrap",
            simultaneous_n_bootstrap=96,
            simultaneous_random_state=7,
            simultaneous_include_intercept=False,
        )
        m1 = Lasso(**kwargs)
        m1.fit(self.X, self.y)
        m2 = Lasso(**kwargs)
        m2.fit(self.X, self.y)
        assert np.allclose(m1._conf_int_simultaneous, m2._conf_int_simultaneous)
        assert np.isclose(m1._simultaneous_critical_value, m2._simultaneous_critical_value)

    def test_simultaneous_guardrails(self):
        with pytest.raises(ValueError, match="simultaneous_method must be 'maxz_bootstrap'"):
            Lasso(
                alpha=0.1,
                inference_method="debiased",
                device="cpu",
                compute_inference=True,
                enable_simultaneous_inference=True,
                simultaneous_method="bonferroni",
            ).fit(self.X, self.y)

        with pytest.raises(ValueError, match="requires inference_method='debiased'"):
            Lasso(
                alpha=0.1,
                inference_method="cpu_ols_inference",
                device="cpu",
                compute_inference=True,
                enable_simultaneous_inference=True,
                simultaneous_method="maxz_bootstrap",
            ).fit(self.X, self.y)

        with pytest.raises(ValueError, match="requires compute_inference=True"):
            Lasso(
                alpha=0.1,
                inference_method="debiased",
                device="cpu",
                compute_inference=False,
                enable_simultaneous_inference=True,
                simultaneous_method="maxz_bootstrap",
            ).fit(self.X, self.y)

    def test_summary_includes_simultaneous_block(self, capsys):
        m = Lasso(
            alpha=0.05,
            inference_method="debiased",
            device="cpu",
            compute_inference=True,
            max_iter=500,
            tol=1e-5,
            cpu_solver="fista",
            enable_simultaneous_inference=True,
            simultaneous_method="maxz_bootstrap",
            simultaneous_n_bootstrap=64,
            simultaneous_random_state=11,
            simultaneous_include_intercept=False,
        )
        m.fit(self.X, self.y)
        m.summary()
        captured = capsys.readouterr()
        assert "Simultaneous inference" in captured.out
        assert "critical value (max|Z|)" in captured.out


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
        assert isinstance(m._inference_result, DebiasedInferenceResult)
        assert m._inference_result.precision_method == "nodewise_lasso"
        assert m._inference_result.metadata["backend_path"] == "cupy_debiased"

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

    def test_gpu_simultaneous_ci_runs(self):
        m = Lasso(
            alpha=0.05,
            inference_method="debiased",
            device="cuda",
            compute_inference=True,
            max_iter=500,
            tol=1e-5,
            solver="fista",
            enable_simultaneous_inference=True,
            simultaneous_method="maxz_bootstrap",
            simultaneous_n_bootstrap=96,
            simultaneous_random_state=7,
            simultaneous_include_intercept=False,
        )
        m.fit(self.X, self.y)
        assert m._conf_int_simultaneous is not None
        assert m._conf_int_simultaneous.shape == m._conf_int.shape
        assert isinstance(m._inference_result, DebiasedInferenceResult)
        assert np.allclose(
            m._inference_result.simultaneous_conf_int,
            m._conf_int_simultaneous,
        )
        assert m._resid is None
        assert m._X_design is None
        idx = np.arange(1, m._conf_int.shape[0])
        width_m = m._conf_int[idx, 1] - m._conf_int[idx, 0]
        width_s = m._conf_int_simultaneous[idx, 1] - m._conf_int_simultaneous[idx, 0]
        assert np.all(width_s >= width_m - 1e-12)

    def test_gpu_simultaneous_ci_cpu_gpu_consistency(self):
        """GPU batched node-wise path should preserve simultaneous inference numerics."""
        common = dict(
            alpha=0.05,
            inference_method="debiased",
            compute_inference=True,
            max_iter=500,
            tol=1e-5,
            enable_simultaneous_inference=True,
            simultaneous_method="maxz_bootstrap",
            simultaneous_n_bootstrap=96,
            simultaneous_random_state=7,
            simultaneous_include_intercept=False,
        )
        m_cpu = Lasso(device="cpu", cpu_solver="fista", **common)
        m_cpu.fit(self.X, self.y)

        m_gpu = Lasso(device="cuda", solver="fista", **common)
        m_gpu.fit(self.X, self.y)

        # Core parameters should remain numerically aligned after batched GPU node-wise solves.
        assert np.allclose(m_cpu.coef_, m_gpu.coef_, rtol=1e-3, atol=1e-4)
        # Simultaneous CI is bootstrap-based, allow a slightly looser tolerance.
        assert np.isclose(
            m_cpu._simultaneous_critical_value,
            m_gpu._simultaneous_critical_value,
            rtol=0.1,
            atol=0.15,
        )
        assert np.allclose(
            m_cpu._conf_int_simultaneous,
            m_gpu._conf_int_simultaneous,
            rtol=0.12,
            atol=0.02,
        )

        idx = np.arange(1, m_gpu._conf_int.shape[0])
        width_m = m_gpu._conf_int[idx, 1] - m_gpu._conf_int[idx, 0]
        width_s = m_gpu._conf_int_simultaneous[idx, 1] - m_gpu._conf_int_simultaneous[idx, 0]
        assert np.all(width_s >= width_m - 1e-12)
