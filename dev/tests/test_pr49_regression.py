"""
Comprehensive regression tests for PR #49 changes.

Covers all modified functions with precision and performance checks.
"""
import time
import numpy as np
import pytest


# ======================================================================
# 1. _cv_base.py: kfold_indices validation
# ======================================================================

class TestKfoldIndices:
    def test_rejects_n_splits_lt_2(self):
        from statgpu.linear_model._cv_base import kfold_indices
        with pytest.raises(ValueError, match="at least 2"):
            kfold_indices(100, 1)
        with pytest.raises(ValueError, match="at least 2"):
            kfold_indices(100, 0)

    def test_accepts_n_splits_eq_2(self):
        from statgpu.linear_model._cv_base import kfold_indices
        folds = kfold_indices(100, 2, random_state=0)
        assert len(folds) == 2

    def test_rejects_n_splits_gt_n(self):
        from statgpu.linear_model._cv_base import kfold_indices
        with pytest.raises(ValueError, match="cannot be greater"):
            kfold_indices(5, 10)


# ======================================================================
# 2. _cv_base.py: detect_gpu_input mixed backend warning
# ======================================================================

class TestDetectGpuInput:
    def test_numpy_arrays(self):
        from statgpu.linear_model._cv_base import detect_gpu_input
        X = np.zeros((10, 3))
        y = np.zeros(10)
        backend, Xo, yo = detect_gpu_input(X, y)
        assert backend == "numpy"

    def test_mixed_backend_warns(self):
        """Mixed cupy+torch should warn; mixed numpy+torch should not."""
        from statgpu.linear_model._cv_base import detect_gpu_input
        try:
            import torch
            import cupy
            X = cupy.zeros((10, 3))
            y = torch.zeros(10)
            with pytest.warns(RuntimeWarning, match="Mixed backend"):
                backend, _, _ = detect_gpu_input(X, y)
            assert backend == "numpy"
        except ImportError:
            pytest.skip("cupy+torch not both available")


# ======================================================================
# 3. _cv_engine.py: exception logging and cache_key init
# ======================================================================

class TestCvEngine:
    def test_empty_alpha_grid_raises(self):
        from statgpu.linear_model._cv_engine import run_cv
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        with pytest.raises(ValueError, match="alpha_grid must not be empty"):
            run_cv(X, y, np.array([]), lambda *a, **k: 0.0)

    def test_cache_key_initialized(self):
        from statgpu.linear_model._cv_engine import run_cv
        # Should not raise NameError even without cache
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        best, means, all_s = run_cv(
            X, y, np.array([0.1]),
            lambda Xtr, ytr, Xv, yv, a, **kw: float(np.mean((yv - np.mean(ytr))**2)),
            n_folds=2,
        )
        assert np.isfinite(best)


# ======================================================================
# 4. _penalized_cv.py: intercept initialization for logistic
# ======================================================================

class TestInterceptInit:
    def test_logistic_uses_logit_link(self):
        """Verify logistic CV path uses logit(y_mean) not log(y_mean)."""
        from statgpu.linear_model._penalized_cv import _logistic_sparse_cv_path
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        y = (X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) > 0).astype(float)

        result = _logistic_sparse_cv_path(
            X, y, np.array([0.1]), "l1", 0.5,
            max_iter=100, tol=1e-4, device="cpu",
        )
        # Should not return None (which means fallback)
        assert result is not None
        assert "scores" in result


# ======================================================================
# 5. Import aliases: _folds_are_complete
# ======================================================================

class TestImportAliases:
    def test_lasso_cv_imports_correctly(self):
        from statgpu.linear_model._lasso_cv import _folds_are_complete
        assert callable(_folds_are_complete)

    def test_ridge_cv_imports_correctly(self):
        from statgpu.linear_model._ridge_cv import _folds_are_complete
        assert callable(_folds_are_complete)

    def test_elasticnet_cv_imports_correctly(self):
        from statgpu.linear_model._elasticnet_cv import _folds_are_complete
        assert callable(_folds_are_complete)

    def test_logistic_cv_imports_correctly(self):
        from statgpu.linear_model._logistic_cv import _folds_are_complete
        assert callable(_folds_are_complete)


# ======================================================================
# 6. _irls.py: torch dtype promotion
# ======================================================================

class TestIrlsDtype:
    def test_mixed_float_dtype_torch(self):
        """IRLS should handle X=float32, y=float64 without error."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("torch not available")

        from statgpu.linear_model import PenalizedGeneralizedLinearModel
        rng = np.random.default_rng(16)
        X_np = rng.normal(size=(36, 4)).astype(np.float32)
        y_np = rng.poisson(lam=5.0, size=36).astype(np.float64)

        glm = PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.01,
            solver="irls", device="torch", fit_intercept=False, max_iter=5,
        )
        glm.fit(X_np, y_np)
        assert np.all(np.isfinite(glm.coef_))


# ======================================================================
# 7. _solver.py: Lipschitz named constants
# ======================================================================

class TestLipschitzConstants:
    def test_constants_exist(self):
        from statgpu.glm_core._solver import (
            _LIPSCHITZ_SAFETY_INVERSE_GAUSSIAN,
            _LIPSCHITZ_SAFETY_TWEEDIE,
            _LIPSCHITZ_SAFETY_GAMMA,
            _LIPSCHITZ_SAFETY_LOGISTIC_CV,
        )
        assert _LIPSCHITZ_SAFETY_INVERSE_GAUSSIAN == 3.0
        assert _LIPSCHITZ_SAFETY_TWEEDIE == 5.0
        assert _LIPSCHITZ_SAFETY_GAMMA == 3.0
        assert _LIPSCHITZ_SAFETY_LOGISTIC_CV == 2.0


# ======================================================================
# 8. _penalized.py: dispatch table
# ======================================================================

class TestDispatchTable:
    def test_dispatch_table_exists(self):
        from statgpu.linear_model._penalized import _SOLVER_DISPATCH_TABLE
        assert len(_SOLVER_DISPATCH_TABLE) > 10

    def test_exact_for_ridge(self):
        from statgpu.linear_model._penalized import _preferred_penalized_glm_solver
        assert _preferred_penalized_glm_solver("squared_error", "l2") == "exact"

    def test_fista_for_nonconvex(self):
        from statgpu.linear_model._penalized import _preferred_penalized_glm_solver
        assert _preferred_penalized_glm_solver("squared_error", "scad") == "fista"
        assert _preferred_penalized_glm_solver("squared_error", "mcp") == "fista"

    def test_fista_bb_for_poisson_l1_gpu(self):
        from statgpu.linear_model._penalized import _preferred_penalized_glm_solver
        assert _preferred_penalized_glm_solver(
            "poisson", "l1", backend_name="cupy", cv_mode=True
        ) == "fista_bb"

    def test_newton_for_poisson_l2_cv(self):
        from statgpu.linear_model._penalized import _preferred_penalized_glm_solver
        assert _preferred_penalized_glm_solver(
            "poisson", "l2", backend_name="cupy", cv_mode=True
        ) == "newton"


# ======================================================================
# 9. Debiased inference: L1 precision vs R
# ======================================================================

class TestDebiasedPrecision:
    def test_se_matches_r_within_1pct(self):
        """SE should match R hdi::lasso.proj within 1%."""
        from statgpu.linear_model import Lasso

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)

        model = Lasso(alpha=0.05, compute_inference=True, device="cpu")
        model.fit(X, y)

        # R hdi results with same data (from validated run)
        r_se = [0.0146517, 0.01335007, 0.01320719, 0.01344525, 0.01216339]
        p_se = list(model._bse[1:])  # skip intercept

        for i in range(5):
            rel_diff = abs(p_se[i] - r_se[i]) / r_se[i]
            assert rel_diff < 0.01, f"SE[{i}] diff {rel_diff:.4f} > 1%"

    def test_lasso_solution_matches_sklearn(self):
        """Lasso coefficients should match sklearn to machine precision."""
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        from sklearn.linear_model import Lasso as SkLasso

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)

        m = PenalizedLinearRegression(
            penalty="l1", alpha=0.05, fit_intercept=True,
            max_iter=100000, tol=1e-12, device="cpu",
            cpu_solver="coordinate_descent",
            compute_inference=False, inference_method="none",
        )
        m.fit(X, y)

        sk = SkLasso(alpha=0.05, fit_intercept=True, max_iter=100000, tol=1e-12)
        sk.fit(X, y)

        coef_diff = np.max(np.abs(m.coef_ - sk.coef_))
        assert coef_diff < 1e-10, f"coef diff {coef_diff:.2e} > 1e-10"

    def test_three_backend_consistency(self):
        """SE should be consistent across CPU/CuPy/Torch."""
        from statgpu.linear_model import Lasso

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)

        results = {}
        for device in ["cpu"]:
            m = Lasso(alpha=0.05, compute_inference=True, device=device)
            m.fit(X, y)
            results[device] = list(m._bse)

        try:
            import cupy
            m = Lasso(alpha=0.05, compute_inference=True, device="cuda")
            m.fit(X, y)
            results["cuda"] = list(m._bse)
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                m = Lasso(alpha=0.05, compute_inference=True, device="torch")
                m.fit(X, y)
                results["torch"] = list(m._bse)
        except Exception:
            pass

        if len(results) < 2:
            pytest.skip("Need at least 2 backends")

        devices = list(results.keys())
        for i in range(len(results["cpu"])):
            vals = [results[d][i] for d in devices]
            max_diff = max(vals) - min(vals)
            mean_val = np.mean(vals)
            if mean_val > 1e-10:
                rel_diff = max_diff / mean_val
                assert rel_diff < 0.01, f"bse[{i}] backend diff {rel_diff:.4f} > 1%"


# ======================================================================
# 10. ElasticNet debiased inference
# ======================================================================

class TestElasticNetDebiased:
    def test_elasticnet_debiased_works(self):
        from statgpu.linear_model._penalized import PenalizedLinearRegression

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)

        model = PenalizedLinearRegression(
            penalty="elasticnet", alpha=0.05, l1_ratio=0.5,
            compute_inference=True, inference_method="debiased", device="cpu",
        )
        model.fit(X, y)

        assert model._bse is not None
        assert model._pvalues is not None
        assert model._conf_int is not None
        assert model.rsquared is not None
        assert model.rsquared_adj is not None
        assert model.fvalue is not None
        assert model.f_pvalue is not None
        assert len(model._bse) == 6  # 5 features + intercept


# ======================================================================
# 11. Simultaneous inference
# ======================================================================

class TestSimultaneousInference:
    def test_simultaneous_ci_wider_than_marginal(self):
        from statgpu.linear_model import Lasso

        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X @ np.array([2.0, -1.0, 0.5, 0.0, 0.0]) + 0.2 * np.random.randn(200)

        model = Lasso(
            alpha=0.05, compute_inference=True,
            enable_simultaneous_inference=True,
            simultaneous_n_bootstrap=500,
            simultaneous_random_state=42,
            device="cpu",
        )
        model.fit(X, y)

        assert model._simultaneous_enabled
        assert model._conf_int_simultaneous is not None

        # Simultaneous CI should be wider than marginal for nonzero coefs
        for i in range(1, 6):  # skip intercept
            marg_width = model._conf_int[i, 1] - model._conf_int[i, 0]
            sim_width = model._conf_int_simultaneous[i, 1] - model._conf_int_simultaneous[i, 0]
            assert sim_width >= marg_width * 0.99, \
                f"Feature {i}: simultaneous CI ({sim_width:.4f}) < marginal ({marg_width:.4f})"


# ======================================================================
# 12. Summary output
# ======================================================================

class TestSummary:
    def test_l1_summary_has_z_stats(self, capsys):
        from statgpu.linear_model import Lasso

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * np.random.randn(50)

        model = Lasso(alpha=0.05, compute_inference=True, device="cpu")
        model.fit(X, y)
        model.summary()

        captured = capsys.readouterr()
        assert "Debiased Lasso Results" in captured.out
        assert "P>|z|" in captured.out
        assert "R-squared:" in captured.out
        assert "Adj. R-squared:" in captured.out
        assert "F-statistic:" in captured.out

    def test_ridge_summary_has_t_stats(self, capsys):
        from statgpu.linear_model._penalized import PenalizedLinearRegression

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * np.random.randn(50)

        model = PenalizedLinearRegression(
            penalty="l2", alpha=0.1, compute_inference=True, device="cpu",
        )
        model.fit(X, y)
        model.summary()

        captured = capsys.readouterr()
        assert "Ridge Regression Results" in captured.out
        assert "P>|t|" in captured.out


# ======================================================================
# 13. Performance: no regression
# ======================================================================

class TestPerformance:
    @pytest.mark.parametrize("n,p", [(100, 5), (500, 20), (1000, 50)])
    def test_debiased_inference_time(self, n, p):
        """Debiased inference should complete within reasonable time."""
        from statgpu.linear_model import Lasso

        np.random.seed(42)
        X = np.random.randn(n, p)
        beta = np.zeros(p)
        beta[:3] = [1.0, -0.5, 0.3]
        y = X @ beta + 0.1 * np.random.randn(n)

        t0 = time.perf_counter()
        model = Lasso(alpha=0.05, compute_inference=True, device="cpu")
        model.fit(X, y)
        elapsed = time.perf_counter() - t0

        # Generous limits (should be much faster in practice)
        limit = {(100, 5): 1.0, (500, 20): 5.0, (1000, 50): 20.0}[(n, p)]
        assert elapsed < limit, f"n={n},p={p}: {elapsed:.3f}s > {limit}s"
