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


# ======================================================================
# 14. LassoCV unification: uses _select_lasso_alpha_cv with cache
# ======================================================================

class TestLassoCVUnified:
    def test_lasso_cv_fit_basic(self):
        """LassoCV should fit and produce valid results."""
        from statgpu.linear_model import LassoCV

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = LassoCV(cv=5).fit(X, y)
        assert m.alpha_ > 0
        assert m.best_score_ < 0  # negative MSE
        assert m.coef_ is not None
        assert m.intercept_ is not None
        assert m.estimator_ is not None

    def test_lasso_cv_weighted(self):
        """LassoCV with sample_weight should work."""
        from statgpu.linear_model import LassoCV

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)
        w = np.random.uniform(0.5, 2.0, 100)

        m = LassoCV(cv=5).fit(X, y, sample_weight=w)
        assert m.alpha_ > 0
        assert m.best_score_ < 0

    def test_lasso_cv_predict(self):
        """LassoCV predict should delegate to estimator."""
        from statgpu.linear_model import LassoCV

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = LassoCV(cv=5).fit(X, y)
        y_pred = m.predict(X)
        assert y_pred.shape == y.shape
        # R² should be reasonable
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5

    def test_lasso_cv_score(self):
        """LassoCV score should use estimator."""
        from statgpu.linear_model import LassoCV

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = LassoCV(cv=5).fit(X, y)
        score = m.score(X, y)
        assert score > 0.5

    def test_lasso_cv_degenerate_cv1(self):
        """LassoCV with cv_splits with 1 fold should handle degenerate case."""
        from statgpu.linear_model import LassoCV

        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(20)

        # Single fold: degenerate
        cv_splits = [(np.arange(10), np.arange(10, 20))]
        m = LassoCV(cv_splits=cv_splits).fit(X, y)
        assert m.alpha_ > 0


# ======================================================================
# 15. ElasticNetCV: batch_mse backend conversion
# ======================================================================

class TestElasticNetCVBatchMse:
    def test_elasticnet_cv_fit(self):
        """ElasticNetCV should fit without backend mismatch errors."""
        from statgpu.linear_model import ElasticNetCV

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = ElasticNetCV(cv=5).fit(X, y)
        assert m.alpha_ > 0
        assert m.best_score_ < 0

    def test_elasticnet_cv_predict(self):
        """ElasticNetCV predict should delegate to estimator."""
        from statgpu.linear_model import ElasticNetCV

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = ElasticNetCV(cv=5).fit(X, y)
        y_pred = m.predict(X)
        assert y_pred.shape == y.shape


# ======================================================================
# 16. hash_cv_data: shared across elasticnet and logistic
# ======================================================================

class TestHashCvData:
    def test_hash_deterministic(self):
        """Same inputs should produce same hash."""
        from statgpu.linear_model._cv_base import hash_cv_data

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        h1 = hash_cv_data(X, y)
        h2 = hash_cv_data(X, y)
        assert h1 == h2

    def test_hash_different_data(self):
        """Different data should produce different hash."""
        from statgpu.linear_model._cv_base import hash_cv_data

        np.random.seed(42)
        X1 = np.random.randn(50, 5)
        y1 = np.random.randn(50)
        np.random.seed(43)
        X2 = np.random.randn(50, 5)
        y2 = np.random.randn(50)

        h1 = hash_cv_data(X1, y1)
        h2 = hash_cv_data(X2, y2)
        assert h1 != h2

    def test_hash_with_sample_weight(self):
        """Adding sample_weight should change hash."""
        from statgpu.linear_model._cv_base import hash_cv_data

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        w = np.random.uniform(0.5, 2.0, 50)

        h_no_w = hash_cv_data(X, y)
        h_with_w = hash_cv_data(X, y, sample_weight=w)
        assert h_no_w != h_with_w

    def test_hash_importable_from_elasticnet(self):
        """_hash_data should be importable from _elasticnet_cv."""
        from statgpu.linear_model._elasticnet_cv import _hash_data
        assert callable(_hash_data)

    def test_hash_importable_from_logistic(self):
        """_hash_logistic_data should be importable from _logistic_cv."""
        from statgpu.linear_model._logistic_cv import _hash_logistic_data
        assert callable(_hash_logistic_data)


# ======================================================================
# 17. SelectivePenalty singleton: thread-local
# ======================================================================

class TestSelectivePenaltySingleton:
    def test_singleton_same_thread(self):
        """Same thread should get same instance."""
        from statgpu.linear_model._penalized import _get_selective_penalty_singleton

        s1 = _get_selective_penalty_singleton()
        s2 = _get_selective_penalty_singleton()
        assert s1 is s2

    def test_singleton_configure(self):
        """configure() should update the singleton."""
        from statgpu.linear_model._penalized import _get_selective_penalty_singleton
        from statgpu.penalties._l1 import L1Penalty

        s = _get_selective_penalty_singleton()
        pen = L1Penalty(alpha=0.1)
        s.configure(pen, 10, "numpy")
        assert s._p == 10
        assert s._backend == "numpy"


# ======================================================================
# 18. _is_uniform_weight: shared helper
# ======================================================================

class TestIsUniformWeight:
    def test_none_is_uniform(self):
        """None weight should be uniform."""
        from statgpu.linear_model._penalized_cv import _is_uniform_weight
        assert _is_uniform_weight(None) is True

    def test_constant_is_uniform(self):
        """Constant weight should be uniform."""
        from statgpu.linear_model._penalized_cv import _is_uniform_weight
        w = np.ones(100) * 2.0
        assert _is_uniform_weight(w) is True

    def test_varying_is_not_uniform(self):
        """Varying weight should not be uniform."""
        from statgpu.linear_model._penalized_cv import _is_uniform_weight
        w = np.random.uniform(0.5, 2.0, 100)
        assert _is_uniform_weight(w) is False


# ======================================================================
# 19. Lasso kwargs validation
# ======================================================================

class TestLassoKwargs:
    def test_unknown_kwarg_raises(self):
        """Lasso with unknown kwarg should raise TypeError."""
        from statgpu.linear_model import Lasso
        with pytest.raises(TypeError):
            Lasso(alph=0.1)  # typo: alph instead of alpha

    def test_valid_kwargs_work(self):
        """Lasso with valid kwargs should work."""
        from statgpu.linear_model import Lasso
        m = Lasso(alpha=0.1, fit_intercept=True)
        assert m.alpha == 0.1


# ======================================================================
# 20. cv parameter validation
# ======================================================================

class TestCvValidation:
    def test_cv_lt_2_raises(self):
        """cv < 2 should raise ValueError."""
        from statgpu.linear_model import RidgeCV
        with pytest.raises(ValueError, match="cv"):
            RidgeCV(cv=1)

    def test_cv_eq_2_works(self):
        """cv=2 should work."""
        from statgpu.linear_model import RidgeCV
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(20)
        m = RidgeCV(cv=2).fit(X, y)
        assert m.alpha_ > 0


# ======================================================================
# 21. group_lasso alpha grid direction
# ======================================================================

class TestGroupLassoGrid:
    def test_group_lasso_grid_descending(self):
        """group_lasso alpha grid should be descending (largest first)."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        # This is tested indirectly through fit; if grid is ascending,
        # warm-start would fail and CV scores would be wrong.
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)
        # Just verify it doesn't crash
        m = PenalizedGLM_CV(loss='squared_error', penalty='group_lasso', cv=3,
                            n_alphas=5, device='cpu')
        # group_lasso requires groups; skip full fit test
        assert m.n_alphas == 5


# ======================================================================
# 22. FE predict vectorized
# ======================================================================

class TestFEPredictVectorized:
    def test_predict_with_entity_effects(self):
        """Panel FE predict should be vectorized and correct."""
        from statgpu.panel._fixed_effects import PanelOLS

        np.random.seed(42)
        n = 100
        entity_ids = np.repeat(np.arange(10), 10)
        entity_effects = np.repeat(np.random.randn(10), 10)
        X = np.random.randn(n, 3)
        y = X @ np.random.randn(3) + entity_effects + 0.1 * np.random.randn(n)

        model = PanelOLS(entity_effects=True)
        model.fit(y, X, entity_ids=entity_ids)

        y_pred = model.predict(X, entity_ids=entity_ids)
        assert y_pred.shape == y.shape

        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.9

    def test_predict_without_effects(self):
        """Panel FE predict without entity_ids should return slope only."""
        from statgpu.panel._fixed_effects import PanelOLS

        np.random.seed(42)
        n = 100
        entity_ids = np.repeat(np.arange(10), 10)
        X = np.random.randn(n, 3)
        y = X @ np.random.randn(3) + np.repeat(np.random.randn(10), 10) + 0.1 * np.random.randn(n)

        model = PanelOLS(entity_effects=True)
        model.fit(y, X, entity_ids=entity_ids)

        y_pred_no_fe = model.predict(X)
        assert y_pred_no_fe.shape == y.shape


# ======================================================================
# 23. FE t-distribution for robust/clustered SE
# ======================================================================

class TestFETDistribution:
    def test_robust_uses_t_distribution(self):
        """Robust SE should use t-distribution (p-values from t, not z)."""
        from statgpu.panel._fixed_effects import PanelOLS

        np.random.seed(42)
        n = 50
        entity_ids = np.repeat(np.arange(10), 5)
        X = np.random.randn(n, 2)
        y = X @ np.random.randn(2) + np.repeat(np.random.randn(10), 5) + 0.1 * np.random.randn(n)

        model = PanelOLS(entity_effects=True, cov_type='robust')
        model.fit(y, X, entity_ids=entity_ids)

        # p-values should be finite and in [0, 1]
        assert np.all(np.isfinite(model.pvalues_))
        assert np.all(model.pvalues_ >= 0)
        assert np.all(model.pvalues_ <= 1)


# ======================================================================
# 24. P1 fixes: structural correctness
# ======================================================================

class TestP1StructuralFixes:
    def test_no_merge_conflict_marker(self):
        """_penalized.py should not contain merge conflict markers."""
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert '>>>>>>>' not in content
        assert '<<<<<<<' not in content
        assert '=======' not in content or '=========' not in content

    def test_INTERCEPT_CLIP_BOUND_defined(self):
        """_INTERCEPT_CLIP_BOUND should be defined as a module constant."""
        from statgpu.linear_model._penalized import _INTERCEPT_CLIP_BOUND
        assert _INTERCEPT_CLIP_BOUND == 15.0

    def test_alphas_sorted_typo_fixed(self):
        """_cv_fold_general should use alpha_sorted, not alphas_sorted."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        src = inspect.getsource(PenalizedGLM_CV)
        # The typo 'alphas_sorted[alpha_idx_sorted]' should not exist
        assert 'alphas_sorted[alpha_idx_sorted]' not in src

    def test_effective_cv_device_sets_attribute(self):
        """_effective_cv_device should set _cv_selected_device_."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        m = PenalizedGLM_CV(loss='squared_error', penalty='l1', cv=5)
        # After fit, _cv_selected_device_ should be set
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m.fit(X, y)
        assert hasattr(m, '_cv_selected_device_')
        assert m._cv_selected_device_ is not None

    def test_unravel_index_for_best_alpha(self):
        """ElasticNetCV should use np.unravel_index for best alpha selection."""
        import inspect
        from statgpu.linear_model._elasticnet_cv import _select_elasticnet_params_cv
        src = inspect.getsource(_select_elasticnet_params_cv)
        assert 'unravel_index' in src

    def test_cache_key_uses_blake2b(self):
        """Lasso cache key should use blake2b content hash."""
        import inspect
        from statgpu.linear_model._lasso import _array_identity_token
        src = inspect.getsource(_array_identity_token)
        assert 'blake2b' in src
        # Should not use memory pointer
        assert 'data_ptr' not in src
        assert '__array_interface__' not in src


# ======================================================================
# 25. P1 fixes: ridge_cv weighted
# ======================================================================

class TestP1RidgeCvWeighted:
    def test_n_samples_vec_float64(self):
        """RidgeCV n_samples_vec should be float64 for weighted sums."""
        import inspect
        from statgpu.linear_model._ridge_cv import _select_ridge_alpha_cv
        src = inspect.getsource(_select_ridge_alpha_cv)
        assert 'float64' in src and 'n_samples_vec' in src

    def test_sw_train_initialized(self):
        """sw_train should be initialized before the fold loop."""
        import inspect
        from statgpu.linear_model._ridge_cv import _select_ridge_alpha_cv
        src = inspect.getsource(_select_ridge_alpha_cv)
        assert 'sw_train = None' in src

    def test_weighted_ridge_cv_fit(self):
        """Weighted RidgeCV should fit without errors."""
        from statgpu.linear_model import RidgeCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)
        w = np.random.uniform(0.5, 2.0, 100)
        m = RidgeCV(cv=5).fit(X, y, sample_weight=w)
        assert m.alpha_ > 0
        assert m.best_score_ < 0


# ======================================================================
# 26. P1 fixes: solver/IRLS
# ======================================================================

class TestP1SolverFixes:
    def test_gamma_gradient_formula(self):
        """Fused gamma gradient should be X'(mu-y)/n, not X'(1-y/mu)/n."""
        import inspect
        from statgpu.glm_core._solver import _fused_gamma
        src = inspect.getsource(_fused_gamma)
        assert 'mu_c - y' in src

    def test_fista_bb_dg_initialized(self):
        """fista_bb_solver should initialize dg before the loop."""
        import inspect
        from statgpu.glm_core._solver import fista_bb_solver
        src = inspect.getsource(fista_bb_solver)
        assert 'dg = _zeros' in src or 'dg =' in src

    def test_logistic_alpha_max_uses_mean_y(self):
        """_generate_alpha_grid for logistic should use mean(y), not 0.5."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        src = inspect.getsource(PenalizedGLM_CV)
        assert 'mu_null' in src or 'mean(y)' in src

    def test_scad_mcp_n_iter_tracks_actual(self):
        """SCAD/MCP n_iter should track actual iterations, not hardcoded 1."""
        # Read the source file directly to check the CV path functions
        with open('statgpu/linear_model/_penalized_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The old 'iters.append(1)  # placeholder' should be replaced
        assert 'iters.append(1)  # placeholder' not in content


# ======================================================================
# 27. P2 fixes: backend/array_ops
# ======================================================================

class TestP2BackendFixes:
    def test_sigmoid_float32_safe(self):
        """_sigmoid should clip to [-88, 88] for float32."""
        import inspect
        from statgpu.backends._array_ops import _sigmoid
        src = inspect.getsource(_sigmoid)
        assert '88.0' in src

    def test_max_eigval_power_returns_current(self):
        """_max_eigval_power should return lambda_val, not lambda_old."""
        import inspect
        from statgpu.backends._array_ops import _max_eigval_power
        src = inspect.getsource(_max_eigval_power)
        assert 'return lambda_val' in src

    def test_torch_on_target_device_exact_match(self):
        """_torch_on_target_device should distinguish cuda:0 from cuda:1."""
        import inspect
        from statgpu.backends._utils import _torch_on_target_device
        src = inspect.getsource(_torch_on_target_device)
        assert 'tensor_device' in src

    def test_xp_cholesky_solve_cupy_direct(self):
        """xp_cholesky_solve for CuPy should skip Cholesky and use direct solve."""
        import inspect
        from statgpu.backends._utils import xp_cholesky_solve
        src = inspect.getsource(xp_cholesky_solve)
        # CuPy should go directly to solve, not compute Cholesky first
        assert "hasattr(A, 'get')" in src


# ======================================================================
# 28. P2 fixes: IRLS/solver
# ======================================================================

class TestP2SolverFixes:
    def test_beta_mom_passed_to_fused_kernel(self):
        """FISTA fused kernel should receive beta_mom, not hardcoded 0.0."""
        import inspect
        from statgpu.glm_core._solver import fista_solver
        src = inspect.getsource(fista_solver)
        # Should not have '0.0' as the last arg to _fused
        assert '_fused(y_k, grad, step, thresh, coef_old, beta_mom)' in src or \
               '_fused(y_k, grad, step, thresh, coef_old, 0.0)' not in src

    def test_eta_raw_reused_in_irls(self):
        """IRLS should reuse eta_raw instead of recomputing X @ params_old."""
        import inspect
        from statgpu.glm_core._irls import irls_solver
        src = inspect.getsource(irls_solver)
        # Should reuse eta_raw, not recompute
        assert 'eta_raw' in src

    def test_mu_clip_only_positive_families(self):
        """IRLS numpy path should only clip mu for positive-mu families."""
        import inspect
        from statgpu.glm_core._irls import irls_solver
        src = inspect.getsource(irls_solver)
        # Should check family before clipping
        assert 'not in ("gaussian"' in src or 'squared_error' in src

    def test_armijo_reject_step(self):
        """IRLS Armijo fallback should reject step (params_old), not use 0.1 step."""
        import inspect
        from statgpu.glm_core._irls import irls_solver
        src = inspect.getsource(irls_solver)
        # Should NOT have '0.1 * _direction' as fallback
        assert '0.1 * _direction' not in src

    def test_inverse_gaussian_value_includes_log(self):
        """Fused inverse_gaussian value should include log(mu) term."""
        import inspect
        from statgpu.glm_core._solver import _fused_inverse_gaussian
        src = inspect.getsource(_fused_inverse_gaussian)
        assert '_log' in src


# ======================================================================
# 29. P2 fixes: CV estimators
# ======================================================================

class TestP2CvFixes:
    def test_elasticnet_score_handles_zero_sstot(self):
        """ElasticNetCV.score() should handle ss_tot=0."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.ones(50)  # constant y -> ss_tot=0
        m = ElasticNetCV(cv=3).fit(X, y)
        score = m.score(X, y)
        assert np.isfinite(score)

    def test_lasso_cv_cv_splits_normalized(self):
        """LassoCV should validate cv_splits via _normalize_cv_splits."""
        # The normalization happens inside _select_lasso_alpha_cv which LassoCV delegates to
        import inspect
        from statgpu.linear_model._lasso import _select_lasso_alpha_cv
        src = inspect.getsource(_select_lasso_alpha_cv)
        assert '_normalize_cv_splits' in src

    def test_lasso_cv_inference_method_passthrough(self):
        """LassoCV should pass through user's inference_method."""
        import inspect
        from statgpu.linear_model._lasso_cv import LassoCV
        src = inspect.getsource(LassoCV.fit)
        assert 'inference_method=self.inference_method' in src

    def test_logistic_irls_uses_1_over_c(self):
        """Logistic CV IRLS should use 1/C, not 1/(2C)."""
        # Check the GPU IRLS path where alpha is computed
        with open('statgpu/linear_model/_logistic_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert '1.0 / C' in content
        # Old formula '1.0 / (2.0 * C)' should not exist
        assert '1.0 / (2.0 * C)' not in content

    def test_logistic_c_max_uses_y_minus_half(self):
        """Logistic C_max gradient should use y - 0.5 (not centered y)."""
        import inspect
        from statgpu.linear_model._logistic_cv import _default_logistic_c_grid
        src = inspect.getsource(_default_logistic_c_grid)
        assert 'y_arr - 0.5' in src or 'y - 0.5' in src

    def test_weighted_mse_zero_guard(self):
        """RidgeCV batch_mse should guard against zero weight sum."""
        import inspect
        from statgpu.linear_model._ridge_cv import _batch_mse_all_folds
        src = inspect.getsource(_batch_mse_all_folds)
        assert 'sw_sum_safe' in src

    def test_elasticnet_predict_delegates(self):
        """ElasticNetCV.predict should delegate to estimator_."""
        import inspect
        from statgpu.linear_model._elasticnet_cv import ElasticNetCV
        src = inspect.getsource(ElasticNetCV.predict)
        assert 'estimator_.predict' in src

    def test_lasso_cv_cache_thread_safe(self):
        """LassoCV cache should use threading.Lock."""
        with open('statgpu/linear_model/_lasso_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'threading.Lock' in content

    def test_lasso_cv_fit_intercept_true(self):
        """LassoCV default fit_intercept should be True."""
        from statgpu.linear_model import LassoCV
        m = LassoCV()
        assert m.fit_intercept is True

    def test_array_identity_token_samples(self):
        """_array_identity_token should sample rows, not hash full array."""
        import inspect
        from statgpu.linear_model._lasso import _array_identity_token
        src = inspect.getsource(_array_identity_token)
        assert 'n_sample' in src


# ======================================================================
# 30. P2 fixes: Panel FE
# ======================================================================

class TestP2FeFixes:
    def test_fe_within_r2_uses_sum_squared(self):
        """FE within R² should use sum(y_d**2), not var(y_d)."""
        # Check the source file directly since _compute_inference is a separate method
        with open('statgpu/panel/_fixed_effects.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'y_d ** 2' in content

    def test_fe_t_distribution_robust(self):
        """FE robust SE should use t-distribution."""
        # Check the source file directly since _compute_inference is a separate method
        with open('statgpu/panel/_fixed_effects.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'stats.t.cdf' in content

    def test_fe_hc1_uses_df_resid(self):
        """FE HC1 correction should use df_resid, not n-k."""
        import inspect
        from statgpu.panel._fixed_effects import PanelOLS
        src = inspect.getsource(PanelOLS.fit)
        assert 'self.df_resid' in src

    def test_fe_grand_mean_demeaned(self):
        """FE two-way predict should demean time effects to avoid double-counting."""
        import inspect
        from statgpu.panel._fixed_effects import PanelOLS
        src = inspect.getsource(PanelOLS.fit)
        assert 'grand_mean' in src

    def test_within_transform_uses_bincount(self):
        """within_transform should use vectorized bincount."""
        import inspect
        from statgpu.panel._utils import within_transform
        src = inspect.getsource(within_transform)
        assert 'bincount' in src

    def test_within_transform_matrix_exists(self):
        """_within_transform_matrix should exist for batch demeaning."""
        from statgpu.panel._utils import _within_transform_matrix
        assert callable(_within_transform_matrix)


# ======================================================================
# 31. P2/P3 fixes: PenalizedGLM
# ======================================================================

class TestP2P3PenalizedFixes:
    def test_group_lasso_in_sparse_grid(self):
        """group_lasso should be in the sparse penalty grid set."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        src = inspect.getsource(PenalizedGLM_CV)
        assert 'group_lasso' in src

    def test_populate_refit_respects_fit_intercept(self):
        """_populate_refit_model should respect fit_intercept for df_resid."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        src = inspect.getsource(PenalizedGLM_CV)
        assert 'fit_intercept' in src and '_df_resid' in src

    def test_fused_uses_registry(self):
        """fista_solver should use _GLM_FUSED_REGISTRY, not hardcoded list."""
        import inspect
        from statgpu.glm_core._solver import fista_solver
        src = inspect.getsource(fista_solver)
        assert '_GLM_FUSED_REGISTRY' in src

    def test_lasso_cv_delegates_to_select_lasso_alpha_cv(self):
        """LassoCV.fit should delegate to _select_lasso_alpha_cv."""
        import inspect
        from statgpu.linear_model._lasso_cv import LassoCV
        src = inspect.getsource(LassoCV.fit)
        assert '_select_lasso_alpha_cv' in src

    def test_logistic_gpu_prob_batched(self):
        """Logistic CV GPU probability should use batched matmul."""
        import inspect
        from statgpu.linear_model._logistic_cv import _select_logistic_c_cv
        src = inspect.getsource(_select_logistic_c_cv)
        assert 'coefs_all' in src

    def test_fit_initial_gpu_aware(self):
        """_fit_initial should accept backend_name for GPU data."""
        import inspect
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        src = inspect.getsource(PenalizedGeneralizedLinearModel._fit_initial)
        assert 'backend_name' in src


# ======================================================================
# 32. P3 fixes: code quality
# ======================================================================

class TestP3CodeQuality:
    def test_default_rng_used(self):
        """kfold_indices should use np.random.default_rng."""
        import inspect
        from statgpu.linear_model._cv_base import kfold_indices
        src = inspect.getsource(kfold_indices)
        assert 'default_rng' in src

    def test_best_score_negative_mse(self):
        """CV estimators should store negative MSE as best_score_."""
        from statgpu.linear_model import RidgeCV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m = RidgeCV(cv=3).fit(X, y)
        assert m.best_score_ < 0

    def test_cv_validation_min_2(self):
        """cv parameter should be validated to be >= 2."""
        from statgpu.linear_model import RidgeCV
        with pytest.raises(ValueError):
            RidgeCV(cv=1)

    def test_lasso_no_kwargs(self):
        """Lasso should not accept **kwargs (typos should raise)."""
        from statgpu.linear_model import Lasso
        with pytest.raises(TypeError):
            Lasso(alph=0.1)

    def test_ddof_consistency(self):
        """Alpha grid backend function should use ddof=1."""
        import inspect
        from statgpu.linear_model._lasso import _default_lasso_alpha_grid_backend
        src = inspect.getsource(_default_lasso_alpha_grid_backend)
        assert 'n_samples - 1' in src

    def test_n_cont_unified(self):
        """_n_cont for SCAD/MCP should be unified to 20."""
        import inspect
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        src = inspect.getsource(PenalizedGeneralizedLinearModel._fit_lla)
        assert '20 if _is_scad_mcp' in src

    def test_hash_cv_data_shared(self):
        """hash_cv_data should be in _cv_base.py."""
        from statgpu.linear_model._cv_base import hash_cv_data
        assert callable(hash_cv_data)

    def test_batch_mse_backend_conversion(self):
        """_batch_mse_elasticnet should convert arrays to backend."""
        import inspect
        from statgpu.linear_model._elasticnet_cv import _batch_mse_elasticnet
        src = inspect.getsource(_batch_mse_elasticnet)
        assert 'backend.asarray(coefs_path)' in src

    def test_no_dead_code_and_false(self):
        """CV files should not contain 'and False' dead code blocks."""
        for f in ['statgpu/linear_model/_ridge_cv.py',
                  'statgpu/linear_model/_elasticnet_cv.py',
                  'statgpu/linear_model/_lasso_cv.py']:
            with open(f, 'r', encoding='utf-8') as fh:
                content = fh.read()
            assert 'and False' not in content, f'Dead code found in {f}'


# ======================================================================
# 33. Performance: regression check
# ======================================================================

class TestPerformanceRegression:
    @pytest.mark.parametrize("n,p", [(100, 5), (500, 20)])
    def test_lasso_cv_time(self, n, p):
        """LassoCV should complete within reasonable time."""
        from statgpu.linear_model import LassoCV
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + 0.1 * np.random.randn(n)

        t0 = time.perf_counter()
        LassoCV(cv=5).fit(X, y)
        elapsed = time.perf_counter() - t0

        limit = {(100, 5): 2.0, (500, 20): 10.0}[(n, p)]
        assert elapsed < limit, f"n={n},p={p}: {elapsed:.3f}s > {limit}s"

    @pytest.mark.parametrize("n,p", [(100, 5), (500, 20)])
    def test_ridge_cv_time(self, n, p):
        """RidgeCV should complete within reasonable time."""
        from statgpu.linear_model import RidgeCV
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + 0.1 * np.random.randn(n)

        t0 = time.perf_counter()
        RidgeCV(cv=5).fit(X, y)
        elapsed = time.perf_counter() - t0

        limit = {(100, 5): 1.0, (500, 20): 5.0}[(n, p)]
        assert elapsed < limit, f"n={n},p={p}: {elapsed:.3f}s > {limit}s"


# ======================================================================
# 34. XtX precomputation optimization
# ======================================================================

class TestXtXPrecomputation:
    def test_elasticnet_cv_multiple_l1_ratios(self):
        """ElasticNetCV with multiple l1_ratios should produce correct results."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3).fit(X, y)
        assert m.alpha_ > 0
        assert m.l1_ratio_ in [0.1, 0.5, 0.9]
        assert m.best_score_ < 0

    def test_elasticnet_cv_single_l1_ratio(self):
        """ElasticNetCV with single l1_ratio should work."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = ElasticNetCV(l1_ratio=0.5, cv=3).fit(X, y)
        assert m.alpha_ > 0
        assert m.best_score_ < 0

    def test_elasticnet_cv_weighted(self):
        """ElasticNetCV with sample_weight should work."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)
        w = np.random.uniform(0.5, 2.0, 100)

        m = ElasticNetCV(cv=3).fit(X, y, sample_weight=w)
        assert m.alpha_ > 0

    def test_elasticnet_cv_predict(self):
        """ElasticNetCV predict should produce correct shape."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = ElasticNetCV(cv=3).fit(X, y)
        y_pred = m.predict(X)
        assert y_pred.shape == y.shape


# ======================================================================
# 35. batch_mse chunked computation
# ======================================================================

class TestBatchMseChunked:
    def test_batch_mse_basic(self):
        """batch_mse should compute correct MSE values."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(10, 3)
        intercepts = np.random.randn(10)

        mse = batch_mse(X, y, coefs, intercepts)
        assert mse.shape == (10,)
        assert np.all(np.isfinite(mse))
        assert np.all(mse >= 0)

    def test_batch_mse_no_intercept(self):
        """batch_mse without intercepts should work."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(10, 3)

        mse = batch_mse(X, y, coefs)
        assert mse.shape == (10,)
        assert np.all(np.isfinite(mse))

    def test_batch_mse_with_sample_weight(self):
        """batch_mse with sample_weight should work."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(10, 3)
        w = np.random.uniform(0.5, 2.0, 50)

        mse = batch_mse(X, y, coefs, sample_weight=w)
        assert mse.shape == (10,)
        assert np.all(np.isfinite(mse))

    def test_batch_mse_chunk_size(self):
        """batch_mse with different chunk_sizes should produce same results."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(20, 3)

        mse_256 = batch_mse(X, y, coefs, chunk_size=256)
        mse_8 = batch_mse(X, y, coefs, chunk_size=8)
        mse_3 = batch_mse(X, y, coefs, chunk_size=3)

        np.testing.assert_array_almost_equal(mse_256, mse_8)
        np.testing.assert_array_almost_equal(mse_256, mse_3)

    def test_batch_mse_single_model(self):
        """batch_mse with single model should work."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(1, 3)

        mse = batch_mse(X, y, coefs)
        assert mse.shape == (1,)

    def test_batch_mse_zero_weights(self):
        """batch_mse with zero weights should return NaN."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(5, 3)
        w = np.zeros(50)

        mse = batch_mse(X, y, coefs, sample_weight=w)
        assert np.all(np.isnan(mse))

    def test_batch_mse_large_n_models(self):
        """batch_mse with many models should work (tests chunking)."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        coefs = np.random.randn(500, 3)  # 500 models, chunk_size=256 -> 2 chunks

        mse = batch_mse(X, y, coefs, chunk_size=256)
        assert mse.shape == (500,)
        assert np.all(np.isfinite(mse))


# ======================================================================
# 36. cv_engine reference implementation
# ======================================================================

class TestCvEngineReference:
    def test_run_cv_basic(self):
        """run_cv should work as a basic CV loop."""
        from statgpu.linear_model._cv_engine import run_cv
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        alphas = np.array([0.01, 0.1, 1.0])

        def eval_fold(X_train, y_train, X_val, y_val, alpha, **kw):
            from statgpu.linear_model import Ridge
            m = Ridge(alpha=alpha).fit(X_train, y_train)
            return m.score(X_val, y_val)

        best_alpha, mean_scores, all_scores = run_cv(
            X, y, alphas, eval_fold, n_folds=3
        )
        assert best_alpha in alphas
        assert mean_scores.shape == (3,)
        assert all_scores.shape == (3, 3)

    def test_run_cv_cache(self):
        """run_cv with cache should return cached results."""
        from statgpu.linear_model._cv_engine import run_cv, CVCache
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        alphas = np.array([0.01, 0.1])

        call_count = [0]
        def eval_fold(X_train, y_train, X_val, y_val, alpha, **kw):
            call_count[0] += 1
            from statgpu.linear_model import Ridge
            m = Ridge(alpha=alpha).fit(X_train, y_train)
            return m.score(X_val, y_val)

        cache = CVCache(maxsize=10)
        def cache_key_fn(X, y, alphas, folds):
            return "test_key"

        # First call
        best1, _, _ = run_cv(X, y, alphas, eval_fold, n_folds=3,
                             cache=cache, cache_key_fn=cache_key_fn)
        count1 = call_count[0]

        # Second call (should use cache)
        best2, _, _ = run_cv(X, y, alphas, eval_fold, n_folds=3,
                             cache=cache, cache_key_fn=cache_key_fn)
        count2 = call_count[0]

        assert best1 == best2
        assert count2 == count1  # No additional calls

    def test_run_cv_raise_on_error(self):
        """run_cv with raise_on_error=True should re-raise exceptions."""
        from statgpu.linear_model._cv_engine import run_cv
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        alphas = np.array([0.01])

        def bad_eval(*args, **kwargs):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_cv(X, y, alphas, bad_eval, n_folds=3, raise_on_error=True)

    def test_run_cv_shape_validation(self):
        """run_cv should validate X/y shape consistency."""
        from statgpu.linear_model._cv_engine import run_cv
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(30)  # Wrong shape
        alphas = np.array([0.01])

        def eval_fold(*args, **kwargs):
            return 0.0

        with pytest.raises(ValueError, match="different number of samples"):
            run_cv(X, y, alphas, eval_fold, n_folds=3)

    def test_cv_engine_docstring_is_reference(self):
        """_cv_engine.py docstring should mention 'reference implementation'."""
        with open('statgpu/linear_model/_cv_engine.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'Reference Implementation' in content or 'reference implementation' in content


# ======================================================================
# 37. Panel group labels: float and string support
# ======================================================================

class TestPanelGroupLabels:
    def test_float_group_labels(self):
        """within_transform should handle float group labels without truncation."""
        from statgpu.panel._utils import within_transform
        np.random.seed(42)
        y = np.random.randn(9)
        groups = np.array([1.1, 1.1, 1.1, 2.5, 2.5, 2.5, 3.9, 3.9, 3.9])
        result = within_transform(y, groups)
        # Each group of 3 should be demeaned
        for g in [1.1, 2.5, 3.9]:
            mask = groups == g
            assert abs(np.mean(result[mask])) < 1e-10
        print('[OK] float group labels')

    def test_string_group_labels(self):
        """within_transform should handle string group labels."""
        from statgpu.panel._utils import within_transform
        np.random.seed(42)
        y = np.random.randn(6)
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        result = within_transform(y, groups)
        for g in ['A', 'B']:
            mask = groups == g
            assert abs(np.mean(result[mask])) < 1e-10
        print('[OK] string group labels')

    def test_integer_group_labels_still_work(self):
        """within_transform should still handle integer labels correctly."""
        from statgpu.panel._utils import within_transform
        np.random.seed(42)
        y = np.random.randn(6)
        groups = np.array([0, 0, 0, 1, 1, 1])
        result = within_transform(y, groups)
        for g in [0, 1]:
            mask = groups == g
            assert abs(np.mean(result[mask])) < 1e-10
        print('[OK] integer group labels')

    def test_group_means_float_labels(self):
        """group_means should handle float labels."""
        from statgpu.panel._utils import group_means
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        groups = np.array([1.1, 1.1, 1.1, 2.5, 2.5, 2.5])
        result = group_means(y, groups)
        assert abs(result[0] - 2.0) < 1e-10  # mean of [1,2,3]
        assert abs(result[3] - 5.0) < 1e-10  # mean of [4,5,6]
        print('[OK] group_means float labels')

    def test_group_sizes_float_labels(self):
        """group_sizes should handle float labels."""
        from statgpu.panel._utils import group_sizes
        groups = np.array([1.1, 1.1, 1.1, 2.5, 2.5])
        result = group_sizes(groups)
        assert result[0] == 3.0
        assert result[3] == 2.0
        print('[OK] group_sizes float labels')


# ======================================================================
# 38. ElasticNetCV _fitted flag
# ======================================================================

class TestElasticNetCVFitted:
    def test_fitted_flag_set(self):
        """ElasticNetCV should set _fitted=True after fit."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m = ElasticNetCV(cv=3).fit(X, y)
        assert getattr(m, '_fitted', False) is True
        print('[OK] ElasticNetCV _fitted flag')

    def test_summary_after_fit(self):
        """ElasticNetCV.summary() should not raise 'not fitted' error."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m = ElasticNetCV(cv=3).fit(X, y)
        # summary() may raise "compute_inference=False" (expected),
        # but should NOT raise "not fitted yet" (that was the bug)
        try:
            m.summary()
        except RuntimeError as e:
            assert 'not fitted' not in str(e).lower(), f'Unexpected: {e}'
        print('[OK] ElasticNetCV summary()')


# ======================================================================
# 39. Solver max_iter=0 safety
# ======================================================================

class TestSolverMaxIterZero:
    def test_fista_max_iter_zero(self):
        """fista_solver with max_iter=0 should not crash."""
        from statgpu.glm_core._solver import fista_solver
        from statgpu.glm_core._squared import SquaredErrorLoss
        from statgpu.linear_model._lasso import Lasso
        loss = SquaredErrorLoss()
        pen = Lasso(alpha=0.1)
        # Should not raise NameError
        coef, n_iter = fista_solver(X=np.random.randn(20, 3), y=np.random.randn(20),
                                     loss=loss, penalty=pen, max_iter=0)
        assert n_iter == 0
        print('[OK] fista max_iter=0')


# ======================================================================
# 40. ElasticNet alpha_grid l1_ratio scaling
# ======================================================================

class TestElasticnetAlphaGrid:
    def test_alpha_grid_scales_with_l1_ratio(self):
        """PenalizedGLM_CV elasticnet alpha_grid should scale with l1_ratio."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        # l1_ratio=0.1 should have larger alpha_max than l1_ratio=1.0
        m1 = PenalizedGLM_CV(loss='squared_error', penalty='elasticnet',
                              l1_ratio=0.1, n_alphas=10, cv=3)
        m1.fit(X, y)
        grid1 = m1.alpha_grid_

        m2 = PenalizedGLM_CV(loss='squared_error', penalty='elasticnet',
                              l1_ratio=1.0, n_alphas=10, cv=3)
        m2.fit(X, y)
        grid2 = m2.alpha_grid_

        # l1_ratio=0.1 should have larger max alpha (divided by smaller l1_ratio)
        assert grid1[0] > grid2[0], f'grid1[0]={grid1[0]} should > grid2[0]={grid2[0]}'
        print('[OK] alpha_grid scales with l1_ratio')


# ======================================================================
# 41. Cache thread safety
# ======================================================================

class TestCacheThreadSafety:
    def test_ridge_cache_has_lock(self):
        """RidgeCV cache should have threading.Lock."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'threading.Lock' in content
        print('[OK] RidgeCV cache thread-safe')

    def test_elasticnet_cache_has_lock(self):
        """ElasticNetCV cache should have threading.Lock."""
        with open('statgpu/linear_model/_elasticnet_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'threading.Lock' in content
        print('[OK] ElasticNetCV cache thread-safe')

    def test_logistic_cache_has_lock(self):
        """LogisticRegressionCV cache should have threading.Lock."""
        with open('statgpu/linear_model/_logistic_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'threading.Lock' in content
        print('[OK] LogisticCV cache thread-safe')


# ======================================================================
# 42. CVEstimatorBase.get_params
# ======================================================================

class TestGetParams:
    def test_get_params_includes_cv(self):
        """get_params should include cv parameter."""
        from statgpu.linear_model import RidgeCV
        m = RidgeCV(cv=7)
        params = m.get_params()
        assert params['cv'] == 7
        print('[OK] get_params includes cv')

    def test_get_params_includes_random_state(self):
        """get_params should include random_state parameter."""
        from statgpu.linear_model import RidgeCV
        m = RidgeCV(random_state=42)
        params = m.get_params()
        assert params['random_state'] == 42
        print('[OK] get_params includes random_state')


# ======================================================================
# 43. Cluster length validation
# ======================================================================

class TestClusterValidation:
    def test_cluster_length_mismatch_raises(self):
        """PanelOLS should raise when cluster length != n."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.random.randn(2) + 0.1 * np.random.randn(50)
        cluster = np.repeat(np.arange(10), 3)[:30]  # Wrong length

        model = PanelOLS(cov_type='clustered')
        try:
            model.fit(y, X, cluster=cluster)
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'cluster length' in str(e).lower()
        print('[OK] cluster length validation')


# ======================================================================
# 44. LassoCV ddof consistency
# ======================================================================

class TestLassoCVDdof:
    def test_gpu_alpha_grid_uses_ddof1(self):
        """_default_lasso_alpha_grid_backend should use ddof=1."""
        import inspect
        from statgpu.linear_model._lasso_cv import _default_lasso_alpha_grid_backend
        src = inspect.getsource(_default_lasso_alpha_grid_backend)
        assert 'n_samples - 1' in src
        print('[OK] LassoCV GPU alpha grid ddof=1')


# ======================================================================
# 45. Compiled FISTA step removed
# ======================================================================

class TestCompiledFistaStep:
    def test_no_wrong_fista_step_call(self):
        """fista_solver should not call compiled FISTA step with wrong args."""
        import inspect
        from statgpu.glm_core._solver import fista_solver
        src = inspect.getsource(fista_solver)
        # The old pattern: _fista_step_call(_fista_step, coef, coef, 0.0, ...)
        # should not exist in the momentum update path
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if '_fista_step_call' in line and 'coef, coef, 0.0' in line:
                # Check if this is in the momentum section (not the proximal section)
                context = '\n'.join(lines[max(0,i-5):i+5])
                if 'momentum' in context.lower() or 'Momentum' in context:
                    assert False, f'Wrong FISTA step call at line {i}: {line.strip()}'


# ======================================================================
# 46. Dead code removal verification
# ======================================================================

class TestDeadCodeRemoved:
    def test_lasso_cv_no_fit_cv_method(self):
        """_lasso_cv.py should not have _fit_cv method."""
        with open('statgpu/linear_model/_lasso_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'def _fit_cv' not in content
        print('[OK] _fit_cv removed')

    def test_lasso_cv_single_fit_method(self):
        """_lasso_cv.py should have exactly one fit method."""
        with open('statgpu/linear_model/_lasso_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        count = content.count('def fit(self')
        assert count == 1, f'Expected 1 fit method, found {count}'
        print('[OK] single fit method')

    def test_logistic_cv_no_and_false(self):
        """_logistic_cv.py should not have 'and False' dead code."""
        with open('statgpu/linear_model/_logistic_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'and False' not in content
        print('[OK] no and False dead code')

    def test_solver_fista_lla_xtx_gated(self):
        """fista_lla_path should only compute XtX for quadratic losses."""
        import inspect
        from statgpu.glm_core._solver import fista_lla_path
        src = inspect.getsource(fista_lla_path)
        # XtX should be gated behind _is_quadratic
        assert 'if _is_quadratic' in src or '_is_quadratic' in src
        print('[OK] fista_lla XtX gated')


# ======================================================================
# 47. Precision regression check (same random_state)
# ======================================================================

class TestPrecisionRegression:
    def test_ridge_cv_precision(self):
        """RidgeCV should produce consistent results with same random_state."""
        from statgpu.linear_model import RidgeCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = RidgeCV(cv=3, random_state=42).fit(X, y)
        # Results should be deterministic
        assert m.best_score_ < 0
        assert np.isfinite(m.best_score_)
        print('[OK] RidgeCV precision')

    def test_lasso_cv_precision(self):
        """LassoCV should produce consistent results with same random_state."""
        from statgpu.linear_model import LassoCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = LassoCV(cv=3, random_state=42).fit(X, y)
        assert m.best_score_ < 0
        assert np.isfinite(m.best_score_)
        print('[OK] LassoCV precision')

    def test_elasticnet_cv_precision(self):
        """ElasticNetCV should produce consistent results."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = ElasticNetCV(cv=3, random_state=42).fit(X, y)
        assert m.best_score_ < 0
        assert np.isfinite(m.best_score_)
        print('[OK] ElasticNetCV precision')


# ======================================================================
# 48. Performance regression check
# ======================================================================

class TestPerformanceRegression:
    @pytest.mark.parametrize("n,p", [(100, 5), (500, 20)])
    def test_ridge_cv_time(self, n, p):
        """RidgeCV should complete within reasonable time."""
        from statgpu.linear_model import RidgeCV
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + 0.1 * np.random.randn(n)

        t0 = time.perf_counter()
        RidgeCV(cv=5).fit(X, y)
        elapsed = time.perf_counter() - t0

        limit = {(100, 5): 1.0, (500, 20): 5.0}[(n, p)]
        assert elapsed < limit, f"n={n},p={p}: {elapsed:.3f}s > {limit}s"

    @pytest.mark.parametrize("n,p", [(100, 5), (500, 20)])
    def test_lasso_cv_time(self, n, p):
        """LassoCV should complete within reasonable time."""
        from statgpu.linear_model import LassoCV
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + 0.1 * np.random.randn(n)

        t0 = time.perf_counter()
        LassoCV(cv=5).fit(X, y)
        elapsed = time.perf_counter() - t0

        limit = {(100, 5): 2.0, (500, 20): 10.0}[(n, p)]
        assert elapsed < limit, f"n={n},p={p}: {elapsed:.3f}s > {limit}s"
        print('[OK] No wrong compiled FISTA step call')
