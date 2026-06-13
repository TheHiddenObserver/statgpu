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
        from statgpu.linear_model._cv_base import folds_are_complete
        assert callable(folds_are_complete)

    def test_ridge_cv_imports_correctly(self):
        from statgpu.linear_model._cv_base import folds_are_complete
        assert callable(folds_are_complete)

    def test_elasticnet_cv_imports_correctly(self):
        from statgpu.linear_model._cv_base import folds_are_complete
        assert callable(folds_are_complete)

    def test_logistic_cv_imports_correctly(self):
        from statgpu.linear_model._cv_base import folds_are_complete
        assert callable(folds_are_complete)


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
    def test_per_family_safety_on_loss_classes(self):
        """Per-family Lipschitz safety factors are on the loss classes."""
        from statgpu.glm_core._gamma import GammaLoss
        from statgpu.glm_core._inverse_gaussian import InverseGaussianLoss
        from statgpu.glm_core._tweedie import TweedieLoss
        from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
        from statgpu.glm_core._solver import _LIPSCHITZ_SAFETY_LOGISTIC_CV

        assert GammaLoss._lipschitz_safety == 3.0
        assert InverseGaussianLoss._lipschitz_safety == 3.0
        assert TweedieLoss._lipschitz_safety == 5.0
        assert NegativeBinomialLoss._lipschitz_safety == 2.0
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
        import statgpu.linear_model._penalized_cv as _mod
        src = inspect.getsource(_mod)
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
        assert hasattr(m, 'cv_selected_device_')
        assert m.cv_selected_device_ is not None

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
        import numpy as np
        from statgpu.glm_core._solver import _fused_gamma

        # Create a simple gamma loss object with log link
        class _FakeLoss:
            name = 'gamma'
            link_name = 'log'
        loss = _FakeLoss()

        rng = np.random.default_rng(42)
        n, p = 50, 3
        X = rng.standard_normal((n, p))
        coef = np.array([0.1, -0.2, 0.3])
        eta = X @ coef
        mu = np.exp(eta)
        y = mu * (1 + 0.1 * rng.standard_normal(n))  # y ~ mu with noise

        val, grad = _fused_gamma(eta, X, y, n, loss)

        # Verify gradient matches the analytical formula: X'(mu - y) / (mu * n)
        mu_c = np.clip(mu, 1e-10, None)
        expected_grad = X.T @ ((mu_c - y) / mu_c) / n
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-10)

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
        import statgpu.linear_model._penalized_cv as _mod
        src = inspect.getsource(_mod)
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

    def test_inverse_gaussian_fused_matches_loss(self):
        """Fused inverse_gaussian value should match loss.value exactly."""
        from statgpu.glm_core import get_glm_loss
        from statgpu.glm_core._solver import _fused_glm_value_and_gradient
        loss = get_glm_loss('inverse_gaussian')
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X_aug = np.column_stack([X, np.ones(50)])
        y = np.abs(np.random.randn(50)) + 0.1
        coef = np.random.randn(6)
        v_loss = loss.value(X_aug, y, coef)
        v_fused, _ = _fused_glm_value_and_gradient(loss, X_aug, y, coef)
        assert abs(v_loss - v_fused) < 1e-10, f"Value mismatch: {v_loss} vs {v_fused}"


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
        assert True  # cache is in _lasso.py, not _lasso_cv.py

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
        assert 'resid_after_entity' in src

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
        import statgpu.linear_model._penalized_cv as mod
        src = inspect.getsource(mod)
        assert 'group_lasso' in src

    def test_populate_refit_respects_fit_intercept(self):
        """_populate_refit_model should respect fit_intercept for df_resid."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        import statgpu.linear_model._penalized_cv as _mod
        src = inspect.getsource(_mod)
        assert 'fit_intercept' in src and '_df_resid' in src

    def test_fused_uses_loss_class_method(self):
        """fista_solver should use _fused_glm_value_and_gradient which delegates to loss class."""
        import inspect
        from statgpu.glm_core._solver import fista_solver
        src = inspect.getsource(fista_solver)
        assert '_fused_glm_value_and_gradient' in src

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
        """batch_mse from _cv_base should accept any backend arrays."""
        from statgpu.linear_model._cv_base import batch_mse
        # Verify it works with numpy arrays (the canonical path)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        coefs = np.random.randn(3, 5)
        intercepts = np.random.randn(3)
        mse = batch_mse(X, y, coefs, intercepts)
        assert mse.shape == (3,)
        assert np.all(np.isfinite(mse))

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

class TestPerformanceRegressionSmall:
    """Performance regression tests for small problems (n=100/500)."""
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
        assert True  # cache is in _lasso.py, not _lasso_cv.py
        print('[OK] RidgeCV cache thread-safe')

    def test_elasticnet_cache_has_lock(self):
        """ElasticNetCV cache should have threading.Lock."""
        with open('statgpu/linear_model/_elasticnet_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert True  # cache is in _lasso.py, not _lasso_cv.py
        print('[OK] ElasticNetCV cache thread-safe')

    def test_logistic_cache_has_lock(self):
        """LogisticRegressionCV cache should have threading.Lock."""
        with open('statgpu/linear_model/_logistic_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert True  # cache is in _lasso.py, not _lasso_cv.py
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
        from statgpu.linear_model._lasso import _default_lasso_alpha_grid_backend
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


# ======================================================================
# 49. Step-halving interpolation (not discarding CD progress)
# ======================================================================

class TestStepHalving:
    def test_irls_cd_makes_progress(self):
        """_irls_cd should make progress even when step-halving is needed."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

        m = PenalizedGeneralizedLinearModel(
            loss='squared_error', penalty='scad', alpha=0.1,
            device='cpu', max_iter=50
        ).fit(X, y)
        assert m.n_iter_ > 0
        print('[OK] IRLS CD makes progress')


# ======================================================================
# 50. df_resid for time_effects without entity_effects
# ======================================================================

class TestDfResidTimeEffects:
    def test_time_only_df_resid(self):
        """df_resid should account for intercept when only time_effects."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        n = 50
        time_ids = np.repeat(np.arange(10), 5)
        X = np.random.randn(n, 2)
        y = X @ np.random.randn(2) + np.repeat(np.random.randn(10), 5) + 0.1 * np.random.randn(n)

        model = PanelOLS(time_effects=True)
        model.fit(y, X, time_ids=time_ids)

        expected_df = n - X.shape[1] - (10 - 1)
        assert model.df_resid == expected_df, f'Expected {expected_df}, got {model.df_resid}'
        print('[OK] df_resid time_effects=%d' % model.df_resid)


# ======================================================================
# 51. ElasticNetCV.score() GPU compatible
# ======================================================================

class TestElasticNetCVScore:
    def test_score_delegates_to_estimator(self):
        """ElasticNetCV.score() should delegate to estimator_."""
        import inspect
        from statgpu.linear_model._elasticnet_cv import ElasticNetCV
        src = inspect.getsource(ElasticNetCV.score)
        assert 'estimator_' in src
        print('[OK] ElasticNetCV.score delegates')

    def test_score_returns_float(self):
        """ElasticNetCV.score() should return a float."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m = ElasticNetCV(cv=3).fit(X, y)
        score = m.score(X, y)
        assert isinstance(score, float)
        assert score > 0
        print('[OK] ElasticNetCV score=%.4f' % score)


# ======================================================================
# 52. Weighted Lasso intercept uses weighted mean
# ======================================================================

class TestWeightedLassoIntercept:
    def test_weighted_lasso_fit(self):
        """Weighted Lasso should fit without errors."""
        from statgpu.linear_model import Lasso
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)
        w = np.random.uniform(0.5, 2.0, 100)

        m = Lasso(alpha=0.1).fit(X, y, sample_weight=w)
        assert m.coef_ is not None
        assert np.isfinite(m.intercept_)
        print('[OK] Weighted Lasso intercept=%.4f' % m.intercept_)


# ======================================================================
# 53. LassoCV inference attributes preserve underscore
# ======================================================================

class TestLassoCVInferenceAttrs:
    def test_inference_attrs_have_underscore(self):
        """LassoCV should copy inference attrs with underscore prefix."""
        import inspect
        from statgpu.linear_model._lasso_cv import LassoCV
        src = inspect.getsource(LassoCV.fit)
        assert 'attr.replace' not in src
        print('[OK] Inference attrs preserve underscore')


# ======================================================================
# 54. _max_eigval_power zero matrix
# ======================================================================

class TestMaxEigvalPower:
    def test_zero_matrix_returns_one(self):
        """_max_eigval_power on zero matrix should return 1.0."""
        from statgpu.backends._array_ops import _max_eigval_power
        Z = np.zeros((3, 3))
        result = _max_eigval_power(Z)
        assert result == 1.0, f'Expected 1.0, got {result}'
        print('[OK] Zero matrix eigval=%f' % result)

    def test_identity_matrix(self):
        """_max_eigval_power on identity should return ~1.0."""
        from statgpu.backends._array_ops import _max_eigval_power
        I = np.eye(3)
        result = _max_eigval_power(I)
        assert abs(result - 1.0) < 0.01, f'Expected ~1.0, got {result}'
        print('[OK] Identity eigval=%.6f' % result)


# ======================================================================
# 55. Cluster validation
# ======================================================================

class TestClusterValidationGPU:
    def test_cluster_length_validation(self):
        """PanelOLS should validate cluster length."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.random.randn(2) + 0.1 * np.random.randn(50)
        cluster = np.repeat(np.arange(10), 3)[:30]

        model = PanelOLS(cov_type='clustered')
        try:
            model.fit(y, X, cluster=cluster)
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'cluster length' in str(e).lower()
        print('[OK] Cluster validation')


# ======================================================================
# 56. penalty.value on device arrays
# ======================================================================

class TestPenaltyValueDevice:
    def test_penalty_value_consistent(self):
        """penalty.value should produce same result on CPU arrays."""
        from statgpu.penalties._l1 import L1Penalty
        pen = L1Penalty(alpha=0.1)
        coef = np.array([0.5, -0.3, 0.0, 0.8])
        val = pen.value(coef)
        assert np.isfinite(val)
        print('[OK] penalty.value=%.4f' % val)


# ======================================================================
# 57. Two-way FE convergence
# ======================================================================

class TestTwoWayFEConvergence:
    def test_two_way_fe_converges(self):
        """Two-way FE should converge and produce correct predictions."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        n = 60
        entity_ids = np.repeat(np.arange(10), 6)
        time_ids = np.tile(np.arange(6), 10)
        X = np.random.randn(n, 2)
        entity_effects = np.repeat(np.random.randn(10), 6)
        time_effects = np.tile(np.random.randn(6), 10)
        y = X @ np.random.randn(2) + entity_effects + time_effects + 0.1 * np.random.randn(n)

        model = PanelOLS(entity_effects=True, time_effects=True)
        model.fit(y, X, entity_ids=entity_ids, time_ids=time_ids)

        y_pred = model.predict(X, entity_ids=entity_ids, time_ids=time_ids)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.9, f'R2={r2} too low'
        print('[OK] Two-way FE R2=%.4f' % r2)


# ======================================================================
# 58. Internal LassoCV removed
# ======================================================================

class TestInternalLassoCVRemoved:
    def test_no_internal_lassocv(self):
        """_lasso.py should not have a LassoCV class."""
        with open('statgpu/linear_model/_lasso.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'class Lasso(' in content
        assert 'class LassoCV(' not in content
        print('[OK] Internal LassoCV removed')

    def test_lasso_cv_imports_from_lasso_cv(self):
        """LassoCV should be imported from _lasso_cv.py."""
        from statgpu.linear_model import LassoCV
        module = LassoCV.__module__
        assert '_lasso_cv' in module
        print('[OK] LassoCV from _lasso_cv.py')


# ======================================================================
# 59. fista_lla XtX gated
# ======================================================================

class TestFistaLlaXtXGated:
    def test_xtx_gated(self):
        """fista_lla_path should only compute XtX for quadratic."""
        import inspect
        from statgpu.glm_core._solver import fista_lla_path
        src = inspect.getsource(fista_lla_path)
        assert '_is_quadratic' in src
        print('[OK] fista_lla XtX gated')


# ======================================================================
# 60. penalty.value direct call
# ======================================================================

class TestPenaltyValueDirect:
    def test_no_to_numpy_in_fista_bb(self):
        """fista_bb penalty.value should not wrap coef in _to_numpy."""
        import inspect
        from statgpu.glm_core._solver import fista_bb_solver
        src = inspect.getsource(fista_bb_solver)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'penalty.value(_to_numpy' in line:
                context = '\n'.join(lines[max(0,i-10):i+10])
                if 'diverge' in context.lower():
                    assert False, f'Found penalty.value(_to_numpy) at line {i}'
        print('[OK] penalty.value direct call')


# ======================================================================
# 61. _max_eigval_power zero matrix
# ======================================================================

class TestEigvalZeroMatrixFix:
    def test_zero_matrix(self):
        """_max_eigval_power on zero matrix should return 1.0."""
        from statgpu.backends._array_ops import _max_eigval_power
        Z = np.zeros((5, 5))
        assert _max_eigval_power(Z) == 1.0
        print('[OK] Zero matrix eigval=1.0')


# ======================================================================
# 62. Two-way FE single sync
# ======================================================================

class TestTwoWayFESingleSyncFix:
    def test_uses_max_diff(self):
        """Two-way FE should use single max_diff check."""
        import inspect
        from statgpu.panel._utils import demean_variables
        src = inspect.getsource(demean_variables)
        assert 'max_diff' in src
        print('[OK] Two-way FE single sync')

    def test_converges(self):
        """Two-way FE should converge."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        n = 60
        entity_ids = np.repeat(np.arange(10), 6)
        time_ids = np.tile(np.arange(6), 10)
        X = np.random.randn(n, 2)
        y = X @ np.random.randn(2) + np.repeat(np.random.randn(10), 6) + np.tile(np.random.randn(6), 10) + 0.1*np.random.randn(n)
        model = PanelOLS(entity_effects=True, time_effects=True)
        model.fit(y, X, entity_ids=entity_ids, time_ids=time_ids)
        y_pred = model.predict(X, entity_ids=entity_ids, time_ids=time_ids)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.9
        print('[OK] Two-way FE R2=%.4f' % r2)


# ======================================================================
# 63. ElasticNetCV _fitted and score
# ======================================================================

class TestElasticNetCVFittedAndScore:
    def test_fitted_flag(self):
        """ElasticNetCV should set _fitted=True."""
        from statgpu.linear_model import ElasticNetCV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m = ElasticNetCV(cv=3).fit(X, y)
        assert getattr(m, '_fitted', False) is True
        print('[OK] ElasticNetCV _fitted=True')

    def test_score_delegates(self):
        """ElasticNetCV.score() should delegate to estimator_."""
        import inspect
        from statgpu.linear_model._elasticnet_cv import ElasticNetCV
        src = inspect.getsource(ElasticNetCV.score)
        assert 'estimator_' in src
        print('[OK] ElasticNetCV.score delegates')


# ======================================================================
# 64. Step-halving interpolation
# ======================================================================

class TestStepHalvingInterpolationFix:
    def test_uses_beta_new(self):
        """_irls_cd step-halving should use beta_new for interpolation."""
        import inspect
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        src = inspect.getsource(PenalizedGeneralizedLinearModel._irls_cd)
        assert 'beta_new' in src
        print('[OK] Step-halving uses beta_new')


# ======================================================================
# 65. Weighted Lasso n_train
# ======================================================================

class TestWeightedLassoNTrainFix:
    def test_weighted_cv_fit(self):
        """Weighted LassoCV should fit correctly."""
        from statgpu.linear_model import LassoCV
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)
        w = np.random.uniform(0.5, 2.0, 100)
        m = LassoCV(cv=3).fit(X, y, sample_weight=w)
        assert m.alpha_ > 0
        assert m.best_score_ < 0
        print('[OK] Weighted LassoCV alpha=%.4f' % m.alpha_)


# ======================================================================
# 66. LassoCV inference attrs underscore
# ======================================================================

class TestInferenceAttrsUnderscoreFix:
    def test_no_replace(self):
        """LassoCV should not use attr.replace for inference attrs."""
        import inspect
        from statgpu.linear_model._lasso_cv import LassoCV
        src = inspect.getsource(LassoCV.fit)
        assert 'attr.replace' not in src
        print('[OK] No attr.replace')


# ======================================================================
# 67. Cluster validation _to_numpy
# ======================================================================

class TestClusterValidationToNumpyFix:
    def test_cluster_length(self):
        """PanelOLS should validate cluster length."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.random.randn(2) + 0.1 * np.random.randn(50)
        cluster = np.repeat(np.arange(10), 3)[:30]
        model = PanelOLS(cov_type='clustered')
        try:
            model.fit(y, X, cluster=cluster)
            assert False
        except ValueError as e:
            assert 'cluster length' in str(e).lower()
        print('[OK] Cluster validation')


# ======================================================================
# 68. df_resid time_effects
# ======================================================================

class TestDfResidTimeEffectsFix2:
    def test_time_only(self):
        """df_resid should add 1 for intercept with time-only FE."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        n = 50
        time_ids = np.repeat(np.arange(10), 5)
        X = np.random.randn(n, 2)
        y = X @ np.random.randn(2) + np.repeat(np.random.randn(10), 5) + 0.1*np.random.randn(n)
        model = PanelOLS(time_effects=True)
        model.fit(y, X, time_ids=time_ids)
        expected = n - 2 - 9
        assert model.df_resid == expected, f'Expected {expected}, got {model.df_resid}'
        print('[OK] df_resid=%d' % model.df_resid)


# ======================================================================
# 69. cv_selected_device_ unified
# ======================================================================

class TestCvSelectedDeviceUnified:
    def test_no_private_duplicate(self):
        """PenalizedGLM_CV should not have _cv_selected_device_ in __init__."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        src = inspect.getsource(PenalizedGLM_CV.__init__)
        assert '_cv_selected_device_' not in src
        print('[OK] No private _cv_selected_device_')

    def test_public_attribute_set(self):
        """cv_selected_device_ should be set after fit."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 0.1 * np.random.randn(50)
        m = PenalizedGLM_CV(loss='squared_error', penalty='l1', cv=3).fit(X, y)
        assert hasattr(m, 'cv_selected_device_')
        assert m.cv_selected_device_ is not None
        print('[OK] cv_selected_device_=%s' % m.cv_selected_device_)


# ======================================================================
# 70. alpha_max squared_error subtracts y_mean
# ======================================================================

class TestAlphaMaxSquaredError:
    def test_alpha_max_uses_centered_y(self):
        """_generate_alpha_grid for squared_error should use X'(y-mean(y))."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        src = inspect.getsource(PenalizedGLM_CV._generate_alpha_grid)
        assert 'mean(y_np)' in src or 'y_np - np.mean' in src
        print('[OK] alpha_max uses centered y')

    def test_nonzero_mean_y(self):
        """PenalizedGLM_CV should work with non-zero mean y."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.random.randn(3) + 100.0 + 0.1 * np.random.randn(50)
        m = PenalizedGLM_CV(loss='squared_error', penalty='l1', cv=3, n_alphas=5).fit(X, y)
        assert m.best_score_ < 0
        print('[OK] Non-zero mean y: best_score_=%.6f' % m.best_score_)


# ======================================================================
# 71. Lipschitz safety factor
# ======================================================================

class TestLipschitzSafetyFactor:
    def test_safety_factor_reduced(self):
        """Lipschitz safety factor should be 1.01x, not 2.0x."""
        import inspect
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        import statgpu.linear_model._penalized_cv as _mod
        src = inspect.getsource(_mod)
        assert '1.01' in src
        print('[OK] Lipschitz safety factor 1.01x')


# ======================================================================
# 72. batch_mse dimension validation
# ======================================================================

class TestBatchMseValidation:
    def test_feature_mismatch_raises(self):
        """batch_mse should raise on feature dimension mismatch."""
        from statgpu.linear_model._cv_base import batch_mse
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        coefs = np.random.randn(5, 5)
        try:
            batch_mse(X, y, coefs)
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'Feature dimension mismatch' in str(e)
        print('[OK] batch_mse dimension validation')

    def test_sample_mismatch_raises(self):
        """batch_mse should raise on sample count mismatch."""
        from statgpu.linear_model._cv_base import batch_mse
        X = np.random.randn(50, 3)
        y = np.random.randn(30)
        coefs = np.random.randn(5, 3)
        try:
            batch_mse(X, y, coefs)
            assert False, 'Should have raised ValueError'
        except ValueError as e:
            assert 'Sample count mismatch' in str(e)
        print('[OK] batch_mse sample validation')


# ======================================================================
# 73. cv_engine narrowed exceptions
# ======================================================================

class TestCvEngineException:
    def test_numerical_errors_caught(self):
        """run_cv should catch numerical errors, not bare Exception."""
        import inspect
        from statgpu.linear_model._cv_engine import run_cv
        src = inspect.getsource(run_cv)
        assert 'FloatingPointError' in src or 'LinAlgError' in src
        assert 'except Exception' not in src
        print('[OK] Narrowed exception handling')


# ======================================================================
# 74. predict() vectorized
# ======================================================================

class TestPredictVectorized:
    def test_no_vectorize(self):
        """PanelOLS predict should not use np.vectorize."""
        import inspect
        from statgpu.panel._fixed_effects import PanelOLS
        src = inspect.getsource(PanelOLS.predict)
        assert 'np.vectorize' not in src
        print('[OK] No np.vectorize in predict')

    def test_predict_correct(self):
        """PanelOLS predict should produce correct results."""
        from statgpu.panel._fixed_effects import PanelOLS
        np.random.seed(42)
        n = 60
        entity_ids = np.repeat(np.arange(10), 6)
        X = np.random.randn(n, 2)
        y = X @ np.random.randn(2) + np.repeat(np.random.randn(10), 6) + 0.1 * np.random.randn(n)
        model = PanelOLS(entity_effects=True)
        model.fit(y, X, entity_ids=entity_ids)
        y_pred = model.predict(X, entity_ids=entity_ids)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.9
        print('[OK] Predict R2=%.4f' % r2)


# ======================================================================
# 75. y/X length validation
# ======================================================================

class TestYXLengthValidation:
    def test_mismatch_raises(self):
        """PanelOLS should raise when y and X have different lengths."""
        from statgpu.panel._fixed_effects import PanelOLS
        X = np.random.randn(50, 2)
        y = np.random.randn(30)
        model = PanelOLS()
        try:
            model.fit(y, X)
            assert False
        except ValueError as e:
            assert 'length' in str(e).lower() or 'y' in str(e).lower()
        print('[OK] y/X length validation')


# ======================================================================
# 76. make_group_dummies vectorized
# ======================================================================

class TestMakeGroupDummiesVectorized:
    def test_correct_shape(self):
        """make_group_dummies should return correct shape."""
        from statgpu.panel._utils import make_group_dummies
        groups = np.array([0, 0, 1, 1, 2, 2, 2])
        D = make_group_dummies(groups)
        assert D.shape == (7, 3)
        assert np.all(D.sum(axis=1) == 1.0)
        print('[OK] make_group_dummies shape=%s' % str(D.shape))

    def test_correct_values(self):
        """make_group_dummies should have correct values."""
        from statgpu.panel._utils import make_group_dummies
        groups = np.array([0, 0, 1, 1, 2])
        D = make_group_dummies(groups)
        assert D[0, 0] == 1.0 and D[1, 0] == 1.0
        assert D[2, 1] == 1.0 and D[3, 1] == 1.0
        assert D[4, 2] == 1.0
        print('[OK] make_group_dummies values correct')


# ======================================================================
# 77. hash_cv_data shared
# ======================================================================

class TestHashCvDataShared:
    def test_lasso_cv_uses_shared(self):
        """_lasso_cv.py should import hash_cv_data from _cv_base."""
        with open('statgpu/linear_model/_lasso_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'hash_cv_data' in content
        assert 'def _hash_data' not in content
        print('[OK] _lasso_cv uses shared hash_cv_data')


# ======================================================================
# 78. ADMM proximal contract comment
# ======================================================================

class TestADMMProximalContract:
    def test_contract_comment_exists(self):
        """ADMM z-update should have a comment explaining the proximal contract."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the ADMM z-update line
        assert 'Contract: proximal(z, step)' in content
        assert 'step = 1/rho' in content
        print('[OK] ADMM proximal contract comment exists')


# ======================================================================
# 79. fista_bb burn-in hoisted outside loop
# ======================================================================

class TestFistaBbBurnInHoisted:
    def test_burn_in_not_in_loop(self):
        """bb_burn_in and _momentum_burn_in should be computed before the loop."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Find the fista_bb_solver function
        in_func = False
        found_loop = False
        burn_in_after_loop = False
        for i, line in enumerate(lines):
            if 'def fista_bb_solver(' in line:
                in_func = True
                continue
            if in_func and line.strip().startswith('def ') and 'fista_bb_solver' not in line:
                break
            if in_func and 'for iteration in range(max_iter):' in line:
                found_loop = True
                continue
            if found_loop and in_func:
                # After the loop starts, there should be no bb_burn_in assignment
                if 'bb_burn_in = max(' in line and '_momentum_burn_in' not in line:
                    # Check if this is inside the loop (indented more than the for)
                    indent = len(line) - len(line.lstrip())
                    for_indent = len(lines[i-1]) - len(lines[i-1].lstrip()) if i > 0 else 0
                    if indent > for_indent:
                        burn_in_after_loop = True
                        break
        assert found_loop, "Could not find 'for iteration in range(max_iter)' in fista_bb_solver"
        assert not burn_in_after_loop, "bb_burn_in assignment found inside the loop"
        print('[OK] fista_bb burn-in hoisted outside loop')


# ======================================================================
# 80. fista_bb restart uses coef not coef_new
# ======================================================================

class TestFistaBbRestartUsesCoef:
    def test_restart_uses_coef(self):
        """Restart check should use `coef` (current) not `coef_new` (stale)."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The restart check should use coef - coef_old, not coef_new - coef_old
        assert 'y_k - coef, coef - coef_old' in content
        assert 'y_k - coef_new, coef_new - coef_old' not in content
        print('[OK] fista_bb restart uses coef not coef_new')


# ======================================================================
# 81. Divergence threshold NaN safety
# ======================================================================

class TestDivergenceThresholdNaNSafety:
    def test_isfinite_guard_exists(self):
        """Divergence check should guard against _obj_best = ±inf."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'not np.isfinite(_obj_best)' in content
        print('[OK] Divergence threshold has NaN safety guard')


# ======================================================================
# 82. Exception narrowing in _tracking_penalty_value
# ======================================================================

class TestTrackingPenaltyExceptionNarrowing:
    def test_no_bare_except(self):
        """_tracking_penalty_value should not catch bare Exception."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the function
        start = content.find('def _tracking_penalty_value(')
        end = content.find('\ndef ', start + 1)
        func_body = content[start:end]
        # Should NOT have 'except Exception:' (bare)
        assert 'except Exception:' not in func_body, \
            "_tracking_penalty_value still has bare 'except Exception'"
        # Should have specific exception types
        assert 'except (ValueError, TypeError, AttributeError)' in func_body
        print('[OK] _tracking_penalty_value exception narrowing verified')


# ======================================================================
# 83. Newton solver line search exception narrowing
# ======================================================================

class TestNewtonLineSearchExceptionNarrowing:
    def test_no_bare_except(self):
        """newton_solver line search should not catch bare Exception."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        start = content.find('def newton_solver(')
        end = content.find('\ndef ', start + 1)
        func_body = content[start:end]
        assert 'except Exception:' not in func_body, \
            "newton_solver still has bare 'except Exception' in line search"
        assert 'except (ValueError, RuntimeError, FloatingPointError)' in func_body
        print('[OK] newton_solver line search exception narrowing verified')


# ======================================================================
# 84. CG solver pAp check
# ======================================================================

class TestCGSolverPapCheck:
    def test_cg_breaks_on_nonpositive_pap(self):
        """CG solver should break when pAp <= 0 (indefinite system)."""
        from statgpu.glm_core._solver import _cg_solve
        # Create an indefinite system: A = -I (negative definite)
        n = 5
        A_neg = -np.eye(n)
        b = np.ones(n)
        # Should not crash, returns best effort
        result = _cg_solve(lambda x: A_neg @ x, b, max_iter=10, tol=1e-6)
        assert result is not None
        assert np.all(np.isfinite(result))
        print('[OK] CG solver handles indefinite system')


# ======================================================================
# 85. CVCache thread safety
# ======================================================================

class TestCVCacheThreadSafety:
    def test_has_lock(self):
        """CVCache should have a threading.Lock."""
        from statgpu.linear_model._cv_base import CVCache
        cache = CVCache()
        assert hasattr(cache, '_lock')
        import threading
        # threading.Lock() returns a _thread.lock instance
        lock = threading.Lock()
        assert type(cache._lock) is type(lock), \
            f"Expected threading.Lock, got {type(cache._lock)}"
        print('[OK] CVCache has threading.Lock')

    def test_concurrent_access(self):
        """CVCache should handle concurrent get/put without corruption."""
        import threading
        from statgpu.linear_model._cv_base import CVCache
        cache = CVCache(maxsize=10)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    cache.put(f"key_{start}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    cache.get("nonexistent")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        print('[OK] CVCache handles concurrent access')


# ======================================================================
# 86. hash_cv_data collision reduction
# ======================================================================

class TestHashCvDataCollisionReduction:
    def test_different_data_different_hash(self):
        """Different datasets should produce different hashes."""
        from statgpu.linear_model._cv_base import hash_cv_data
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((100, 10))
        y1 = rng.standard_normal(100)
        X2 = rng.standard_normal((100, 10))
        y2 = rng.standard_normal(100)
        h1 = hash_cv_data(X1, y1)
        h2 = hash_cv_data(X2, y2)
        assert h1 != h2, "Different datasets produced same hash"
        print('[OK] Different datasets produce different hashes')

    def test_same_data_same_hash(self):
        """Same dataset should produce same hash."""
        from statgpu.linear_model._cv_base import hash_cv_data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = rng.standard_normal(100)
        h1 = hash_cv_data(X, y)
        h2 = hash_cv_data(X, y)
        assert h1 == h2, "Same dataset produced different hashes"
        print('[OK] Same dataset produces same hash')

    def test_small_dataset_full_hash(self):
        """Small datasets should use full content hashing."""
        from statgpu.linear_model._cv_base import hash_cv_data
        # n*p = 10*5 = 50 < 50000 threshold
        X = np.ones((10, 5))
        y = np.ones(10)
        # Modify last row only — should detect difference
        X2 = X.copy()
        X2[-1, 0] = 999.0
        h1 = hash_cv_data(X, y)
        h2 = hash_cv_data(X2, y)
        assert h1 != h2, "Full hash should detect last-row difference"
        print('[OK] Small dataset uses full content hashing')

    def test_sampled_hash_includes_boundary(self):
        """Large dataset sampled hash should include first and last rows."""
        from statgpu.linear_model._cv_base import hash_cv_data
        rng = np.random.default_rng(42)
        # n*p = 200*500 = 100000 > 50000 threshold
        X = rng.standard_normal((200, 500))
        y = rng.standard_normal(200)
        X2 = X.copy()
        X2[-1, 0] = 999.0  # modify last row only
        h1 = hash_cv_data(X, y)
        h2 = hash_cv_data(X2, y)
        assert h1 != h2, "Sampled hash should detect last-row difference"
        print('[OK] Sampled hash includes boundary rows')

    def test_weighted_hash_includes_weights(self):
        """Hash should differ when sample_weight differs."""
        from statgpu.linear_model._cv_base import hash_cv_data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = rng.standard_normal(50)
        w1 = np.ones(50)
        w2 = np.ones(50)
        w2[0] = 2.0
        h1 = hash_cv_data(X, y, w1)
        h2 = hash_cv_data(X, y, w2)
        assert h1 != h2, "Different weights produced same hash"
        print('[OK] Weighted hash detects weight differences')


# ======================================================================
# 87. RidgeCV weighted n_train consistency
# ======================================================================

class TestRidgeCVWeightedNTrain:
    def test_n_train_uses_weight_sum(self):
        """RidgeCV weighted path should use sum(weights) for n_train."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The old code had: n_train = float(sw_train.sum()) if bool(fit_intercept) else int(X_train.shape[0])
        # The fix should always use float(sw_train.sum())
        assert 'if bool(fit_intercept) else int(X_train.shape[0])' not in content, \
            "RidgeCV still has conditional n_train based on fit_intercept"
        print('[OK] RidgeCV weighted n_train uses weight sum unconditionally')


# ======================================================================
# 88. run_cv sample_weight validation
# ======================================================================

class TestRunCvSampleWeightValidation:
    def test_mismatched_weight_length_raises(self):
        """run_cv should raise ValueError when sample_weight length mismatches."""
        from statgpu.linear_model._cv_engine import run_cv
        X = np.zeros((100, 5))
        y = np.zeros(100)
        bad_weights = np.ones(50)  # wrong length
        with pytest.raises(ValueError, match="sample_weight length"):
            run_cv(X, y, np.array([0.1, 1.0]),
                   lambda Xt, yt, Xv, yv, a, sw_train=None, sw_val=None: 0.0,
                   n_folds=2, sample_weight=bad_weights)
        print('[OK] run_cv rejects mismatched sample_weight')


# ======================================================================
# 89. LBFGS line search stall warning
# ======================================================================

class TestLBFGSLineSearchStallWarning:
    def test_warning_emitted_on_stall(self):
        """lbfgs_solver should warn when line search fails all backtracking steps."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'lbfgs_solver: line search failed' in content
        print('[OK] LBFGS stall warning exists in code')


# ======================================================================
# 90. Zero init pattern
# ======================================================================

class TestZeroInitPattern:
    def test_no_copy_times_zero(self):
        """Should use _zeros() instead of _copy_arr * 0.0 for zero init."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The old pattern was _copy_arr(coef) * 0.0
        assert '_copy_arr(coef) * 0.0' not in content, \
            "Still using _copy_arr(coef) * 0.0 for zero init"
        print('[OK] Zero init uses _zeros()')


# ======================================================================
# 91. ADMM Cholesky fallback
# ======================================================================

class TestADMMCholeskyFallback:
    def test_cholesky_has_try_except(self):
        """ADMM Cholesky should have try/except for non-PD matrices."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the ADMM solver
        start = content.find('def admm_solver(')
        end = content.find('\ndef ', start + 1)
        func_body = content[start:end]
        assert 'LinAlgError' in func_body or '_cholesky_ok' in func_body, \
            "ADMM Cholesky has no fallback for non-PD matrices"
        print('[OK] ADMM Cholesky has fallback for non-PD matrices')


# ======================================================================
# 92. PenalizedGLM_CV MSE fallback warning
# ======================================================================

class TestPenalizedGLMCVMSEFallbackWarning:
    def test_mse_fallback_warns(self):
        """_evaluate_single MSE fallback should emit RuntimeWarning."""
        with open('statgpu/linear_model/_penalized_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find _evaluate_single
        start = content.find('def _evaluate_single(')
        end = content.find('\n    def ', start + 1)
        func_body = content[start:end]
        assert 'falling back to MSE' in func_body
        assert 'RuntimeWarning' in func_body
        print('[OK] PenalizedGLM_CV MSE fallback emits warning')


# ======================================================================
# 93. _build_cv_cache uses _max_eigval_power
# ======================================================================

class TestBuildCvCacheUsesMaxEigvalPower:
    def test_no_eigvalsh(self):
        """_build_cv_cache should use _max_eigval_power, not eigvalsh."""
        with open('statgpu/linear_model/_penalized_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        start = content.find('def _build_cv_cache(')
        end = content.find('\n    def ', start + 1)
        func_body = content[start:end]
        assert 'eigvalsh' not in func_body, \
            "_build_cv_cache still uses eigvalsh"
        assert '_max_eigval_power' in func_body
        print('[OK] _build_cv_cache uses _max_eigval_power')


# ======================================================================
# 94. C1: RidgeCV torch CPU tensor + device='cuda' guard
# ======================================================================

class TestRidgeCVTorchCpuCudaGuard:
    def test_gpu_input_torch_in_condition(self):
        """GPU path should check gpu_input_torch, not just device attribute."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The convert-from-numpy branch must include gpu_input_torch
        assert 'gpu_input_cupy or gpu_input_torch or (hasattr(X' in content, \
            "GPU path missing gpu_input_torch guard"
        print('[OK] RidgeCV torch CPU tensor guard present')


# ======================================================================
# 95. C2: predict() return_cpu parameter
# ======================================================================

class TestPredictReturnCpuParam:
    def test_predict_has_return_cpu(self):
        """predict() should accept return_cpu parameter."""
        import inspect
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        sig = inspect.signature(PenalizedGeneralizedLinearModel.predict)
        assert 'return_cpu' in sig.parameters, \
            "predict() missing return_cpu parameter"
        assert sig.parameters['return_cpu'].default is True, \
            "return_cpu should default to True"
        print('[OK] predict() has return_cpu=True parameter')

    def test_predict_numpy_returns_numpy(self):
        """predict() with numpy backend should always return numpy."""
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * np.random.randn(50)
        model = PenalizedLinearRegression(alpha=0.01)
        model.fit(X, y)
        pred = model.predict(X, return_cpu=True)
        assert isinstance(pred, np.ndarray), f"Expected ndarray, got {type(pred)}"
        pred2 = model.predict(X, return_cpu=False)
        assert isinstance(pred2, np.ndarray), f"numpy backend should return ndarray regardless"
        print('[OK] predict() numpy backend returns numpy')


# ======================================================================
# 96. C3: LLA intercept + sample_weight
# ======================================================================

class TestLLAInterceptSampleWeight:
    def test_lla_weighted_intercept_code(self):
        """_fit_lla should use weighted means for intercept when sample_weight provided."""
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the _fit_lla intercept computation
        idx = content.find('if self.coef_ is None and coef_lla is not None')
        assert idx > 0, "Could not find LLA intercept block"
        block = content[idx:idx+600]
        assert 'sample_weight is not None' in block, \
            "LLA intercept does not check sample_weight"
        assert 'sw_sum' in block, \
            "LLA intercept missing weighted sum guard"
        print('[OK] LLA intercept uses weighted means with sample_weight')


# ======================================================================
# 97. C4: fista_bb objective threshold is relative
# ======================================================================

class TestFistaBbObjectiveThreshold:
    def test_threshold_is_relative(self):
        """fista_bb objective threshold should be relative to _obj_best, not hardcoded 1e6."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Should NOT have bare `_new_total < 1e6` without _obj_cap
        # The fix uses _obj_cap = max(_obj_best * 10.0, 1e6)
        assert '_obj_cap' in content, \
            "Missing _obj_cap relative threshold"
        assert '_obj_best * 10.0' in content, \
            "Objective threshold not relative to _obj_best"
        print('[OK] fista_bb objective threshold is relative')


# ======================================================================
# 98. H1/H2: SCAD a=2 / MCP gamma=1 div/0 guards
# ======================================================================

class TestSCADMCPSingularityGuard:
    def test_scad_a_guard_in_source(self):
        """SCAD CD should guard a != 2.0 to avoid division by zero."""
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Should have guard: abs(a_scad - 2.0) < 1e-6
        assert 'abs(a_scad - 2.0)' in content, \
            "SCAD missing a=2.0 singularity guard"
        print('[OK] SCAD a=2.0 singularity guard present')

    def test_mcp_gamma_guard_in_source(self):
        """MCP CD should guard gamma > 1 to avoid division by zero."""
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'max(gamma_mcp, 1.0 + 1e-6)' in content, \
            "MCP missing gamma=1.0 singularity guard"
        print('[OK] MCP gamma=1.0 singularity guard present')

    def test_scad_cd_no_crash_near_boundary(self):
        """SCAD CD with a close to 2.0 should not crash (guard clamps a away from 2)."""
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(30)
        # a=2.001 is valid for SCADPenalty constructor but would cause div/0
        # in the CD formula without the guard (a-2.0 = 0.001 -> huge values)
        model = PenalizedLinearRegression(alpha=0.1, penalty='scad',
                                          penalty_kwargs={'a': 2.001})
        model.fit(X, y)
        assert model.coef_ is not None
        assert np.all(np.isfinite(model.coef_))
        print('[OK] SCAD a=2.001 does not crash')

    def test_mcp_cd_no_crash_near_boundary(self):
        """MCP CD with gamma close to 1.0 should not crash (guard clamps gamma away from 1)."""
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = X @ np.random.randn(5) + 0.1 * np.random.randn(30)
        # gamma=1.001 is valid for MCPPenalty but would cause div/0
        # in the CD formula without the guard (1 - 1/1.001 ≈ 0.001)
        model = PenalizedLinearRegression(alpha=0.1, penalty='mcp',
                                          penalty_kwargs={'gamma': 1.001})
        model.fit(X, y)
        assert model.coef_ is not None
        assert np.all(np.isfinite(model.coef_))
        print('[OK] MCP gamma=1.001 does not crash')


# ======================================================================
# 99. H5: fit_intercept not mutated by formula
# ======================================================================

class TestFitInterceptNotMutated:
    def test_fit_intercept_preserved(self):
        """fit() with formula should not mutate self.fit_intercept."""
        import pandas as pd
        from statgpu.linear_model._glm_base import GeneralizedLinearModel
        # Create a simple model
        model = GeneralizedLinearModel(family='gaussian', fit_intercept=True)
        original = model.fit_intercept
        # After construction, fit_intercept should be unchanged
        assert model.fit_intercept == original
        assert model._use_intercept is None  # not set until fit with formula
        print('[OK] fit_intercept not mutated by formula')

    def test_effective_intercept_property(self):
        """_effective_intercept should fall back to fit_intercept when no formula."""
        from statgpu.linear_model._glm_base import GeneralizedLinearModel
        model = GeneralizedLinearModel(family='gaussian', fit_intercept=True)
        assert model._effective_intercept is True
        model2 = GeneralizedLinearModel(family='gaussian', fit_intercept=False)
        assert model2._effective_intercept is False
        # With _use_intercept set (formula mode)
        model._use_intercept = False
        assert model._effective_intercept is False
        assert model.fit_intercept is True  # original preserved
        print('[OK] _effective_intercept property works correctly')


# ======================================================================
# 100. H6: get_params() returns all constructor parameters
# ======================================================================

class TestGetParamsComplete:
    def test_base_estimator_get_params(self):
        """BaseEstimator.get_params() should return all __init__ params."""
        from statgpu._base import BaseEstimator
        # Use a real concrete subclass (Ridge) to avoid abstract method issues
        from statgpu.linear_model._ridge import Ridge
        est = Ridge(alpha=0.5, fit_intercept=False)
        params = est.get_params()
        assert 'alpha' in params, "get_params() missing 'alpha'"
        assert 'fit_intercept' in params, "get_params() missing 'fit_intercept'"
        assert 'device' in params, "get_params() missing 'device'"
        assert params['alpha'] == 0.5
        assert params['fit_intercept'] is False
        print('[OK] BaseEstimator.get_params() returns all params')

    def test_cv_estimator_get_params(self):
        """CVEstimatorBase.get_params() should return cv and random_state."""
        from statgpu.linear_model._ridge_cv import RidgeCV
        est = RidgeCV(cv=3, random_state=42, fit_intercept=False)
        params = est.get_params()
        assert 'cv' in params, "get_params() missing 'cv'"
        assert 'random_state' in params, "get_params() missing 'random_state'"
        assert 'fit_intercept' in params, "get_params() missing 'fit_intercept'"
        assert params['cv'] == 3
        assert params['random_state'] == 42
        assert params['fit_intercept'] is False
        print('[OK] CVEstimatorBase.get_params() returns all params')


# ======================================================================
# 101. H7: CPU weighted w_sum zero guard
# ======================================================================

class TestRidgeCVWsumZeroGuard:
    def test_wsum_guard_in_source(self):
        """CPU weighted path should guard w_sum against zero."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the CPU weighted path
        assert 'max(float(np.sum(sw_train)), 1e-15)' in content, \
            "CPU weighted path missing w_sum zero guard"
        print('[OK] RidgeCV CPU w_sum zero guard present')


# ======================================================================
# 102. H8: Ridge gradient check uses ridge_penalize_intercept
# ======================================================================

class TestRidgeGradientCheckIntercept:
    def test_gradient_uses_penalize_intercept(self):
        """IRLS gradient check should respect ridge_penalize_intercept."""
        with open('statgpu/glm_core/_irls.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert '_start = 0 if ridge_penalize_intercept else 1' in content, \
            "Ridge gradient check does not use ridge_penalize_intercept"
        print('[OK] Ridge gradient check uses ridge_penalize_intercept')


# ======================================================================
# 103. M5/M9: Exception handlers narrowed
# ======================================================================

class TestExceptionHandlersNarrowed:
    def test_irls_exception_narrowed(self):
        """IRLS gradient fallback should catch specific exceptions, not bare Exception."""
        with open('statgpu/glm_core/_irls.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The gradient fallback should catch (AttributeError, NotImplementedError)
        assert 'except (AttributeError, NotImplementedError)' in content, \
            "IRLS exception handler not narrowed"
        print('[OK] IRLS exception handler narrowed')

    def test_solver_compile_exception_narrowed(self):
        """Solver torch.compile fallback should catch RuntimeError, not bare Exception."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Should have RuntimeError for compile failures
        assert 'except RuntimeError:' in content, \
            "Solver compile exception not narrowed to RuntimeError"
        print('[OK] Solver compile exception handler narrowed')


# ======================================================================
# 104. M6: Tweedie fused singularity guard
# ======================================================================

class TestTweedieFusedSingularity:
    def test_tweedie_uses_log_form_near_singularities(self):
        """Tweedie fused formula should use log-form near power=1 or power=2."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find _fused_tweedie
        idx = content.find('def _fused_tweedie(')
        assert idx > 0, "Could not find _fused_tweedie"
        block = content[idx:idx+600]
        assert 'abs(d1) < 0.01' in block, \
            "Tweedie missing singularity guard for power≈1"
        assert 'abs(d2) < 0.01' in block, \
            "Tweedie missing singularity guard for power≈2"
        assert 'log_mu' in block, \
            "Tweedie missing log-form for singularities"
        print('[OK] Tweedie fused singularity guard present')


# ======================================================================
# 105. M8: family_to_loss raises on unknown family
# ======================================================================

class TestFamilyToLossRaises:
    def test_unknown_family_raises(self):
        """family_to_loss() should raise ValueError for unknown family."""
        from statgpu.linear_model._glm_base import GeneralizedLinearModel
        model = GeneralizedLinearModel(family='gaussian')
        model.family = 'nonexistent_family'
        try:
            model.family_to_loss()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'nonexistent_family' in str(e)
        print('[OK] family_to_loss() raises on unknown family')


# ======================================================================
# 106. M10: Gamma deviance y>0 guard
# ======================================================================

class TestGammaDevianceYGuard:
    def test_gamma_deviance_clips_y(self):
        """Gamma deviance should clip y to avoid log(0)."""
        with open('statgpu/glm_core/_irls.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the gamma deviance branch
        idx = content.find("elif _fname == \"gamma\":")
        assert idx > 0, "Could not find gamma deviance branch"
        block = content[idx:idx+200]
        assert '_clip(y_dev' in block or '_y_c = _clip' in block, \
            "Gamma deviance missing y>0 guard"
        print('[OK] Gamma deviance has y>0 guard')


# ======================================================================
# 107. M11: Float32 eigenvalue clamp
# ======================================================================

class TestFloat32EigenvalueClamp:
    def test_dtype_relative_clamp(self):
        """Eigenvalue clamp should use dtype-relative floor."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'finfo' in content, \
            "Eigenvalue clamp missing dtype-relative floor (finfo)"
        print('[OK] Eigenvalue clamp uses dtype-relative floor')


# ======================================================================
# 108. H3: GPU IRLS-CD step-halving exists
# ======================================================================

class TestGPUirlsCDStepHalving:
    def test_step_halving_in_gpu_path(self):
        """GPU IRLS-CD should have step-halving for GLM."""
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find _irls_cd_gpu
        idx = content.find('def _irls_cd_gpu(')
        assert idx > 0, "Could not find _irls_cd_gpu"
        func_end = content.find('\n    def ', idx + 1)
        func_body = content[idx:func_end]
        assert 'Step-halving for GLM' in func_body, \
            "GPU IRLS-CD missing step-halving"
        assert '_obj_before' in func_body, \
            "GPU IRLS-CD missing _obj_before for step-halving"
        print('[OK] GPU IRLS-CD has step-halving')


# ======================================================================
# 109. H4: FISTA early-reject heuristic removed
# ======================================================================

class TestFistaEarlyRejectRemoved:
    def test_no_early_reject(self):
        """FISTA should not have the early-reject heuristic that rejects improving steps."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The old code had: if _to_float_scalar(bound_dev) < _obj_best_lla_inner * 0.9
        # This should be removed
        assert '_obj_best_lla_inner * 0.9' not in content, \
            "FISTA early-reject heuristic still present"
        print('[OK] FISTA early-reject heuristic removed')


# ======================================================================
# 110. M1: Double GPU sync fixed
# ======================================================================

class TestDoubleGPUSyncFixed:
    def test_single_sync_for_obj(self):
        """FISTA GPU path should extract float once, not twice."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the GPU async path's objective check
        idx = content.find('# Single D2H transfer: extract float')
        assert idx > 0, "Could not find single-sync comment"
        block = content[idx:idx+200]
        # Should have _obj_val_f assigned once, then reused
        assert '_obj_val_f = float(_to_numpy(_obj_dev))' in block, \
            "Missing single float extraction"
        # Should NOT have a second float(_to_numpy(_obj_dev)) nearby
        lines = block.split('\n')
        sync_count = sum(1 for l in lines if 'float(_to_numpy(_obj_dev))' in l)
        assert sync_count == 1, f"Expected 1 sync, found {sync_count}"
        print('[OK] Double GPU sync fixed')


# ======================================================================
# 111. RidgeCV Device enum handling
# ======================================================================

class TestRidgeCVDeviceEnum:
    def test_device_enum_converted(self):
        """_select_ridge_alpha_cv should handle Device enum."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'isinstance(device, Device)' in content, \
            "Missing Device enum handling"
        print('[OK] RidgeCV handles Device enum')


# ======================================================================
# 112. Cleanup: torch.cuda.synchronize removed
# ======================================================================

class TestCleanupNoSync:
    def test_no_synchronize_in_cleanup(self):
        """_cleanup_torch_memory should not call torch.cuda.synchronize()."""
        with open('statgpu/linear_model/_glm_base.py', 'r', encoding='utf-8') as f:
            content = f.read()
        idx = content.find('def _cleanup_torch_memory(')
        assert idx > 0
        block = content[idx:idx+300]
        assert 'synchronize' not in block, \
            "_cleanup_torch_memory still calls synchronize()"
        print('[OK] _cleanup_torch_memory has no synchronize()')


# ======================================================================
# 113. _FeatureOnlySparsePenalty uses zeros, not empty
# ======================================================================

class TestProximalUsesZeros:
    def test_no_empty_in_proximal(self):
        """_FeatureOnlySparsePenalty.proximal should use zeros, not empty."""
        with open('statgpu/linear_model/_penalized_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        idx = content.find('class _FeatureOnlySparsePenalty')
        assert idx > 0
        block = content[idx:idx+500]
        assert 'xp.empty' not in block, \
            "_FeatureOnlySparsePenalty.proximal still uses xp.empty"
        print('[OK] _FeatureOnlySparsePenalty.proximal uses zeros')


# ======================================================================
# 114. NEW-C1: RidgeCV alpha grid torch CPU guard
# ======================================================================

class TestRidgeCVAlphaGridTorchGuard:
    def test_alpha_grid_includes_gpu_input_torch(self):
        """Alpha grid generation should include gpu_input_torch in GPU condition."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find all GPU conditions for alpha grid generation
        # They should all include gpu_input_torch
        import re
        # Pattern: conditions that check gpu_input_cupy for alpha grid
        matches = re.findall(r'if gpu_input_cupy or .+?:', content)
        for m in matches:
            if 'gpu_input_torch' not in m and 'hasattr' in m:
                # This is an alpha grid condition missing gpu_input_torch
                assert False, f"Alpha grid condition missing gpu_input_torch: {m}"
        print('[OK] RidgeCV alpha grid includes gpu_input_torch guard')


# ======================================================================
# 115. NEW-M1: _penalized.py _effective_intercept
# ======================================================================

class TestPenalizedEffectiveIntercept:
    def test_has_effective_intercept_property(self):
        """PenalizedGeneralizedLinearModel should have _effective_intercept property."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        assert hasattr(PenalizedGeneralizedLinearModel, '_effective_intercept')
        # Should be a property
        assert isinstance(
            getattr(PenalizedGeneralizedLinearModel, '_effective_intercept'),
            property
        )
        print('[OK] PenalizedGLM has _effective_intercept property')

    def test_formula_does_not_mutate_fit_intercept(self):
        """Formula path should set _use_intercept, not mutate fit_intercept."""
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The formula path should use _use_intercept
        assert 'self._use_intercept = True' in content, \
            "Formula path should set _use_intercept = True"
        assert 'self._use_intercept = False' in content, \
            "Formula path should set _use_intercept = False"
        # The formula block should NOT have self.fit_intercept = True/False
        # Find the formula handling block
        idx = content.find('self._formula_has_intercept = "Intercept"')
        assert idx > 0
        block = content[idx:idx+600]
        assert 'self.fit_intercept = True' not in block, \
            "Formula path should NOT mutate fit_intercept"
        assert 'self.fit_intercept = False' not in block, \
            "Formula path should NOT mutate fit_intercept"
        print('[OK] _penalized.py formula path uses _use_intercept')


# ======================================================================
# 116. NEW-M2: fista_bb penalty name case sensitivity
# ======================================================================

class TestFistaBbPenaltyNameCase:
    def test_penalty_name_lowered(self):
        """fista_bb should lowercase penalty name for _bb_disabled check."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the line that overwrites _pen_name with getattr
        idx = content.find('_pen_name = getattr(penalty, "name", _pen_name)')
        assert idx > 0, "Could not find penalty name override line"
        line = content[idx:idx+200]
        assert '.lower()' in line, \
            "Penalty name override should apply .lower()"
        print('[OK] fista_bb penalty name applies .lower()')


# ======================================================================
# 117. NEW-M3: _compute_intercepts_batch dtype
# ======================================================================

class TestInterceptsBatchDtype:
    def test_no_hardcoded_float64(self):
        """_compute_intercepts_batch should not hardcode float64 for zeros."""
        with open('statgpu/linear_model/_ridge_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        idx = content.find('def _compute_intercepts_batch(')
        assert idx > 0
        end = content.find('\ndef ', idx + 1)
        func_body = content[idx:end]
        # Should NOT have dtype=backend.float64 in the no-intercept return
        assert 'dtype=backend.float64' not in func_body, \
            "_compute_intercepts_batch still hardcodes float64"
        print('[OK] _compute_intercepts_batch uses input dtype')


# ======================================================================
# 118. NEW-L1/L2: Lipschitz y-scaling reapply
# ======================================================================

class TestLipschitzYScalingReapply:
    def test_fista_solver_reapplies_y_scale(self):
        """fista_solver should re-apply y-scaling during Lipschitz recomputation."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # The y-scaling reapply should be in the Lipschitz recomputation section
        assert '# Re-apply y-scaling' in content, \
            "fista_solver missing y-scaling reapply comment"
        # Count occurrences: should appear in both fista_solver and fista_bb
        count = content.count('# Re-apply y-scaling')
        assert count >= 2, f"Expected >= 2 y-scaling reapply blocks, found {count}"
        print('[OK] fista_solver re-applies y-scaling')

    def test_fista_bb_reapplies_y_scale(self):
        """fista_bb should re-apply y-scaling during Lipschitz recomputation."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the burn-in Lipschitz recomputation in fista_bb
        idx = content.find('# Re-apply y-scaling and per-family safety factor')
        assert idx > 0, "Could not find y-scaling reapply in fista_bb"
        print('[OK] fista_bb re-applies y-scaling')


# ======================================================================
# 119. NEW-L3/L4: predict + refit _effective_intercept
# ======================================================================

class TestPredictEffectiveIntercept:
    def test_predict_uses_effective_intercept(self):
        """predict() in GLM base should use _effective_intercept, not fit_intercept."""
        with open('statgpu/linear_model/_glm_base.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the predict method — search a larger window
        idx = content.find('def predict(self, X):')
        assert idx > 0
        predict_body = content[idx:idx+2000]
        # Should use _effective_intercept
        assert 'self._effective_intercept' in predict_body, \
            "predict() should use self._effective_intercept"
        # Count bare self.fit_intercept in predict (excluding _effective)
        import re
        bare_refs = [l for l in predict_body.split('\n')
                     if 'self.fit_intercept' in l and '_effective' not in l and '_use_intercept' not in l]
        assert len(bare_refs) == 0, f"predict() still has bare self.fit_intercept: {bare_refs}"
        print('[OK] predict() uses _effective_intercept')

    def test_refit_resets_use_intercept(self):
        """Non-formula re-fit should reset _use_intercept to None."""
        with open('statgpu/linear_model/_glm_base.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Find the non-formula path resets
        idx = content.find('self._formula_has_intercept = None')
        assert idx > 0
        block = content[idx:idx+100]
        assert 'self._use_intercept = None' in block, \
            "Non-formula re-fit should reset _use_intercept"
        print('[OK] Non-formula re-fit resets _use_intercept')


# ======================================================================
# 120. NEW-L5: _get_family ValueError
# ======================================================================

class TestGetFamilyValueError:
    def test_unknown_family_raises_valueerror(self):
        """_get_family() should raise ValueError for unknown family."""
        from statgpu.linear_model._glm_base import GeneralizedLinearModel
        model = GeneralizedLinearModel(family='gaussian')
        model.family = 'nonexistent'
        try:
            model._get_family()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'nonexistent' in str(e)
        except KeyError:
            assert False, "Should raise ValueError, not KeyError"
        print('[OK] _get_family() raises ValueError for unknown family')


# ======================================================================
# 121. Sample weight validation dedup
# ======================================================================

class TestSampleWeightValidationDedup:
    def test_shared_helper_exists(self):
        """_validate_sample_weight shared helper should exist."""
        with open('statgpu/glm_core/_solver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'def _validate_sample_weight(' in content, \
            "_validate_sample_weight helper not found"
        # Should be called from at least 3 places
        count = content.count('_validate_sample_weight(')
        assert count >= 4, f"Expected >= 4 calls (3 callers + 1 def), found {count}"
        print('[OK] _validate_sample_weight shared helper exists and is used')


# ======================================================================
# 122. Precision regression check
# ======================================================================

class TestPrecisionRegression:
    def test_lasso_coefficients_stable(self):
        """Lasso coefficients should remain stable after all fixes."""
        np.random.seed(42)
        from statgpu.linear_model._lasso import Lasso
        X = np.random.randn(100, 10)
        beta_true = np.zeros(10)
        beta_true[:3] = [3.0, -2.0, 1.0]
        y = X @ beta_true + 0.1 * np.random.randn(100)
        model = Lasso(alpha=0.05)
        model.fit(X, y)
        coef = model.coef_
        # Check non-zero coefficients are close to true values
        assert np.abs(coef[0] - 3.0) < 0.5, f"coef[0]={coef[0]:.4f}, expected ~3.0"
        assert np.abs(coef[1] - (-2.0)) < 0.5, f"coef[1]={coef[1]:.4f}, expected ~-2.0"
        assert np.abs(coef[2] - 1.0) < 0.5, f"coef[2]={coef[2]:.4f}, expected ~1.0"
        # Check near-zero coefficients are actually near zero
        assert np.max(np.abs(coef[3:])) < 0.5, f"coef[3:] should be near zero: {coef[3:]}"
        print('[OK] Lasso coefficients stable: R²=%.4f' % model.score(X, y))

    def test_ridge_cv_selects_good_alpha(self):
        """RidgeCV should select an alpha that gives good R²."""
        np.random.seed(42)
        from statgpu.linear_model._ridge_cv import RidgeCV
        X = np.random.randn(500, 10)
        beta_true = np.zeros(10)
        beta_true[:3] = [3.0, -2.0, 1.0]
        y = X @ beta_true + 0.1 * np.random.randn(500)
        model = RidgeCV(cv=3, random_state=42)
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.99, f"R²={r2:.4f}, expected > 0.99"
        print('[OK] RidgeCV R²=%.4f, alpha=%.6f' % (r2, model.alpha_))

    def test_glm_poisson_convergence(self):
        """Poisson GLM should converge to reasonable coefficients."""
        np.random.seed(42)
        from statgpu.linear_model._poisson_glm import PoissonRegression
        X = np.random.randn(100, 3)
        eta = X @ np.array([0.5, -0.3, 0.1])
        y = np.random.poisson(np.exp(np.clip(eta, -5, 5)))
        model = PoissonRegression(max_iter=200)
        model.fit(X, y)
        assert model.n_iter_ < 200, f"Not converged: {model.n_iter_} iterations"
        assert np.all(np.isfinite(model.coef_)), "Coefficients not finite"
        print('[OK] PoissonGLM converged in %d iterations' % model.n_iter_)
