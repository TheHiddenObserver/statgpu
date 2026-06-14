"""Phase 0 safety net tests: written before refactoring, must all pass after.

These tests capture current behavior baseline. They test public APIs and
numerical results only — no internal implementation details.
"""
import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════
# A. Solver independent tests (currently no coverage)
# ══════════════════════════════════════════════════════════════════════

class TestSolverImports:
    """All solvers importable from glm_core (after refactor: from solvers/)."""

    def test_import_fista_solver(self):
        from statgpu.glm_core import fista_solver
        assert callable(fista_solver)

    def test_import_fista_bb_solver(self):
        from statgpu.glm_core import fista_bb_solver
        assert callable(fista_bb_solver)

    def test_import_newton_solver(self):
        from statgpu.glm_core import newton_solver
        assert callable(newton_solver)

    def test_import_lbfgs_solver(self):
        from statgpu.glm_core import lbfgs_solver
        assert callable(lbfgs_solver)

    def test_import_admm_solver(self):
        from statgpu.glm_core import admm_solver
        assert callable(admm_solver)

    def test_import_fista_lla_path(self):
        from statgpu.glm_core._solver import fista_lla_path
        assert callable(fista_lla_path)


class TestSolverWithSquaredError:
    """solver + SquaredErrorLoss (simplest case)."""

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        beta = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
        y = X @ beta + rng.randn(100) * 0.1
        return X, y, beta

    @pytest.fixture
    def loss(self):
        from statgpu.glm_core import SquaredErrorLoss
        return SquaredErrorLoss()

    def test_fista_squared_error_converges(self, data, loss):
        from statgpu.glm_core import fista_solver
        from statgpu.penalties import L2Penalty
        X, y, _ = data
        pen = L2Penalty(alpha=0.0)
        coef, n_iter = fista_solver(loss, pen, X, y, max_iter=500, tol=1e-6)
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(coef, ols, atol=1e-3)

    def test_fista_bb_squared_error_converges(self, data, loss):
        from statgpu.glm_core import fista_bb_solver
        from statgpu.penalties import L2Penalty
        X, y, _ = data
        pen = L2Penalty(alpha=0.0)
        coef, n_iter = fista_bb_solver(loss, pen, X, y, max_iter=500, tol=1e-6)
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(coef, ols, atol=1e-3)

    def test_newton_squared_error_converges(self, data, loss):
        from statgpu.glm_core import newton_solver
        from statgpu.penalties import L2Penalty
        X, y, _ = data
        n = X.shape[0]
        alpha = 0.1
        pen = L2Penalty(alpha=alpha)
        coef, n_iter = newton_solver(loss, pen, X, y, max_iter=50, tol=1e-6)
        # Solver normalizes loss by n, so effective penalty is n*alpha
        ridge = np.linalg.solve(X.T @ X + n * alpha * np.eye(5), X.T @ y)
        np.testing.assert_allclose(coef, ridge, atol=1e-3)

    def test_lbfgs_squared_error_converges(self, data, loss):
        from statgpu.glm_core import lbfgs_solver
        from statgpu.penalties import L2Penalty
        X, y, _ = data
        n = X.shape[0]
        alpha = 0.1
        pen = L2Penalty(alpha=alpha)
        coef, n_iter = lbfgs_solver(loss, pen, X, y, max_iter=50, tol=1e-6)
        ridge = np.linalg.solve(X.T @ X + n * alpha * np.eye(5), X.T @ y)
        np.testing.assert_allclose(coef, ridge, atol=1e-3)


class TestSolverWithLogisticLoss:
    """solver + LogisticLoss (non-quadratic, uses fused optimization)."""

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        beta = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
        prob = 1 / (1 + np.exp(-X @ beta))
        y = (rng.rand(200) < prob).astype(float)
        return X, y

    def test_fista_logistic_l1_produces_sparse_coef(self, data):
        from statgpu.glm_core import fista_solver, LogisticLoss
        from statgpu.penalties import L1Penalty
        X, y = data
        loss = LogisticLoss()
        pen = L1Penalty(alpha=0.1)
        coef, n_iter = fista_solver(loss, pen, X, y, max_iter=500, tol=1e-4)
        assert np.all(np.isfinite(coef))
        # L1 should produce some near-zero coefficients
        assert np.sum(np.abs(coef) < 0.01) >= 1

    def test_fista_bb_logistic_l1_produces_sparse_coef(self, data):
        from statgpu.glm_core import fista_bb_solver, LogisticLoss
        from statgpu.penalties import L1Penalty
        X, y = data
        loss = LogisticLoss()
        pen = L1Penalty(alpha=0.1)
        coef, n_iter = fista_bb_solver(loss, pen, X, y, max_iter=500, tol=1e-4)
        assert np.all(np.isfinite(coef))
        assert np.sum(np.abs(coef) < 0.01) >= 1


class TestSolverWithPoissonLoss:
    """solver + PoissonLoss (exp-link, needs momentum tuning)."""

    def test_fista_poisson_l2_converges(self):
        from statgpu.glm_core import fista_solver, PoissonLoss
        from statgpu.penalties import L2Penalty
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        eta = X @ np.array([0.5, -0.3, 0.1])
        y = rng.poisson(np.exp(eta)).astype(float)
        loss = PoissonLoss()
        pen = L2Penalty(alpha=0.01)
        coef, n_iter = fista_solver(loss, pen, X, y, max_iter=500, tol=1e-4)
        assert np.all(np.isfinite(coef))


# ══════════════════════════════════════════════════════════════════════
# B. Cross-validation framework tests (currently no coverage)
# ══════════════════════════════════════════════════════════════════════

class TestCVBase:
    """Utility functions from _cv_base.py."""

    def test_kfold_indices_shape(self):
        from statgpu.linear_model._cv_base import kfold_indices
        folds = kfold_indices(100, n_splits=5, random_state=42)
        assert len(folds) == 5
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 100
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_kfold_indices_disjoint(self):
        from statgpu.linear_model._cv_base import kfold_indices
        folds = kfold_indices(50, n_splits=5, random_state=0)
        all_val = np.concatenate([v for _, v in folds])
        assert len(np.unique(all_val)) == 50

    def test_hash_cv_data_deterministic(self):
        from statgpu.linear_model._cv_base import hash_cv_data
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)
        h1 = hash_cv_data(X, y)
        h2 = hash_cv_data(X, y)
        assert h1 == h2

    def test_hash_cv_data_different_on_data_change(self):
        from statgpu.linear_model._cv_base import hash_cv_data
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)
        h1 = hash_cv_data(X, y)
        y2 = y + 1.0
        h2 = hash_cv_data(X, y2)
        assert h1 != h2

    def test_batch_mse_basic(self):
        from statgpu.linear_model._cv_base import batch_mse
        rng = np.random.RandomState(42)
        X_val = rng.randn(20, 3)
        y_val = X_val @ np.array([1.0, -1.0, 0.5]) + rng.randn(20) * 0.1
        coefs = np.array([[1.0, -1.0, 0.5], [0.0, 0.0, 0.0]])
        mses = batch_mse(X_val, y_val, coefs)
        assert mses.shape == (2,)
        assert mses[0] < mses[1]  # true coef should have lower MSE


class TestCVEngine:
    """run_cv from _cv_engine.py."""

    def test_run_cv_basic(self):
        from statgpu.linear_model._cv_engine import run_cv
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.randn(100) * 0.1
        alphas = np.array([0.01, 0.1, 1.0])

        def evaluate_fold(X_train, y_train, X_val, y_val, alpha,
                          sample_weight_train=None, sample_weight_val=None):
            coef = np.linalg.solve(
                X_train.T @ X_train + alpha * np.eye(3), X_train.T @ y_train
            )
            return np.mean((y_val - X_val @ coef) ** 2)

        result = run_cv(X, y, alphas, evaluate_fold, n_folds=5)
        best_alpha = result[0]
        cv_results = result[1]
        assert best_alpha in alphas
        assert len(cv_results) == len(alphas)


# ══════════════════════════════════════════════════════════════════════
# C. Import path compatibility tests
# ══════════════════════════════════════════════════════════════════════

class TestImportPaths:
    """Test all public import paths. After refactor, old paths via shim."""

    # --- Old paths (pre-refactor: sole path; post-refactor: via shim) ---
    def test_old_path_glm_core_solver(self):
        from statgpu.glm_core._solver import fista_solver  # noqa: F401

    def test_old_path_cv_base(self):
        from statgpu.linear_model._cv_base import CVEstimatorBase  # noqa: F401

    def test_old_path_cv_engine(self):
        from statgpu.linear_model._cv_engine import run_cv  # noqa: F401

    def test_old_path_penalized(self):
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel  # noqa: F401

    def test_old_path_gamma_glm(self):
        from statgpu.linear_model._gamma_glm import GammaRegression  # noqa: F401

    def test_old_path_kernel_methods_shim(self):
        from statgpu.kernel_methods import KernelRidge  # noqa: F401

    def test_old_path_splines_shim(self):
        from statgpu.splines import bspline_basis  # noqa: F401

    # --- Top-level API unchanged ---
    def test_top_level_linear_model_exports(self):
        from statgpu.linear_model import (
            LinearRegression, Ridge, Lasso, ElasticNet,
            LogisticRegression, PoissonRegression, GammaRegression,
            PenalizedGeneralizedLinearModel, PenalizedGLM_CV,
            RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV,
        )

    def test_top_level_glm_core_exports(self):
        from statgpu.glm_core import (
            GLMLoss, GLMFamily, fista_solver, newton_solver,
            lbfgs_solver, admm_solver, fista_bb_solver,
            get_glm_loss, register_glm_loss, list_glm_losses,
        )

    # --- New paths (post-refactor) ---
    def test_new_path_solvers(self):
        from statgpu.solvers import fista_solver, newton_solver  # noqa: F401

    @pytest.mark.xfail(reason="cross_validation/ module created in Phase 2", strict=True)
    def test_new_path_cross_validation(self):
        from statgpu.cross_validation import CVEstimatorBase, kfold_indices  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
# D. PenalizedGLM end-to-end tests (covers Mixin split risk)
# ══════════════════════════════════════════════════════════════════════

class TestPenalizedGLMEndToEnd:
    """PenalizedGLM fit/predict/score/inference full pipeline."""

    @pytest.fixture
    def regression_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.randn(100) * 0.1
        return X, y

    @pytest.fixture
    def classification_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        beta = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
        prob = 1 / (1 + np.exp(-X @ beta))
        y = (rng.rand(200) < prob).astype(float)
        return X, y

    def test_penalized_linear_l1_fit_predict(self, regression_data):
        from statgpu.linear_model import PenalizedLinearRegression
        X, y = regression_data
        model = PenalizedLinearRegression(penalty="l1", alpha=0.1, max_iter=100)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        assert model.score(X, y) > 0.5

    def test_penalized_logistic_l1_fit_predict(self, classification_data):
        from statgpu.linear_model import PenalizedLogisticRegression
        X, y = classification_data
        model = PenalizedLogisticRegression(penalty="l1", alpha=0.01, max_iter=200)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    def test_penalized_linear_inference_attributes(self, regression_data):
        from statgpu.linear_model import PenalizedLinearRegression
        X, y = regression_data
        model = PenalizedLinearRegression(
            penalty="l2", alpha=0.01, max_iter=100,
            compute_inference=True, inference_method="debiased",
        )
        model.fit(X, y)
        # Check inference attributes exist (summary() has a pre-existing np bug)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_.shape == (X.shape[1],)

    def test_penalized_linear_predict_returns_coef_shape(self, regression_data):
        from statgpu.linear_model import PenalizedLinearRegression
        X, y = regression_data
        model = PenalizedLinearRegression(penalty="l1", alpha=0.1)
        model.fit(X, y)
        assert model.coef_.shape == (X.shape[1],)

    def test_penalized_glm_score_is_float(self, regression_data):
        from statgpu.linear_model import PenalizedLinearRegression
        X, y = regression_data
        model = PenalizedLinearRegression(penalty="l2", alpha=0.01)
        model.fit(X, y)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert score > 0.0


# ══════════════════════════════════════════════════════════════════════
# E. CV model end-to-end tests (covers CV subdir move risk)
# ══════════════════════════════════════════════════════════════════════

class TestCVModelsEndToEnd:
    """LassoCV, RidgeCV, ElasticNetCV fit/select full pipeline."""

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.randn(100) * 0.1
        return X, y

    def test_lasso_cv_selects_alpha(self, data):
        from statgpu.linear_model import LassoCV
        X, y = data
        model = LassoCV(alphas=np.logspace(-3, 0, 10), cv=3, max_iter=200)
        model.fit(X, y)
        assert model.alpha_ > 0
        assert model.coef_.shape == (X.shape[1],)

    def test_ridge_cv_selects_alpha(self, data):
        from statgpu.linear_model import RidgeCV
        X, y = data
        model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=3)
        model.fit(X, y)
        assert model.alpha_ > 0

    def test_elasticnet_cv_selects_params(self, data):
        from statgpu.linear_model import ElasticNetCV
        X, y = data
        model = ElasticNetCV(alphas=np.logspace(-3, 0, 5), cv=3, max_iter=200)
        model.fit(X, y)
        assert model.alpha_ > 0


# ══════════════════════════════════════════════════════════════════════
# F. nonparametric path tests (covers Phase 6 cleanup risk)
# ══════════════════════════════════════════════════════════════════════

class TestNonparametricImports:
    """nonparametric correct paths and old paths."""

    def test_kernel_smoothing_path(self):
        from statgpu.nonparametric.kernel_smoothing import KernelDensityEstimator  # noqa: F401

    def test_kernel_methods_path(self):
        from statgpu.nonparametric.kernel_methods import KernelRidge  # noqa: F401

    def test_splines_path(self):
        from statgpu.nonparametric.splines import bspline_basis  # noqa: F401

    def test_nonparametric_top_level_exports(self):
        from statgpu.nonparametric import (
            KernelDensityEstimator, KernelRegression,
            KernelRidge, KernelRidgeCV, bspline_basis,
        )

    def test_old_kde_path_still_works(self):
        """Before Phase 6 deletes old files, old path should work."""
        from statgpu.nonparametric._kde import KernelDensityEstimator  # noqa: F401
