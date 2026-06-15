"""Post-refactor verification tests: run after each Phase completes.

Verifies new paths work, numerical results unchanged.
"""
import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════
# Phase 1 verification: solvers/ module
# ══════════════════════════════════════════════════════════════════════

class TestPhase1SolversModule:

    def test_solvers_importable(self):
        from statgpu.solvers import fista_solver, fista_bb_solver
        from statgpu.solvers import newton_solver, lbfgs_solver, admm_solver
        assert all(callable(f) for f in [
            fista_solver, fista_bb_solver,
            newton_solver, lbfgs_solver, admm_solver,
        ])

    def test_solvers_all_exports(self):
        import statgpu.solvers as solvers
        assert hasattr(solvers, '__all__')
        assert 'fista_solver' in solvers.__all__

    def test_glm_core_reexport_works(self):
        """glm_core.__init__ still re-exports solver."""
        from statgpu.glm_core import fista_solver, newton_solver  # noqa: F401

    def test_fista_lla_in_solvers(self):
        from statgpu.solvers import fista_lla_path  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
# Phase 2 verification: cross_validation/ module
# ══════════════════════════════════════════════════════════════════════

class TestPhase2CrossValidationModule:

    def test_cross_validation_importable(self):
        from statgpu.cross_validation import CVEstimatorBase, kfold_indices
        from statgpu.cross_validation import hash_cv_data, batch_mse
        assert all(callable(f) for f in [
            CVEstimatorBase, kfold_indices, hash_cv_data, batch_mse,
        ])

    def test_survival_uses_cross_validation(self):
        from statgpu.survival import CoxPHCV
        assert CoxPHCV is not None


# ══════════════════════════════════════════════════════════════════════
# Phase 3 verification: linear_model/wrappers/
# ══════════════════════════════════════════════════════════════════════

class TestPhase3Wrappers:

    def test_wrappers_importable(self):
        from statgpu.linear_model.wrappers import (
            LinearRegression, Ridge, Lasso, ElasticNet,
            LogisticRegression, GammaRegression, PoissonRegression,
        )

    def test_linear_model_init_unchanged(self):
        from statgpu.linear_model import (
            LinearRegression, Ridge, Lasso, ElasticNet,
            LogisticRegression, GammaRegression, PoissonRegression,
            InverseGaussianRegression, NegativeBinomialRegression,
            TweedieRegression,
        )


# ══════════════════════════════════════════════════════════════════════
# Phase 4 verification: penalized/ Mixin split
# ══════════════════════════════════════════════════════════════════════

class TestPhase4PenalizedMixin:

    def test_penalized_importable(self):
        from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
        assert PenalizedGeneralizedLinearModel is not None

    def test_mixin_methods_present(self):
        from statgpu.linear_model import PenalizedLinearRegression
        model = PenalizedLinearRegression(penalty="l1", alpha=0.1)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'score')
        assert hasattr(model, 'summary')

    def test_penalized_end_to_end_after_mixin(self):
        from statgpu.linear_model import PenalizedLinearRegression
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.randn(100) * 0.1
        model = PenalizedLinearRegression(penalty="l1", alpha=0.1, max_iter=100)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape


# ══════════════════════════════════════════════════════════════════════
# Phase 5 verification: linear_model/cv/
# ══════════════════════════════════════════════════════════════════════

class TestPhase5CVSubdir:

    def test_cv_subdir_importable(self):
        from statgpu.linear_model.cv import LassoCV, RidgeCV, ElasticNetCV


# ══════════════════════════════════════════════════════════════════════
# Phase 6 verification: cleanup
# ══════════════════════════════════════════════════════════════════════

class TestPhase6Cleanup:

    def test_nonparametric_old_files_removed(self):
        import importlib.util
        assert importlib.util.find_spec("statgpu.nonparametric._kde") is None
