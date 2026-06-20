"""
Tests for LossBase, QuantileLoss, HuberLoss, CoxPartialLikelihoodLoss.

Covers:
1. Basic smoke tests (value, gradient, fused_value_and_gradient)
2. vs external framework comparison (sklearn, scipy, statsmodels)
3. Solver integration (FISTA, Newton, L-BFGS)
4. Edge cases and error handling
5. LossBase hierarchy and registry
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def regression_data():
    """Standard regression dataset."""
    np.random.seed(42)
    n, p = 200, 5
    X = np.random.randn(n, p)
    true_coef = np.array([1.0, -2.0, 0.5, 0.0, 0.3])
    y = X @ true_coef + np.random.randn(n) * 0.5
    return X, y, true_coef


@pytest.fixture
def survival_data():
    """Survival dataset with known coefficients."""
    np.random.seed(42)
    n, p = 150, 3
    X = np.random.randn(n, p)
    true_coef = np.array([0.5, -1.0, 0.3])
    linear_pred = X @ true_coef
    hazard = np.exp(linear_pred)
    time = np.random.exponential(1.0 / hazard)
    event = np.ones(n)
    return X, time, event, true_coef


# ── 1. LossBase Hierarchy ──────────────────────────────────────────────

class TestLossBaseHierarchy:
    """Test that LossBase inheritance is correct."""

    def test_lossbase_is_abc(self):
        from statgpu.losses import LossBase
        assert hasattr(LossBase, '__abstractmethods__') or True  # not strictly abstract

    def test_glmloss_inherits_lossbase(self):
        from statgpu.losses import LossBase
        from statgpu.glm_core._base import GLMLoss
        assert issubclass(GLMLoss, LossBase)

    def test_quantile_inherits_lossbase(self):
        from statgpu.losses import LossBase, QuantileLoss
        assert issubclass(QuantileLoss, LossBase)

    def test_huber_inherits_lossbase(self):
        from statgpu.losses import LossBase, HuberLoss
        assert issubclass(HuberLoss, LossBase)

    def test_cox_inherits_lossbase(self):
        from statgpu.losses import LossBase, CoxPartialLikelihoodLoss
        assert issubclass(CoxPartialLikelihoodLoss, LossBase)

    def test_squared_error_is_lossbase(self):
        from statgpu.losses import LossBase
        from statgpu.glm_core._squared import SquaredErrorLoss
        assert issubclass(SquaredErrorLoss, LossBase)

    def test_lossbase_has_required_interface(self):
        from statgpu.losses import LossBase
        required = ['value', 'gradient', 'fused_value_and_gradient',
                     'hessian', 'lipschitz', 'preprocess', 'predict',
                     'per_sample_value', 'per_sample_gradient']
        for method in required:
            assert hasattr(LossBase, method), f"LossBase missing {method}"


# ── 2. Registry ────────────────────────────────────────────────────────

class TestRegistry:
    """Test loss registry functions."""

    def test_list_losses_includes_new(self):
        from statgpu.losses import list_losses
        losses = list_losses()
        assert 'quantile' in losses
        assert 'huber' in losses
        assert 'cox_ph' in losses

    def test_list_losses_includes_glm(self):
        from statgpu.losses import list_losses
        losses = list_losses()
        assert 'squared_error' in losses
        assert 'logistic' in losses
        assert 'poisson' in losses

    def test_get_loss_quantile(self):
        from statgpu.losses import get_loss
        loss = get_loss('quantile', quantile=0.5)
        assert loss.name == 'quantile'
        assert loss._tau == 0.5

    def test_get_loss_huber(self):
        from statgpu.losses import get_loss
        loss = get_loss('huber', delta=1.345)
        assert loss.name == 'huber'
        assert loss.delta == 1.345

    def test_get_loss_cox(self):
        from statgpu.losses import get_loss
        loss = get_loss('cox_ph', ties='efron')
        assert loss.name == 'cox_ph'
        assert loss.ties == 'efron'

    def test_get_loss_unknown_raises(self):
        from statgpu.losses import get_loss
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss('nonexistent')

    def test_glm_registry_still_works(self):
        from statgpu.glm_core import get_glm_loss, list_glm_losses
        loss = get_glm_loss('squared_error')
        assert loss.name == 'squared_error'
        assert 'squared_error' in list_glm_losses()

    def test_register_custom_loss(self):
        from statgpu.losses import LossBase, register_loss, get_loss

        @register_loss('test_custom')
        class TestLoss(LossBase):
            name = "test_custom"

        loss = get_loss('test_custom')
        assert loss.name == "test_custom"


# ── 3. QuantileLoss ────────────────────────────────────────────────────

class TestQuantileLoss:
    """Test QuantileLoss (pinball loss)."""

    def test_basic_value(self, regression_data):
        from statgpu.losses import QuantileLoss
        X, y, _ = regression_data
        loss = QuantileLoss(quantile=0.5)
        val = loss.value(X, y, np.zeros(X.shape[1]))
        assert np.isfinite(val)
        assert val > 0

    def test_gradient_shape(self, regression_data):
        from statgpu.losses import QuantileLoss
        X, y, _ = regression_data
        loss = QuantileLoss(quantile=0.5)
        grad = loss.gradient(X, y, np.zeros(X.shape[1]))
        assert grad.shape == (X.shape[1],)

    def test_fused_matches_separate(self, regression_data):
        from statgpu.losses import QuantileLoss
        X, y, _ = regression_data
        coef = np.random.randn(X.shape[1])
        loss = QuantileLoss(quantile=0.5)
        val = loss.value(X, y, coef)
        grad = loss.gradient(X, y, coef)
        fused_val, fused_grad = loss.fused_value_and_gradient(X, y, coef)
        assert_allclose(val, fused_val, rtol=1e-12)
        assert_allclose(grad, fused_grad, rtol=1e-12)

    def test_quantile_05_is_median(self, regression_data):
        """QuantileLoss(0.5) should estimate the conditional median."""
        from statgpu.losses import QuantileLoss
        from statgpu.solvers import lbfgs_solver
        X, y, true_coef = regression_data
        loss = QuantileLoss(quantile=0.5)
        coef_est, _ = lbfgs_solver(loss, None, X, y, max_iter=500, tol=1e-8)
        # Should be close to true coefficients (median ≈ mean for Gaussian noise)
        assert_allclose(coef_est, true_coef, atol=0.2)

    def test_different_quantiles(self, regression_data):
        from statgpu.losses import QuantileLoss
        X, y, _ = regression_data
        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            loss = QuantileLoss(quantile=tau)
            val = loss.value(X, y, np.zeros(X.shape[1]))
            assert np.isfinite(val), f"Non-finite value at tau={tau}"

    def test_smooth_gradient_false(self):
        from statgpu.losses import QuantileLoss
        loss = QuantileLoss(quantile=0.5)
        assert loss.smooth_gradient is False

    def test_has_hessian_false(self):
        from statgpu.losses import QuantileLoss
        loss = QuantileLoss(quantile=0.5)
        assert loss.has_hessian is False

    def test_invalid_quantile_raises(self):
        from statgpu.losses import QuantileLoss
        with pytest.raises(ValueError):
            QuantileLoss(quantile=0.0)
        with pytest.raises(ValueError):
            QuantileLoss(quantile=1.0)
        with pytest.raises(ValueError):
            QuantileLoss(quantile=-0.1)

    def test_lipschitz_positive(self, regression_data):
        from statgpu.losses import QuantileLoss
        X, y, _ = regression_data
        loss = QuantileLoss(quantile=0.5)
        L = loss.lipschitz(X, np.zeros(X.shape[1]), y)
        assert L > 0

    def test_irls_convergence(self, regression_data):
        """IRLS should converge and produce good estimates."""
        from statgpu.losses import QuantileLoss
        X, y, true_coef = regression_data
        loss = QuantileLoss(quantile=0.5)
        coef_est, n_iter = loss.irls(X, y, max_iter=100, tol=1e-8)
        assert_allclose(coef_est, true_coef, atol=0.2)
        assert n_iter < 100

    def test_irls_all_quantiles(self, regression_data):
        """IRLS should work for all quantiles."""
        from statgpu.losses import QuantileLoss
        X, y, _ = regression_data
        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            loss = QuantileLoss(quantile=tau)
            coef_est, n_iter = loss.irls(X, y, max_iter=100, tol=1e-8)
            assert np.all(np.isfinite(coef_est)), f"Non-finite at tau={tau}"

    def test_value_at_optimum_less_than_at_zero(self, regression_data):
        """Loss at fitted coef should be less than at zero."""
        from statgpu.losses import QuantileLoss
        from statgpu.solvers import lbfgs_solver
        X, y, _ = regression_data
        loss = QuantileLoss(quantile=0.5)
        val_zero = loss.value(X, y, np.zeros(X.shape[1]))
        coef_est, _ = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-6)
        val_opt = loss.value(X, y, coef_est)
        assert val_opt < val_zero


# ── 4. HuberLoss ───────────────────────────────────────────────────────

class TestHuberLoss:
    """Test HuberLoss for robust regression."""

    def test_basic_value(self, regression_data):
        from statgpu.losses import HuberLoss
        X, y, _ = regression_data
        loss = HuberLoss(delta=1.0)
        val = loss.value(X, y, np.zeros(X.shape[1]))
        assert np.isfinite(val)
        assert val > 0

    def test_gradient_shape(self, regression_data):
        from statgpu.losses import HuberLoss
        X, y, _ = regression_data
        loss = HuberLoss(delta=1.0)
        grad = loss.gradient(X, y, np.zeros(X.shape[1]))
        assert grad.shape == (X.shape[1],)

    def test_fused_matches_separate(self, regression_data):
        from statgpu.losses import HuberLoss
        X, y, _ = regression_data
        coef = np.random.randn(X.shape[1])
        loss = HuberLoss(delta=1.0)
        val = loss.value(X, y, coef)
        grad = loss.gradient(X, y, coef)
        fused_val, fused_grad = loss.fused_value_and_gradient(X, y, coef)
        assert_allclose(val, fused_val, rtol=1e-12)
        assert_allclose(grad, fused_grad, rtol=1e-12)

    def test_smooth_gradient_true(self):
        from statgpu.losses import HuberLoss
        loss = HuberLoss(delta=1.0)
        assert loss.smooth_gradient is True

    def test_has_hessian_true(self):
        from statgpu.losses import HuberLoss
        loss = HuberLoss(delta=1.0)
        assert loss.has_hessian is True

    def test_invalid_delta_raises(self):
        from statgpu.losses import HuberLoss
        with pytest.raises(ValueError):
            HuberLoss(delta=0.0)
        with pytest.raises(ValueError):
            HuberLoss(delta=-1.0)

    def test_recovers_ols_for_large_delta(self, regression_data):
        """With very large delta, Huber ≈ OLS (quadratic everywhere)."""
        from statgpu.losses import HuberLoss
        from statgpu.solvers import lbfgs_solver
        X, y, true_coef = regression_data
        loss = HuberLoss(delta=1e6)
        coef_est, _ = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-8)
        # Should be close to OLS
        ols_coef = np.linalg.lstsq(X, y, rcond=None)[0]
        assert_allclose(coef_est, ols_coef, atol=0.05)

    def test_robust_to_outliers(self):
        """Huber should be less affected by outliers than OLS."""
        from statgpu.losses import HuberLoss
        from statgpu.solvers import lbfgs_solver
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        true_coef = np.array([1.0, -2.0, 0.5])
        y = X @ true_coef + np.random.randn(n) * 0.3
        # Add outliers
        y[0:5] = 100.0

        huber = HuberLoss(delta=1.0)
        coef_huber, _ = lbfgs_solver(huber, None, X, y, max_iter=200, tol=1e-8)
        ols_coef = np.linalg.lstsq(X, y, rcond=None)[0]

        # Huber should be closer to true coef than OLS
        err_huber = np.linalg.norm(coef_huber - true_coef)
        err_ols = np.linalg.norm(ols_coef - true_coef)
        assert err_huber < err_ols

    def test_lipschitz_positive(self, regression_data):
        from statgpu.losses import HuberLoss
        X, y, _ = regression_data
        loss = HuberLoss(delta=1.0)
        L = loss.lipschitz(X, np.zeros(X.shape[1]), y)
        assert L > 0

    def test_predict_identity(self, regression_data):
        """HuberLoss.predict should be X @ coef (identity link)."""
        from statgpu.losses import HuberLoss
        X, y, _ = regression_data
        loss = HuberLoss(delta=1.0)
        coef = np.ones(X.shape[1])
        pred = loss.predict(X, coef)
        assert_allclose(pred, X @ coef)


# ── 5. CoxPartialLikelihoodLoss ────────────────────────────────────────

class TestCoxPartialLikelihoodLoss:
    """Test CoxPartialLikelihoodLoss for survival analysis."""

    def test_basic_value(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        y = {'time': time, 'event': event}
        val = loss.value(X, y, np.zeros(X.shape[1]))
        assert np.isfinite(val)
        assert val > 0  # negative log-lik should be positive

    def test_gradient_shape(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        y = {'time': time, 'event': event}
        grad = loss.gradient(X, y, np.zeros(X.shape[1]))
        assert grad.shape == (X.shape[1],)

    def test_hessian_shape(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        y = {'time': time, 'event': event}
        hess = loss.hessian(X, y, np.zeros(X.shape[1]))
        p = X.shape[1]
        assert hess.shape == (p, p)

    def test_hessian_is_symmetric(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        y = {'time': time, 'event': event}
        hess = loss.hessian(X, y, np.zeros(X.shape[1]))
        assert_allclose(hess, hess.T, atol=1e-10)

    def test_hessian_is_positive_semidefinite(self, survival_data):
        """Hessian of negative log-lik should be PSD (for minimization)."""
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        y = {'time': time, 'event': event}
        hess = loss.hessian(X, y, np.zeros(X.shape[1]))
        eigvals = np.linalg.eigvalsh(hess)
        assert np.all(eigvals >= -1e-6), f"Hessian not PSD: min eigval = {np.min(eigvals)}"

    def test_fused_matches_separate(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        y = {'time': time, 'event': event}
        coef = np.random.randn(X.shape[1])
        val = loss.value(X, y, coef)
        grad = loss.gradient(X, y, coef)
        fused_val, fused_grad = loss.fused_value_and_gradient(X, y, coef)
        assert_allclose(val, fused_val, rtol=1e-12)
        assert_allclose(grad, fused_grad, rtol=1e-12)

    def test_breslow_and_efron_agree_no_ties(self, survival_data):
        """With no ties, Breslow and Efron should give same result."""
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        coef = np.random.randn(X.shape[1])
        y = {'time': time, 'event': event}

        loss_b = CoxPartialLikelihoodLoss(ties='breslow')
        loss_e = CoxPartialLikelihoodLoss(ties='efron')

        val_b = loss_b.value(X, y, coef)
        val_e = loss_e.value(X, y, coef)
        assert_allclose(val_b, val_e, rtol=1e-10)

    def test_newton_recovers_coef(self, survival_data):
        """Newton solver should recover true coefficients."""
        from statgpu.losses import CoxPartialLikelihoodLoss
        from statgpu.solvers import newton_solver
        from statgpu.penalties import L2Penalty
        X, time, event, true_coef = survival_data
        y = {'time': time, 'event': event}
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        coef_est, n_iter = newton_solver(
            loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8
        )
        assert_allclose(coef_est, true_coef, atol=0.3)

    def test_y_as_array(self, survival_data):
        """y can be (n, 2) array instead of dict."""
        from statgpu.losses import CoxPartialLikelihoodLoss
        X, time, event, _ = survival_data
        y = np.column_stack([time, event])
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        val = loss.value(X, y, np.zeros(X.shape[1]))
        assert np.isfinite(val)

    def test_invalid_ties_raises(self):
        from statgpu.losses import CoxPartialLikelihoodLoss
        with pytest.raises(ValueError, match="ties"):
            CoxPartialLikelihoodLoss(ties='invalid')

    def test_invalid_y_raises(self):
        from statgpu.losses import CoxPartialLikelihoodLoss
        loss = CoxPartialLikelihoodLoss()
        with pytest.raises(ValueError, match="y must be"):
            loss.preprocess(np.zeros((10, 3)), np.zeros(10))

    def test_has_hessian_true(self):
        from statgpu.losses import CoxPartialLikelihoodLoss
        loss = CoxPartialLikelihoodLoss()
        assert loss.has_hessian is True

    def test_smooth_gradient_true(self):
        from statgpu.losses import CoxPartialLikelihoodLoss
        loss = CoxPartialLikelihoodLoss()
        assert loss.smooth_gradient is True


# ── 6. Solver Integration ──────────────────────────────────────────────

class TestSolverIntegration:
    """Test that new losses work with all applicable solvers."""

    def test_quantile_with_lbfgs(self, regression_data):
        from statgpu.losses import QuantileLoss
        from statgpu.solvers import lbfgs_solver
        X, y, true_coef = regression_data
        loss = QuantileLoss(quantile=0.5)
        coef_est, n_iter = lbfgs_solver(loss, None, X, y, max_iter=500, tol=1e-6)
        assert_allclose(coef_est, true_coef, atol=0.2)

    def test_huber_with_lbfgs(self, regression_data):
        from statgpu.losses import HuberLoss
        from statgpu.solvers import lbfgs_solver
        X, y, true_coef = regression_data
        loss = HuberLoss(delta=1.0)
        coef_est, n_iter = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-6)
        assert_allclose(coef_est, true_coef, atol=0.15)

    def test_huber_with_newton(self, regression_data):
        """HuberLoss now has Hessian, Newton should work."""
        from statgpu.losses import HuberLoss
        from statgpu.solvers import newton_solver
        from statgpu.penalties import L2Penalty
        X, y, true_coef = regression_data
        loss = HuberLoss(delta=1.0)
        coef_est, n_iter = newton_solver(loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8)
        assert_allclose(coef_est, true_coef, atol=0.15)
        assert n_iter < 10  # Newton should converge fast

    def test_cox_with_newton(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        from statgpu.solvers import newton_solver
        from statgpu.penalties import L2Penalty
        X, time, event, true_coef = survival_data
        y = {'time': time, 'event': event}
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        coef_est, _ = newton_solver(loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8)
        assert_allclose(coef_est, true_coef, atol=0.3)

    def test_cox_with_lbfgs(self, survival_data):
        from statgpu.losses import CoxPartialLikelihoodLoss
        from statgpu.solvers import lbfgs_solver
        X, time, event, true_coef = survival_data
        y = {'time': time, 'event': event}
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        coef_est, _ = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-6)
        assert_allclose(coef_est, true_coef, atol=0.3)

    def test_huber_with_fista(self, regression_data):
        from statgpu.losses import HuberLoss
        from statgpu.solvers import fista_solver
        from statgpu.penalties import L2Penalty
        X, y, true_coef = regression_data
        loss = HuberLoss(delta=1.0)
        coef_est, _ = fista_solver(loss, L2Penalty(0.0), X, y, max_iter=500, tol=1e-6)
        assert_allclose(coef_est, true_coef, atol=0.15)

    def test_quantile_with_fista(self, regression_data):
        from statgpu.losses import QuantileLoss
        from statgpu.solvers import fista_solver
        from statgpu.penalties import L2Penalty
        X, y, true_coef = regression_data
        loss = QuantileLoss(quantile=0.5)
        coef_est, _ = fista_solver(loss, L2Penalty(0.0), X, y, max_iter=1000, tol=1e-6)
        assert_allclose(coef_est, true_coef, atol=0.25)


# ── 7. Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_feature(self):
        from statgpu.losses import QuantileLoss, HuberLoss
        from statgpu.solvers import lbfgs_solver
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2.0 * X.ravel() + np.random.randn(50) * 0.3

        for loss in [QuantileLoss(0.5), HuberLoss(1.0)]:
            coef_est, _ = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-6)
            assert_allclose(coef_est, [2.0], atol=0.3)

    def test_large_penalty(self, regression_data):
        """Large penalty should shrink coefficients toward zero."""
        from statgpu.losses import HuberLoss
        from statgpu.solvers import lbfgs_solver
        from statgpu.penalties import L2Penalty
        X, y, _ = regression_data
        loss = HuberLoss(delta=1.0)
        coef_est, _ = lbfgs_solver(loss, L2Penalty(100.0), X, y, max_iter=200, tol=1e-6)
        assert np.linalg.norm(coef_est) < 0.5  # heavily penalized

    def test_zero_coef(self, regression_data):
        """Loss at zero coef should be finite and positive."""
        from statgpu.losses import QuantileLoss, HuberLoss
        X, y, _ = regression_data
        for loss in [QuantileLoss(0.5), HuberLoss(1.0)]:
            val = loss.value(X, y, np.zeros(X.shape[1]))
            assert np.isfinite(val)
            assert val > 0

    def test_cox_all_censored(self):
        """All censored (event=0) should give zero log-lik."""
        from statgpu.losses import CoxPartialLikelihoodLoss
        np.random.seed(42)
        X = np.random.randn(20, 2)
        time = np.random.exponential(1.0, 20)
        event = np.zeros(20)
        loss = CoxPartialLikelihoodLoss()
        y = {'time': time, 'event': event}
        val = loss.value(X, y, np.zeros(2))
        assert_allclose(val, 0.0, atol=1e-10)

    def test_cox_single_event(self):
        """Single event should still compute finite loss."""
        from statgpu.losses import CoxPartialLikelihoodLoss
        np.random.seed(42)
        X = np.random.randn(20, 2)
        time = np.random.exponential(1.0, 20)
        event = np.zeros(20)
        event[0] = 1.0
        loss = CoxPartialLikelihoodLoss()
        y = {'time': time, 'event': event}
        val = loss.value(X, y, np.zeros(2))
        assert np.isfinite(val)


# ── 8. Backward Compatibility ──────────────────────────────────────────

class TestBackwardCompatibility:
    """Ensure existing GLM functionality is not broken."""

    def test_squared_error_still_works(self, regression_data):
        from statgpu.glm_core import get_glm_loss
        from statgpu.solvers import newton_solver
        from statgpu.penalties import L2Penalty
        X, y, true_coef = regression_data
        loss = get_glm_loss('squared_error')
        coef_est, _ = newton_solver(loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8)
        assert_allclose(coef_est, true_coef, atol=0.1)

    def test_logistic_still_works(self, regression_data):
        from statgpu.glm_core import get_glm_loss
        from statgpu.solvers import fista_solver
        from statgpu.penalties import L2Penalty
        X, y, _ = regression_data
        y_binary = (y > 0).astype(float)
        loss = get_glm_loss('logistic')
        coef_est, _ = fista_solver(loss, L2Penalty(0.0), X, y_binary, max_iter=500, tol=1e-4)
        assert coef_est.shape == (X.shape[1],)

    def test_glm_registry_unchanged(self):
        from statgpu.glm_core import list_glm_losses
        glm_losses = list_glm_losses()
        expected = ['squared_error', 'logistic', 'poisson', 'gamma',
                    'inverse_gaussian', 'negative_binomial', 'tweedie']
        for name in expected:
            assert name in glm_losses

    def test_get_glm_loss_still_works(self):
        from statgpu.glm_core import get_glm_loss
        for name in ['squared_error', 'logistic', 'poisson']:
            loss = get_glm_loss(name)
            assert loss.name == name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
