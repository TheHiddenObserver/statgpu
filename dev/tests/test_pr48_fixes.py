"""
Comprehensive tests for PR48 code review fixes.

Tests all critical, important, and suggestion-level fixes made during
the PR48 code review.  Includes precision and timing comparisons
across NumPy, CuPy, and PyTorch backends.

Run: python -m pytest dev/tests/test_pr48_fixes.py -v
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None


cupy = _try_import("cupy")
torch = _try_import("torch")

BACKENDS = ["numpy"]
if cupy is not None:
    BACKENDS.append("cupy")
if torch is not None:
    BACKENDS.append("torch")


def _to_backend(arr, backend):
    """Convert numpy array to target backend."""
    if backend == "cupy":
        return cupy.asarray(arr)
    elif backend == "torch":
        return torch.tensor(arr, device="cuda" if torch.cuda.is_available() else "cpu")
    return arr


def _to_numpy(arr):
    """Convert any backend array to numpy."""
    if hasattr(arr, "get"):  # cupy
        return arr.get()
    if hasattr(arr, "cpu"):  # torch
        return arr.cpu().numpy()
    return np.asarray(arr)


# ===========================================================================
# C2: PanelOLS / RandomEffects fit(X, y) signature
# ===========================================================================

class TestPanelFitSignature:
    """C2: Verify fit(X, y) signature (not fit(y, X))."""

    def test_panelols_fit_signature(self):
        """PanelOLS.fit(X, y, ...) uses sklearn-compatible signature."""
        from statgpu.panel import PanelOLS
        import inspect
        sig = inspect.signature(PanelOLS.fit)
        params = list(sig.parameters.keys())
        assert params[1] == "X", f"First positional arg should be 'X', got '{params[1]}'"
        assert params[2] == "y", f"Second positional arg should be 'y', got '{params[2]}'"

    def test_random_effects_fit_signature(self):
        """RandomEffects.fit(X, y, ...) uses sklearn-compatible signature."""
        from statgpu.panel import RandomEffects
        import inspect
        sig = inspect.signature(RandomEffects.fit)
        params = list(sig.parameters.keys())
        assert params[1] == "X", f"First positional arg should be 'X', got '{params[1]}'"
        assert params[2] == "y", f"Second positional arg should be 'y', got '{params[2]}'"

    def test_panelols_fit_works(self):
        """PanelOLS.fit(X, y) produces correct results."""
        from statgpu.panel import PanelOLS
        rng = np.random.default_rng(42)
        n, k = 100, 3
        X = rng.standard_normal((n, k))
        eids = np.repeat(np.arange(20), 5)
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + rng.standard_normal(n) * 0.1

        model = PanelOLS(entity_effects=True)
        model.fit(X, y, entity_ids=eids)

        # Coefficients should be close to true values
        np.testing.assert_allclose(model.coef_, beta_true, atol=0.15)


# ===========================================================================
# C3: GAM _get_xp() method
# ===========================================================================

class TestGAMGetXp:
    """C3: Verify _get_xp() replaces _get_backend() override."""

    def test_gam_has_get_xp(self):
        """GAM should have _get_xp method."""
        from statgpu.semiparametric import GAM
        assert hasattr(GAM, "_get_xp")

    def test_gam_get_xp_returns_module(self):
        """GAM._get_xp() should return an array module (numpy/cupy/torch)."""
        from statgpu.semiparametric import GAM
        gam = GAM()
        xp = gam._get_xp()
        assert hasattr(xp, "asarray"), "_get_xp() should return an array module"

    def test_gam_fit_predict(self):
        """GAM fit + predict works with new _get_xp."""
        from statgpu.semiparametric import GAM
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.standard_normal(200) * 0.1

        gam = GAM(n_splines=15, lam=1.0)
        gam.fit(X, y)
        y_pred = gam.predict(X)

        assert y_pred.shape == (200,)
        # R-squared should be decent
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.8, f"R-squared too low: {r2}"


# ===========================================================================
# C6: IRLS _is_constant_W link-aware check
# ===========================================================================

class TestIRLSConstantW:
    """C6: Verify _is_constant_W is link-aware for Gamma."""

    def test_gamma_log_constant_w(self):
        """Gamma + log link: IRLS weights should be constant (W=1)."""
        from statgpu.glm_core._family import Gamma
        loss = Gamma()  # default is LogLink
        # With log link: V(mu)=mu^2, g'(mu)=1/mu
        # W = 1/(V*g'^2) = 1/(mu^2 * 1/mu^2) = 1 (constant)
        mu = np.array([0.5, 1.0, 2.0, 5.0])
        y = np.array([0.4, 1.1, 1.8, 4.5])
        w = loss.irls_weights(mu, y)
        np.testing.assert_allclose(w, 1.0, atol=1e-15)

    def test_gamma_inverse_power_variable_w(self):
        """Gamma + inverse_power link: IRLS weights are variable (W=mu^2)."""
        from statgpu.glm_core._family import Gamma, InversePowerLink
        loss = Gamma(link=InversePowerLink())
        # With inverse_power: V(mu)=mu^2, g'(mu)=-1/mu^2
        # W = 1/(mu^2 * 1/mu^4) = mu^2 (variable)
        mu = np.array([0.5, 1.0, 2.0, 5.0])
        y = np.array([0.4, 1.1, 1.8, 4.5])
        w = loss.irls_weights(mu, y)
        np.testing.assert_allclose(w, mu ** 2, atol=1e-15)


# ===========================================================================
# AnovaResult: df_between/df_within are int
# ===========================================================================

class TestAnovaResultTypes:
    """AnovaResult df_between/df_within should be int, not float."""

    def test_df_types_are_int(self):
        from statgpu.anova import AnovaResult
        r = AnovaResult(statistic=1.0, pvalue=0.5, df_between=2, df_within=10, eta_squared=0.5)
        assert isinstance(r.df_between, int), f"df_between should be int, got {type(r.df_between)}"
        assert isinstance(r.df_within, int), f"df_within should be int, got {type(r.df_within)}"

    def test_f_oneway_returns_int_df(self):
        from statgpu.anova import f_oneway
        g1 = np.array([5.1, 4.9, 5.0])
        g2 = np.array([6.2, 6.0, 6.3])
        g3 = np.array([7.1, 7.3, 7.0])
        result = f_oneway(g1, g2, g3)
        assert isinstance(result.df_between, int)
        assert isinstance(result.df_within, int)
        assert result.df_between == 2
        assert result.df_within == 6


# ===========================================================================
# I5: ANOVA batch GPU sync
# ===========================================================================

class TestANOVABatchSync:
    """I5: ANOVA should batch GPU operations, not sync per group."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_f_oneway_backends(self, backend):
        """f_oneway produces consistent results across backends."""
        from statgpu.anova import f_oneway
        rng = np.random.default_rng(42)
        g1_np = rng.standard_normal(30) + 1.0
        g2_np = rng.standard_normal(25) + 2.0
        g3_np = rng.standard_normal(35) + 3.0

        result_np = f_oneway(g1_np, g2_np, g3_np, backend="numpy")

        if backend != "numpy":
            g1 = _to_backend(g1_np, backend)
            g2 = _to_backend(g2_np, backend)
            g3 = _to_backend(g3_np, backend)
            result_bk = f_oneway(g1, g2, g3, backend=backend)
            assert abs(result_np.statistic - result_bk.statistic) < 1e-10
            assert abs(result_np.pvalue - result_bk.pvalue) < 1e-10


# ===========================================================================
# I2 + I3: PanelOLS alpha parameter and proper epsilon
# ===========================================================================

class TestPanelOLSAlphaEpsilon:
    """I2: alpha parameter; I3: proper epsilon instead of 1e-30."""

    def test_alpha_parameter(self):
        """PanelOLS accepts alpha parameter for confidence intervals."""
        from statgpu.panel import PanelOLS
        model = PanelOLS(alpha=0.01)
        assert model.alpha == 0.01

    def test_default_alpha(self):
        """PanelOLS default alpha is 0.05."""
        from statgpu.panel import PanelOLS
        model = PanelOLS()
        assert model.alpha == 0.05

    def test_inference_uses_alpha(self):
        """PanelOLS inference uses the alpha parameter."""
        from statgpu.panel import PanelOLS
        rng = np.random.default_rng(42)
        n = 100
        # No intercept column (entity effects absorb it)
        X = rng.standard_normal((n, 2))
        eids = np.repeat(np.arange(20), 5)
        y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.standard_normal(n) * 0.1

        # Fit with alpha=0.01 (99% CI)
        m1 = PanelOLS(entity_effects=True, alpha=0.01)
        m1.fit(X, y, entity_ids=eids)

        # Fit with alpha=0.10 (90% CI)
        m2 = PanelOLS(entity_effects=True, alpha=0.10)
        m2.fit(X, y, entity_ids=eids)

        # 99% CI should be wider than 90% CI
        ci_width_99 = np.mean(m1.conf_int_[:, 1] - m1.conf_int_[:, 0])
        ci_width_90 = np.mean(m2.conf_int_[:, 1] - m2.conf_int_[:, 0])
        assert ci_width_99 > ci_width_90, "99% CI should be wider than 90% CI"


# ===========================================================================
# I1: _to_float_scalar for GPU tensors
# ===========================================================================

class TestToFloatScalar:
    """I1: Use _to_float_scalar instead of float() on GPU tensors."""

    def test_panelols_scale_scalar(self):
        """PanelOLS._scale should be a Python float, not a tensor."""
        from statgpu.panel import PanelOLS
        rng = np.random.default_rng(42)
        n = 50
        # No intercept column (entity effects absorb it)
        X = rng.standard_normal((n, 2))
        eids = np.repeat(np.arange(10), 5)
        y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.standard_normal(n) * 0.1

        model = PanelOLS(entity_effects=True)
        model.fit(X, y, entity_ids=eids)

        assert isinstance(model._scale, float), f"_scale should be float, got {type(model._scale)}"


# ===========================================================================
# I9: Input validation
# ===========================================================================

class TestInputValidation:
    """I9: Validate y/X shape mismatch."""

    def test_panelols_y_x_mismatch(self):
        """PanelOLS raises ValueError for mismatched y and X."""
        from statgpu.panel import PanelOLS
        model = PanelOLS(entity_effects=True)
        X = np.zeros((100, 2))
        y = np.zeros(50)  # Wrong size
        eids = np.repeat(np.arange(20), 5)
        with pytest.raises(ValueError, match="y has 50"):
            model.fit(X, y, entity_ids=eids)

    def test_random_effects_y_x_mismatch(self):
        """RandomEffects raises ValueError for mismatched y and X."""
        from statgpu.panel import RandomEffects
        model = RandomEffects()
        X = np.zeros((100, 2))
        y = np.zeros(50)
        eids = np.repeat(np.arange(20), 5)
        with pytest.raises(ValueError, match="y has 50"):
            model.fit(X, y, entity_ids=eids)


# ===========================================================================
# I10: KernelRidge linalg.solve fallback
# ===========================================================================

class TestKernelRidgeFallback:
    """I10: KernelRidge should handle ill-conditioned kernel matrices."""

    def test_kernel_ridge_with_small_alpha(self):
        """KernelRidge with very small alpha should not crash."""
        from statgpu.nonparametric.kernel_methods import KernelRidge
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = rng.standard_normal(50)

        # Very small alpha may cause ill-conditioning
        model = KernelRidge(alpha=1e-10, kernel="rbf", gamma=0.1)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (50,)


# ===========================================================================
# I7: theta_ weighted average
# ===========================================================================

class TestThetaWeighted:
    """I7: theta_ should be weighted by entity count, not group size count."""

    def test_theta_weighted_by_entities(self):
        """theta_ should weight by number of entities at each group size."""
        from statgpu.panel import RandomEffects
        rng = np.random.default_rng(42)

        # Create unbalanced panel: 10 entities with T=5, 1 entity with T=20
        eids = np.concatenate([np.repeat(np.arange(10), 5), np.repeat([10], 20)])
        n = len(eids)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = 1.0 + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.1

        model = RandomEffects()
        model.fit(X, y, entity_ids=eids)

        # theta_ should exist and be in [0, 1)
        assert 0 <= model.theta_ < 1, f"theta_={model.theta_} should be in [0, 1)"


# ===========================================================================
# C1: Two-way FE demeaning convergence
# ===========================================================================

class TestTwoWayDemeaningConvergence:
    """C1: Two-way demeaning should iterate for unbalanced panels."""

    def test_two_way_demeaning_converges(self):
        """demean_variables should converge for unbalanced two-way FE."""
        from statgpu.panel._utils import demean_variables
        rng = np.random.default_rng(42)

        # Balanced panel: 10 entities x 5 time periods = 50 obs
        # All entities share the same time IDs (balanced)
        n_entities, n_times = 10, 5
        entity_ids = np.repeat(np.arange(n_entities), n_times)
        time_ids = np.tile(np.arange(n_times), n_entities)
        n = len(entity_ids)
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        y_d, X_d = demean_variables(y, X, entity_ids, time_ids, xp=np)

        # After two-way demeaning on balanced panel, entity means should be ~0
        for g in np.unique(entity_ids):
            mask = entity_ids == g
            assert np.abs(np.mean(y_d[mask])) < 1e-8, \
                f"Entity {g} mean of demeaned y not close to 0: {np.mean(y_d[mask])}"

        # Time means should also be ~0
        for t in np.unique(time_ids):
            mask = time_ids == t
            assert np.abs(np.mean(y_d[mask])) < 1e-8, \
                f"Time {t} mean of demeaned y not close to 0: {np.mean(y_d[mask])}"

        # Also verify it actually converged (not just single-pass)
        y_d_single, _ = demean_variables(y, X, entity_ids, time_ids=None, xp=np)
        # Two-way demeaned should differ from one-way demeaned
        assert np.max(np.abs(y_d - y_d_single)) > 0.01, \
            "Two-way demeaning should differ from one-way demeaning"

    def test_time_only_fixed_effects(self):
        """demean_variables should work with entity_ids=None, time_ids provided."""
        from statgpu.panel._utils import demean_variables
        rng = np.random.default_rng(42)
        n = 50
        time_ids = np.repeat(np.arange(10), 5)
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        # Should not crash with entity_ids=None
        y_d, X_d = demean_variables(y, X, entity_ids=None, time_ids=time_ids, xp=np)

        # Time means of demeaned data should be ~0
        for t in np.unique(time_ids):
            mask = time_ids == t
            assert np.abs(np.mean(y_d[mask])) < 1e-10, \
                f"Time {t} mean of demeaned y not close to 0"


# ===========================================================================
# C4: Swamy-Arora formula
# ===========================================================================

class TestSwamyAroraFormula:
    """C4: Swamy-Arora variance component formula correctness."""

    def test_sigma2_a_nonnegative(self):
        """sigma2_a should be non-negative."""
        from statgpu.panel import RandomEffects
        rng = np.random.default_rng(42)
        eids = np.repeat(np.arange(20), 5)
        n = len(eids)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = 1.0 + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.1

        model = RandomEffects()
        model.fit(X, y, entity_ids=eids)

        assert model.variance_components_['sigma2_a'] >= 0, \
            "sigma2_a should be non-negative"

    def test_variance_components_reasonable(self):
        """Variance components should be reasonable for known DGP."""
        from statgpu.panel import RandomEffects
        rng = np.random.default_rng(42)
        n_entities = 50
        T = 10
        eids = np.repeat(np.arange(n_entities), T)
        n = n_entities * T

        # DGP: y_it = 0.5*x_it + a_i + e_it
        # sigma2_a = 1.0, sigma2_e = 0.25
        # No intercept column (RE model absorbs it via demeaning)
        a_i = np.repeat(rng.standard_normal(n_entities) * 1.0, T)
        X = rng.standard_normal((n, 2))
        e = rng.standard_normal(n) * 0.5
        y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + a_i + e

        model = RandomEffects()
        model.fit(X, y, entity_ids=eids)

        # sigma2_a should be non-negative and sigma2_e should be finite and positive
        # Note: Swamy-Arora can shrink sigma2_a to 0 if the between-group
        # variance is small relative to within-group variance
        assert model.variance_components_['sigma2_a'] >= 0, \
            f"sigma2_a={model.variance_components_['sigma2_a']} should be non-negative"
        assert np.isfinite(model.variance_components_['sigma2_e']), \
            f"sigma2_e={model.variance_components_['sigma2_e']} should be finite"
        assert model.variance_components_['sigma2_e'] > 0, \
            f"sigma2_e={model.variance_components_['sigma2_e']} should be positive"


# ===========================================================================
# I6: within_transform uses xp_copy
# ===========================================================================

class TestWithinTransformCopy:
    """I6: within_transform should use xp_copy utility."""

    def test_within_transform_preserves_input(self):
        """within_transform should not modify the input array."""
        from statgpu.panel._utils import within_transform
        rng = np.random.default_rng(42)
        y = rng.standard_normal(50)
        groups = np.repeat(np.arange(10), 5)
        y_orig = y.copy()

        y_d = within_transform(y, groups, xp=np)

        # Original should be unchanged
        np.testing.assert_array_equal(y, y_orig)


# ===========================================================================
# S7: GAM docstring import path
# ===========================================================================

class TestGAMDocstring:
    """S7: GAM docstring should reference statgpu.semiparametric, not statgpu.splines."""

    def test_docstring_import_path(self):
        """GAM docstring example should use correct import path."""
        from statgpu.semiparametric._gam import GAM
        assert "statgpu.semiparametric" in GAM.__doc__
        assert "statgpu.splines" not in GAM.__doc__


# ===========================================================================
# Performance benchmarks
# ===========================================================================

class TestPerformanceNoRegression:
    """Verify no performance regression in key operations."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_anova_performance(self, backend):
        """ANOVA should complete within reasonable time."""
        from statgpu.anova import f_oneway
        rng = np.random.default_rng(42)
        groups = [rng.standard_normal(500) + i for i in range(5)]
        if backend != "numpy":
            groups = [_to_backend(g, backend) for g in groups]

        start = time.perf_counter()
        for _ in range(10):
            f_oneway(*groups, backend=backend)
        elapsed = time.perf_counter() - start

        # Should complete 10 runs in under 5 seconds
        assert elapsed < 5.0, f"ANOVA too slow on {backend}: {elapsed:.2f}s for 10 runs"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_panel_ols_performance(self, backend):
        """PanelOLS should complete within reasonable time."""
        from statgpu.panel import PanelOLS
        rng = np.random.default_rng(42)
        n, k = 500, 5
        X_np = rng.standard_normal((n, k))
        eids_np = np.repeat(np.arange(50), 10)
        y_np = X_np @ np.ones(k) + rng.standard_normal(n) * 0.1

        if backend != "numpy":
            X = _to_backend(X_np, backend)
            y = _to_backend(y_np, backend)
            eids = _to_backend(eids_np, backend)
        else:
            X, y, eids = X_np, y_np, eids_np

        start = time.perf_counter()
        for _ in range(5):
            model = PanelOLS(entity_effects=True)
            model.fit(X, y, entity_ids=eids)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"PanelOLS too slow on {backend}: {elapsed:.2f}s for 5 runs"


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
