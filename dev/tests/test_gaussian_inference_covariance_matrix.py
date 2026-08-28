"""Independent covariance/Ridge checks for backend-native Gaussian inference."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
    resolve_hac_maxlags,
)


def _problem():
    rng = np.random.default_rng(127)
    X = rng.normal(size=(48, 4))
    coef = np.asarray([0.8, -0.35, 0.2, 0.55])
    intercept = -0.4
    y = intercept + X @ coef + rng.normal(scale=0.35, size=X.shape[0])
    return X, y, coef, intercept


def _independent_covariance(design, resid, cov_type, df_resid, maxlags=None):
    bread = np.linalg.inv(design.T @ design)
    n, _ = design.shape
    if cov_type == "nonrobust":
        scale = float(np.sum(resid**2) / df_resid)
        return scale * bread
    if cov_type == "hac":
        scores = design * resid[:, None]
        lag_count = resolve_hac_maxlags(n, maxlags)
        meat = scores.T @ scores
        for lag in range(1, lag_count + 1):
            weight = 1.0 - lag / (lag_count + 1.0)
            gamma = scores[lag:].T @ scores[:-lag]
            meat += weight * (gamma + gamma.T)
        return bread @ meat @ bread

    leverage = np.sum(design * (design @ bread), axis=1)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)
    if cov_type == "hc2":
        omega = resid**2 / np.maximum(1.0 - leverage, 1e-12)
    elif cov_type == "hc3":
        omega = resid**2 / np.maximum((1.0 - leverage) ** 2, 1e-12)
    else:
        omega = resid**2
    meat = design.T @ (design * omega[:, None])
    if cov_type == "hc1":
        meat *= n / df_resid
    return bread @ meat @ bread


@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
def test_numpy_covariance_surface_matches_independent_formula(cov_type):
    X, y, coef, intercept = _problem()
    state = build_gaussian_fit_state(X, y, coef, intercept, True, backend="numpy")
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        cov_type,
        hac_maxlags=3,
        backend="numpy",
    )
    cov = _independent_covariance(
        state.X_design,
        state.resid,
        cov_type,
        state.df_resid,
        maxlags=3,
    )
    expected_bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
    expected_stat = state.params / (expected_bse + 1e-30)
    if cov_type == "nonrobust":
        expected_p = 2.0 * stats.t.sf(np.abs(expected_stat), state.df_resid)
    else:
        expected_p = 2.0 * stats.norm.sf(np.abs(expected_stat))

    np.testing.assert_allclose(result.bse, expected_bse, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.tvalues, expected_stat, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.pvalues, expected_p, rtol=1e-9, atol=1e-12)


def test_ridge_nonrobust_preserves_average_loss_penalty_mapping():
    X, y, coef, intercept = _problem()
    state = build_gaussian_fit_state(X, y, coef, intercept, True, backend="numpy")
    alpha = 0.15
    ridge_alpha = state.normalization * alpha
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "nonrobust",
        ridge_alpha=ridge_alpha,
        ridge_penalize_intercept=False,
        backend="numpy",
    )

    XtX = state.X_design.T @ state.X_design
    penalty = np.eye(XtX.shape[0]) * ridge_alpha
    penalty[0, 0] = 0.0
    bread = np.linalg.inv(XtX + penalty)
    expected_cov = float(state.scale) * (bread @ XtX @ bread)
    expected_bse = np.sqrt(np.diag(expected_cov))
    np.testing.assert_allclose(result.bse, expected_bse, rtol=1e-10, atol=1e-12)
    assert result.metadata["ridge_alpha"] == pytest.approx(X.shape[0] * alpha)


def test_weighted_ridge_uses_weight_sum_for_penalty_mapping():
    X, y, coef, intercept = _problem()
    weights = np.linspace(0.5, 2.0, X.shape[0])
    state = build_gaussian_fit_state(
        X,
        y,
        coef,
        intercept,
        True,
        sample_weight=weights,
        backend="numpy",
    )
    alpha = 0.075
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "nonrobust",
        ridge_alpha=state.normalization * alpha,
        ridge_penalize_intercept=False,
        backend="numpy",
    )
    assert result.metadata["ridge_alpha"] == pytest.approx(np.sum(weights) * alpha)


def test_weighted_hc3_matches_explicit_weighted_sandwich_formula():
    X, y, coef, intercept = _problem()
    weights = np.linspace(0.4, 2.2, X.shape[0])
    state = build_gaussian_fit_state(
        X,
        y,
        coef,
        intercept,
        True,
        sample_weight=weights,
        backend="numpy",
    )
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "hc3",
        backend="numpy",
    )
    expected_cov = _independent_covariance(
        state.X_design, state.resid, "hc3", state.df_resid
    )
    np.testing.assert_allclose(
        result.bse,
        np.sqrt(np.maximum(np.diag(expected_cov), 0.0)),
        rtol=1e-10,
        atol=1e-12,
    )


def test_torch_cpu_all_covariances_match_numpy():
    torch = pytest.importorskip("torch")
    X, y, coef, intercept = _problem()
    np_state = build_gaussian_fit_state(X, y, coef, intercept, True, backend="numpy")
    torch_state = build_gaussian_fit_state(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        coef,
        intercept,
        True,
        backend="torch",
        device="cpu",
    )
    for cov_type in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
        ref = compute_gaussian_inference(
            np_state.X_design,
            np_state.params,
            np_state.resid,
            np_state.scale,
            np_state.df_resid,
            cov_type,
            hac_maxlags=2,
            backend="numpy",
        )
        got = compute_gaussian_inference(
            torch_state.X_design,
            torch_state.params,
            torch_state.resid,
            torch_state.scale,
            torch_state.df_resid,
            cov_type,
            hac_maxlags=2,
            backend="torch",
            device="cpu",
        )
        np.testing.assert_allclose(got.bse, ref.bse, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(got.pvalues, ref.pvalues, rtol=2e-7, atol=1e-10)
        np.testing.assert_allclose(got.conf_int, ref.conf_int, rtol=2e-7, atol=1e-10)
        assert got.metadata["numerical_backend"] == "torch"
        assert got.metadata["numerical_device"] == "cpu"


def test_torch_float32_preserves_native_dtype_and_agrees_with_float64_reference():
    torch = pytest.importorskip("torch")
    X, y, coef, intercept = _problem()
    reference_state = build_gaussian_fit_state(
        X, y, coef, intercept, True, backend="numpy"
    )
    reference = compute_gaussian_inference(
        reference_state.X_design,
        reference_state.params,
        reference_state.resid,
        reference_state.scale,
        reference_state.df_resid,
        "hc3",
        backend="numpy",
    )

    state = build_gaussian_fit_state(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        np.asarray(coef, dtype=np.float32),
        np.float32(intercept),
        True,
        backend="torch",
        device="cpu",
    )
    assert state.X_design.dtype == torch.float32
    assert state.resid.dtype == torch.float32
    assert state.params.dtype == torch.float32

    got = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "hc3",
        backend="torch",
        device="cpu",
    )
    np.testing.assert_allclose(got.bse, reference.bse, rtol=5e-4, atol=2e-5)
    np.testing.assert_allclose(got.tvalues, reference.tvalues, rtol=8e-4, atol=5e-5)
    np.testing.assert_allclose(got.pvalues, reference.pvalues, rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(got.conf_int, reference.conf_int, rtol=8e-4, atol=5e-5)
    assert got.metadata["numerical_backend"] == "torch"
    assert got.metadata["numerical_device"] == "cpu"
