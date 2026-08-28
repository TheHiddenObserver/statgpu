"""Regression contracts for issue #127 backend-native Gaussian inference."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
)


def _simple_state(dtype=np.float64):
    X = np.asarray(
        [
            [-2.0, 0.5],
            [-1.0, 1.0],
            [0.0, 1.5],
            [1.0, 2.0],
            [2.0, 2.5],
            [3.0, 3.0],
        ],
        dtype=dtype,
    )
    coef = np.asarray([0.75, -0.4], dtype=dtype)
    intercept = np.asarray(1.2, dtype=dtype)
    noise = np.asarray(
        [0.2, -0.1, 0.15, -0.25, 0.05, -0.05], dtype=dtype
    )
    y = intercept + X @ coef + noise
    return X, y, coef, intercept


def test_numpy_fit_state_keeps_legacy_float64_numerics():
    X, y, coef, intercept = _simple_state(np.float32)
    state = build_gaussian_fit_state(
        X, y, coef, intercept, True, backend="numpy"
    )

    assert state.backend == "numpy"
    assert state.device == "cpu"
    assert state.X_design.dtype == np.float64
    assert state.y.dtype == np.float64
    assert state.resid.dtype == np.float64
    assert state.params.dtype == np.float64


def test_numpy_nonrobust_matches_closed_form_and_scipy():
    X, y, coef, intercept = _simple_state()
    state = build_gaussian_fit_state(
        X, y, coef, intercept, True, backend="numpy"
    )
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "nonrobust",
        backend="numpy",
    )

    XtX_inv = np.linalg.inv(state.X_design.T @ state.X_design)
    expected_bse = np.sqrt(np.diag(float(state.scale) * XtX_inv))
    expected_t = state.params / expected_bse
    expected_p = 2.0 * stats.t.sf(np.abs(expected_t), state.df_resid)
    critical = stats.t.ppf(0.975, state.df_resid)
    expected_ci = np.column_stack(
        [
            state.params - critical * expected_bse,
            state.params + critical * expected_bse,
        ]
    )

    np.testing.assert_allclose(
        result.bse, expected_bse, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        result.tvalues, expected_t, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        result.pvalues, expected_p, rtol=1e-10, atol=1e-13
    )
    np.testing.assert_allclose(
        result.conf_int, expected_ci, rtol=1e-10, atol=1e-12
    )
    assert result.metadata["numerical_backend"] == "numpy"
    assert result.metadata["reporting_backend"] == "numpy"
    assert result.metadata["reporting_boundary"] == "post_numerical_inference"


def test_numpy_hc3_matches_independent_sandwich_formula():
    X, y, coef, intercept = _simple_state()
    state = build_gaussian_fit_state(
        X, y, coef, intercept, True, backend="numpy"
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

    design = state.X_design
    bread = np.linalg.inv(design.T @ design)
    leverage = np.sum(design * (design @ bread), axis=1)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)
    omega = state.resid**2 / np.maximum((1.0 - leverage) ** 2, 1e-12)
    meat = design.T @ (design * omega[:, None])
    cov = bread @ meat @ bread
    expected_bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
    expected_z = state.params / (expected_bse + 1e-30)
    expected_p = 2.0 * stats.norm.sf(np.abs(expected_z))

    np.testing.assert_allclose(
        result.bse, expected_bse, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        result.tvalues, expected_z, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        result.pvalues, expected_p, rtol=1e-11, atol=1e-13
    )


def test_weighted_fit_state_uses_sqrt_weight_and_weight_sum():
    X, y, coef, intercept = _simple_state()
    weights = np.asarray([0.5, 1.0, 1.5, 2.0, 0.75, 1.25])
    state = build_gaussian_fit_state(
        X,
        y,
        coef,
        intercept,
        True,
        sample_weight=weights,
        backend="numpy",
    )

    sqrt_w = np.sqrt(weights)
    expected_design = np.column_stack([sqrt_w, X * sqrt_w[:, None]])
    expected_resid = (y - intercept - X @ coef) * sqrt_w
    np.testing.assert_allclose(state.X_design, expected_design)
    np.testing.assert_allclose(state.resid, expected_resid)
    assert state.normalization == pytest.approx(float(np.sum(weights)))


def test_multitarget_shape_and_targetwise_equivalence():
    X, y, coef, intercept = _simple_state()
    coef2 = np.column_stack([coef, np.asarray([-0.2, 0.6])])
    intercept2 = np.asarray([intercept, -0.3])
    y2 = np.column_stack(
        [
            y,
            intercept2[1]
            + X @ coef2[:, 1]
            + np.asarray([0.1, 0.0, -0.1, 0.2, -0.2, 0.05]),
        ]
    )
    state = build_gaussian_fit_state(
        X, y2, coef2, intercept2, True, backend="numpy"
    )
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "nonrobust",
        backend="numpy",
    )

    assert result.bse.shape == state.params.shape
    assert result.conf_int.shape == (state.params.shape[0], 2, 2)
    for target in range(2):
        one = compute_gaussian_inference(
            state.X_design,
            state.params[:, target],
            state.resid[:, target],
            np.asarray(state.scale).reshape(-1)[target],
            state.df_resid,
            "nonrobust",
            backend="numpy",
        )
        np.testing.assert_allclose(result.bse[:, target], one.bse)
        np.testing.assert_allclose(result.pvalues[:, target], one.pvalues)
        np.testing.assert_allclose(
            result.conf_int[:, target, :], one.conf_int
        )


def test_rank_deficient_nonrobust_uses_pseudoinverse_semantics():
    x = np.arange(1.0, 7.0)
    design = np.column_stack([np.ones_like(x), x, x])
    params = np.asarray([0.5, 0.25, 0.25])
    resid = np.asarray([0.2, -0.1, 0.05, -0.15, 0.1, -0.1])
    df_resid = design.shape[0] - design.shape[1]
    scale = np.sum(resid**2) / df_resid

    result = compute_gaussian_inference(
        design,
        params,
        resid,
        scale,
        df_resid,
        "nonrobust",
        backend="numpy",
    )
    expected_cov = scale * np.linalg.pinv(design.T @ design)
    expected_bse = np.sqrt(np.diag(expected_cov))
    np.testing.assert_allclose(result.bse, expected_bse, rtol=1e-10, atol=1e-12)


def test_student_t_df2_extreme_tail_does_not_cancel_to_zero():
    # df=2 has a representable two-sided tail near 1e-308 at t=1e154.
    design = np.ones((3, 1), dtype=np.float64)
    params = np.asarray([1.0e154])
    resid = np.asarray([1.0, -1.0, 0.0])
    result = compute_gaussian_inference(
        design,
        params,
        resid,
        1.0,
        2,
        "nonrobust",
        backend="numpy",
    )
    assert np.isfinite(result.pvalues[0])
    assert result.pvalues[0] > 0.0


def test_torch_cpu_distribution_receives_native_statistic(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.linear_model._gaussian_inference as gi

    observed = {}

    class _FakeT:
        def two_sided_pvalue(self, statistic, df):
            assert isinstance(statistic, torch.Tensor)
            assert statistic.device.type == "cpu"
            observed["pvalue_native"] = True
            return torch.full_like(statistic, 0.125)

        def two_sided_critical_value(self, alpha, df):
            observed["critical"] = (float(alpha), int(df))
            return torch.tensor(2.5, dtype=torch.float64)

    def _fake_distribution(name, backend="numpy", device=None):
        assert name == "t"
        assert backend == "torch"
        assert str(device) == "cpu"
        return _FakeT()

    monkeypatch.setattr(gi, "get_distribution", _fake_distribution)

    X, y, coef, intercept = _simple_state()
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    state = build_gaussian_fit_state(
        X_t, y_t, coef, intercept, True, backend="torch", device="cpu"
    )
    result = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "nonrobust",
        backend="torch",
        device="cpu",
    )

    assert observed["pvalue_native"] is True
    assert observed["critical"] == (0.05, state.df_resid)
    assert result.metadata["numerical_backend"] == "torch"
    assert result.metadata["numerical_device"] == "cpu"
    assert isinstance(result.pvalues, np.ndarray)
    np.testing.assert_allclose(result.pvalues, 0.125)


def test_ridgecv_final_refit_inference_does_not_change_selection():
    from statgpu.linear_model import RidgeCV

    rng = np.random.default_rng(20260828)
    X = rng.normal(size=(80, 5))
    beta = np.asarray([1.2, -0.7, 0.3, 0.0, 0.5])
    y = 0.4 + X @ beta + 0.1 * rng.normal(size=X.shape[0])
    alphas = np.asarray([0.01, 0.1, 1.0])

    common = dict(
        alphas=alphas,
        cv=4,
        fit_intercept=True,
        device="cpu",
        random_state=20260828,
    )
    no_inference = RidgeCV(compute_inference=False, **common).fit(X, y)
    with_inference = RidgeCV(compute_inference=True, **common).fit(X, y)

    assert with_inference.alpha_ == no_inference.alpha_
    np.testing.assert_allclose(
        with_inference.coef_, no_inference.coef_, rtol=1e-12, atol=1e-12
    )
    assert with_inference.intercept_ == pytest.approx(
        no_inference.intercept_, rel=1e-12, abs=1e-12
    )
    assert with_inference.estimator_._inference_result is not None
    assert (
        with_inference.estimator_._inference_result.metadata["numerical_backend"]
        == "numpy"
    )
