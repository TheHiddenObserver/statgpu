"""Tests for nonparametric kernel regression utilities."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu._config import cuda_available
from statgpu.nonparametric import (
    KernelRegression,
    KernelRegressionRegressor,
    fit_kernel_regression,
    kernel_regression_predict,
)


def _bandwidth_factor_from_abs(samples_2d: np.ndarray, bandwidth_abs: float) -> float:
    sd = np.std(np.asarray(samples_2d, dtype=np.float64), axis=0, ddof=1)
    scale = float(np.mean(sd))
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = 1.0
    return float(bandwidth_abs / scale)


def test_kernel_regression_1d_sine_fit_quality():
    rng = np.random.default_rng(20260414)
    x = rng.uniform(-3.0, 3.0, size=420)
    y = np.sin(x) + rng.normal(scale=0.12, size=x.shape[0])

    model = fit_kernel_regression(x, y, bandwidth="scott", kernel="gaussian", backend="numpy")
    grid = np.linspace(-2.5, 2.5, 121)
    pred = np.asarray(model.predict(grid), dtype=np.float64)

    rmse = float(np.sqrt(np.mean((pred - np.sin(grid)) ** 2)))

    assert isinstance(model, KernelRegression)
    assert pred.shape == grid.shape
    assert rmse < 0.25


def test_kernel_regression_one_shot_matches_model_prediction():
    rng = np.random.default_rng(20260414)
    x = rng.normal(size=240)
    y = 0.8 * x * x - 0.25 * x + rng.normal(scale=0.1, size=x.shape[0])
    points = np.linspace(-2.0, 2.0, 73)

    model = fit_kernel_regression(
        x,
        y,
        bandwidth="silverman",
        kernel="epanechnikov",
        backend="numpy",
    )
    pred_model = model.predict(points)
    pred_one_shot = kernel_regression_predict(
        x,
        y,
        points,
        bandwidth="silverman",
        kernel="epanechnikov",
        backend="numpy",
    )

    assert np.allclose(pred_model, pred_one_shot, atol=1e-12, rtol=1e-12)


def test_local_linear_one_shot_matches_model_prediction():
    rng = np.random.default_rng(20260414)
    x = rng.normal(size=260)
    y = np.sin(x) + 0.15 * x
    points = np.linspace(-2.2, 2.2, 67)

    model = fit_kernel_regression(
        x,
        y,
        bandwidth="scott",
        kernel="gaussian",
        regression="local_linear",
        backend="numpy",
    )
    pred_model = model.predict(points)
    pred_one_shot = kernel_regression_predict(
        x,
        y,
        points,
        bandwidth="scott",
        kernel="gaussian",
        regression="local_linear",
        backend="numpy",
    )

    assert np.allclose(pred_model, pred_one_shot, atol=1e-12, rtol=1e-12)


def test_local_linear_improves_boundary_bias_over_nw_on_quadratic():
    x = np.linspace(0.0, 1.0, 300)
    y = x * x
    points = np.array([0.01, 0.03, 0.97, 0.99], dtype=np.float64)

    model_nw = fit_kernel_regression(
        x,
        y,
        bandwidth=0.12,
        kernel="gaussian",
        regression="nw",
        backend="numpy",
    )
    model_ll = fit_kernel_regression(
        x,
        y,
        bandwidth=0.12,
        kernel="gaussian",
        regression="local_linear",
        backend="numpy",
    )

    pred_nw = np.asarray(model_nw.predict(points), dtype=np.float64)
    pred_ll = np.asarray(model_ll.predict(points), dtype=np.float64)
    truth = points * points

    err_nw = float(np.mean(np.abs(pred_nw - truth)))
    err_ll = float(np.mean(np.abs(pred_ll - truth)))

    assert err_ll <= err_nw


def test_kernel_regression_multi_output_and_metadata():
    rng = np.random.default_rng(20260414)
    x = rng.uniform(-2.0, 2.0, size=280)
    y = np.column_stack([np.sin(x), np.cos(x)])

    model = fit_kernel_regression(x, y, bandwidth="scott", kernel="triweight", backend="numpy")
    pred = np.asarray(model(np.array([-0.4, 0.0, 0.9])), dtype=np.float64)
    meta = model.to_numpy_metadata()

    assert pred.shape == (3, 2)
    assert meta["n_features"] == 1
    assert meta["n_targets"] == 2
    assert meta["kernel"] == "triweight"
    assert "bandwidth_selection" in meta
    assert isinstance(meta["bandwidth_selection"], dict)


def test_compact_kernel_far_points_fall_back_to_weighted_mean():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    model = fit_kernel_regression(x, y, bandwidth=0.15, kernel="rectangular", backend="numpy")
    pred_far = np.asarray(model.predict(np.array([-10.0, 10.0])), dtype=np.float64)

    expected = np.mean(y)
    assert np.allclose(pred_far, expected, atol=1e-10, rtol=0.0)


def test_kernel_regression_weighted_r_selector_works_for_1d():
    rng = np.random.default_rng(20260414)
    x = rng.normal(size=220)
    y = np.sin(x) + 0.2 * x
    w = rng.uniform(0.3, 2.1, size=x.shape[0])
    points = np.linspace(-2.5, 2.5, 51)

    model = fit_kernel_regression(
        x,
        y,
        bandwidth="sj",
        weights=w,
        kernel="gaussian",
        regression="local_linear",
        backend="numpy",
    )
    pred = np.asarray(model.predict(points), dtype=np.float64)

    assert np.all(np.isfinite(pred))
    assert model.bandwidth_factor_ > 0.0
    assert model.bandwidth_info_.weighted is True


def test_kernel_regression_multivariate_selector_rules_work():
    rng = np.random.default_rng(20260414)
    x = rng.normal(size=(240, 3))
    x[:, 1] = 0.4 * x[:, 0] + x[:, 1]
    y = 0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.2 * x[:, 2] + rng.normal(scale=0.1, size=x.shape[0])
    w = rng.uniform(0.2, 1.6, size=x.shape[0])
    points = rng.normal(size=(36, 3))

    for bw in ("nrd", "sj"):
        model = fit_kernel_regression(
            x,
            y,
            bandwidth=bw,
            weights=w,
            kernel="gaussian",
            backend="numpy",
        )
        pred = np.asarray(model.predict(points), dtype=np.float64)

        assert np.all(np.isfinite(pred))
        assert model.bandwidth_factor_ > 0.0
        assert model.bandwidth_info_.selector_dimension == 1


def test_kernel_regression_cv_bandwidth_rules_work_and_expose_diagnostics():
    rng = np.random.default_rng(20260414)
    x = rng.uniform(-2.5, 2.5, size=180)
    y = np.sin(1.2 * x) + 0.15 * x + rng.normal(scale=0.08, size=x.shape[0])
    points = np.linspace(-2.0, 2.0, 61)

    model_nw = fit_kernel_regression(
        x,
        y,
        bandwidth="cv",
        regression="nw",
        kernel="gaussian",
        backend="numpy",
    )
    model_ll = fit_kernel_regression(
        x,
        y,
        bandwidth="cv-ll",
        regression="nw",
        kernel="gaussian",
        backend="numpy",
    )
    model_alias = fit_kernel_regression(
        x,
        y,
        bandwidth="cv_ls",
        regression="local_linear",
        kernel="gaussian",
        backend="numpy",
    )

    pred_nw = np.asarray(model_nw.predict(points), dtype=np.float64)
    pred_ll = np.asarray(model_ll.predict(points), dtype=np.float64)
    pred_alias = np.asarray(model_alias.predict(points), dtype=np.float64)

    assert np.all(np.isfinite(pred_nw))
    assert np.all(np.isfinite(pred_ll))
    assert np.all(np.isfinite(pred_alias))

    assert model_nw.bandwidth_factor_ > 0.0
    assert model_ll.bandwidth_factor_ > 0.0
    assert model_alias.bandwidth_factor_ > 0.0

    assert model_nw.bandwidth_info_.details["rule"] == "regression_cv"
    assert model_nw.bandwidth_info_.details["cv_regression_mode"] == "nw"
    assert model_ll.bandwidth_info_.details["cv_regression_mode"] == "local_linear"
    assert model_alias.bandwidth_info_.details["cv_regression_mode"] == "local_linear"


def test_kernel_regression_estimator_fit_predict_and_score():
    rng = np.random.default_rng(20260415)
    x = rng.uniform(-2.5, 2.5, size=260)
    y = np.sin(x) + 0.1 * x + rng.normal(scale=0.08, size=x.shape[0])
    points = np.linspace(-2.2, 2.2, 61)

    est = KernelRegressionRegressor(
        bandwidth="scott",
        kernel="gaussian",
        regression="local_linear",
        backend="numpy",
    )
    est.fit(x, y)
    pred = np.asarray(est.predict(points), dtype=np.float64)

    assert pred.shape == points.shape
    assert np.all(np.isfinite(pred))
    assert np.isfinite(est.score(points, np.sin(points) + 0.1 * points))


@pytest.mark.parametrize(
    "regression,reg_type,atol",
    [
        ("nw", "lc", 1e-10),
        ("local_linear", "ll", 1e-9),
    ],
)
def test_kernel_regression_multidim_diagonal_metric_matches_statsmodels(
    regression: str,
    reg_type: str,
    atol: float,
):
    sm_np = pytest.importorskip("statsmodels.nonparametric.kernel_regression")
    KernelReg = sm_np.KernelReg

    rng = np.random.default_rng(20260415)
    x = rng.normal(size=(240, 3))
    x[:, 1] = 0.3 * x[:, 0] + x[:, 1]
    y = 0.9 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2] + 0.2 * np.sin(x[:, 0])
    points = rng.normal(size=(64, 3))

    bw_abs = 0.45
    bw_vec = np.full(x.shape[1], bw_abs, dtype=np.float64)
    factor = _bandwidth_factor_from_abs(x, bw_abs)

    sm_model = KernelReg(
        endog=y,
        exog=x,
        var_type="c" * x.shape[1],
        reg_type=reg_type,
        bw=bw_vec,
    )
    sm_pred, _ = sm_model.fit(points)
    sm_pred = np.asarray(sm_pred, dtype=np.float64)

    model = fit_kernel_regression(
        x,
        y,
        bandwidth=factor,
        bandwidth_per_feature=bw_vec,
        kernel="gaussian",
        regression=regression,
        kernel_metric="diagonal",
        backend="numpy",
    )
    pred = np.asarray(model.predict(points), dtype=np.float64)

    assert np.allclose(pred, sm_pred, atol=atol, rtol=1e-9)


def test_kernel_regression_diagonal_metric_bandwidth_per_feature_validation():
    rng = np.random.default_rng(20260415)
    x = rng.normal(size=(100, 2))
    y = 0.5 * x[:, 0] - 0.2 * x[:, 1]

    with pytest.raises(ValueError):
        fit_kernel_regression(
            x,
            y,
            kernel_metric="full",
            bandwidth_per_feature=np.array([0.4, 0.4]),
            backend="numpy",
        )

    with pytest.raises(ValueError):
        fit_kernel_regression(
            x,
            y,
            kernel_metric="diagonal",
            bandwidth_per_feature=np.array([0.4, 0.4, 0.4]),
            backend="numpy",
        )

    with pytest.raises(ValueError):
        fit_kernel_regression(
            x,
            y,
            kernel_metric="diagonal",
            bandwidth_per_feature=np.array([0.4, 0.0]),
            backend="numpy",
        )


def test_kernel_regression_invalid_inputs_raise():
    rng = np.random.default_rng(20260414)
    x = rng.normal(size=120)
    y = rng.normal(size=120)
    x2 = rng.normal(size=(120, 2))

    with pytest.raises(ValueError):
        fit_kernel_regression(x, y[:100], backend="numpy")

    with pytest.raises(ValueError):
        fit_kernel_regression(x2, y, kernel="cosine", backend="numpy")

    with pytest.raises(ValueError):
        fit_kernel_regression(x, y, regression="foo", backend="numpy")

    model = fit_kernel_regression(x, y, backend="numpy")
    with pytest.raises(ValueError):
        model.predict(np.ones((5, 2)))

    with pytest.raises(ValueError):
        model.predict(np.array([0.0]), min_effective_weight=0.0)


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
def test_kernel_regression_cupy_matches_numpy():
    import cupy as cp

    rng = np.random.default_rng(20260414)
    x_np = rng.normal(size=(260, 2))
    x_np[:, 1] = 0.5 * x_np[:, 0] + x_np[:, 1]
    y_np = (1.5 * x_np[:, 0] - 0.7 * x_np[:, 1]) + rng.normal(scale=0.05, size=x_np.shape[0])
    p_np = rng.normal(size=(80, 2))

    model_np = fit_kernel_regression(x_np, y_np, bandwidth="scott", backend="numpy")
    pred_np = np.asarray(model_np.predict(p_np), dtype=np.float64)

    model_cp = fit_kernel_regression(cp.asarray(x_np), cp.asarray(y_np), bandwidth="scott", backend="cupy")
    pred_cp = cp.asnumpy(model_cp.predict(cp.asarray(p_np)))

    assert np.max(np.abs(pred_np - pred_cp)) < 5e-6

    bw_vec = np.array([0.42, 0.48], dtype=np.float64)
    factor_diag = _bandwidth_factor_from_abs(x_np, float(np.mean(bw_vec)))

    model_np_diag = fit_kernel_regression(
        x_np,
        y_np,
        bandwidth=factor_diag,
        bandwidth_per_feature=bw_vec,
        kernel_metric="diagonal",
        backend="numpy",
    )
    pred_np_diag = np.asarray(model_np_diag.predict(p_np), dtype=np.float64)

    model_cp_diag = fit_kernel_regression(
        cp.asarray(x_np),
        cp.asarray(y_np),
        bandwidth=factor_diag,
        bandwidth_per_feature=bw_vec,
        kernel_metric="diagonal",
        backend="cupy",
    )
    pred_cp_diag = cp.asnumpy(model_cp_diag.predict(cp.asarray(p_np)))

    assert np.max(np.abs(pred_np_diag - pred_cp_diag)) < 5e-6
