"""Tests for fixed-X and model-X knockoff feature selection."""

import numpy as np
import pytest

from statgpu.feature_selection import (
    FixedXKnockoffSelector,
    KnockoffSelector,
    fixed_x_knockoff_filter,
    knockoff_filter,
    model_x_knockoff_filter,
)
from statgpu._config import cuda_available


def _make_regression_data(seed=42, n=240, p=20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:5] = np.array([2.0, -1.7, 1.4, 1.1, -0.9])
    y = X @ beta + rng.normal(scale=1.0, size=n)
    return X, y


def test_fixed_x_knockoff_basic_output_cpu():
    X, y = _make_regression_data()

    result = fixed_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="corr_diff",
        fdr_control="knockoff_plus",
        random_state=123,
        backend="numpy",
    )

    assert result.W.shape == (X.shape[1],)
    assert result.selected_features.ndim == 1
    assert result.selected_features.dtype.kind in ("i", "u")
    assert np.all(result.selected_features >= 0)
    assert np.all(result.selected_features < X.shape[1])
    assert 0.0 < result.q < 1.0
    assert 0.0 <= result.estimated_fdr <= 1.0
    assert result.method == "corr_diff"
    assert result.backend == "numpy"
    assert result.knockoff_type == "fixed_x"
    assert isinstance(result.q_trajectory, list)


def test_fixed_x_knockoff_ols_statistic_cpu():
    X, y = _make_regression_data(seed=40, n=260, p=30)

    result = fixed_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="ols_coef_diff",
        fdr_control="knockoff_plus",
        random_state=123,
        backend="numpy",
    )

    assert result.method == "ols_coef_diff"
    assert result.W.shape == (X.shape[1],)
    assert result.selected_features.ndim == 1


def test_fixed_x_knockoff_lasso_statistic_cpu():
    X, y = _make_regression_data(seed=41, n=260, p=30)

    result = fixed_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="lasso_coef_diff",
        fdr_control="knockoff_plus",
        random_state=123,
        backend="numpy",
    )

    assert result.method == "lasso_coef_diff"
    assert result.W.shape == (X.shape[1],)
    assert result.selected_features.ndim == 1


def test_knockoff_filter_dispatcher_fixed_x_matches_direct_call():
    X, y = _make_regression_data(seed=21)

    direct = fixed_x_knockoff_filter(X, y, q=0.2, random_state=7, backend="numpy")
    routed = knockoff_filter(X, y, knockoff_type="fixed_x", q=0.2, random_state=7, backend="numpy")

    assert direct.knockoff_type == routed.knockoff_type == "fixed_x"
    assert np.array_equal(direct.selected_features, routed.selected_features)
    assert np.allclose(direct.W, routed.W, rtol=1e-12, atol=1e-12)
    assert np.isclose(direct.threshold, routed.threshold)


def test_model_x_knockoff_basic_output_cpu():
    X, y = _make_regression_data(seed=22, n=160, p=24)

    result = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="corr_diff",
        fdr_control="knockoff_plus",
        random_state=9,
        backend="numpy",
    )

    assert result.W.shape == (X.shape[1],)
    assert result.selected_features.ndim == 1
    assert result.selected_features.dtype.kind in ("i", "u")
    assert np.all(result.selected_features >= 0)
    assert np.all(result.selected_features < X.shape[1])
    assert result.knockoff_type == "model_x"
    assert 0.0 <= result.estimated_fdr <= 1.0
    assert isinstance(result.q_trajectory, list)
    assert "s_value" in result.metadata
    assert float(result.metadata["s_value"]) > 0.0
    assert int(result.metadata["n_modelx_draws"]) == 3
    assert 0.0 <= float(result.metadata["covariance_shrinkage"]) <= 1.0


def test_model_x_knockoff_ols_statistic_cpu():
    X, y = _make_regression_data(seed=26, n=180, p=28)

    result = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="ols_coef_diff",
        fdr_control="knockoff_plus",
        random_state=9,
        backend="numpy",
    )

    assert result.method == "ols_coef_diff"
    assert result.knockoff_type == "model_x"
    assert result.W.shape == (X.shape[1],)
    assert int(result.metadata["n_modelx_draws"]) == 5


def test_model_x_knockoff_lasso_statistic_cpu():
    X, y = _make_regression_data(seed=28, n=180, p=28)

    result = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="lasso_coef_diff",
        fdr_control="knockoff_plus",
        random_state=9,
        backend="numpy",
    )

    assert result.method == "lasso_coef_diff"
    assert result.knockoff_type == "model_x"
    assert result.W.shape == (X.shape[1],)
    assert int(result.metadata["n_modelx_draws"]) == 5


def test_knockoff_filter_dispatcher_model_x_matches_direct_call():
    X, y = _make_regression_data(seed=23, n=170, p=26)

    direct = model_x_knockoff_filter(X, y, q=0.2, random_state=9, backend="numpy")
    routed = knockoff_filter(X, y, knockoff_type="model_x", q=0.2, random_state=9, backend="numpy")

    assert direct.knockoff_type == routed.knockoff_type == "model_x"
    assert np.array_equal(direct.selected_features, routed.selected_features)
    assert np.allclose(direct.W, routed.W, rtol=1e-12, atol=1e-12)
    assert np.isclose(direct.threshold, routed.threshold)


def test_model_x_with_provided_xk_ignores_seed_and_uses_single_draw():
    X, y = _make_regression_data(seed=230, n=160, p=24)
    rng = np.random.default_rng(991)
    Xk = X + 0.05 * rng.normal(size=X.shape)

    r1 = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="corr_diff",
        random_state=1,
        backend="numpy",
        Xk=Xk,
    )
    r2 = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="corr_diff",
        random_state=999,
        backend="numpy",
        Xk=Xk,
    )

    assert np.allclose(r1.W, r2.W, rtol=1e-12, atol=1e-12)
    assert np.array_equal(r1.selected_features, r2.selected_features)
    assert int(r1.metadata["n_modelx_draws"]) == 1
    assert r1.metadata["xk_source"] == "provided"


def test_model_x_knockpy_compat_mode_with_provided_xk_runs():
    pytest.importorskip("sklearn")
    X, y = _make_regression_data(seed=231, n=170, p=26)
    rng = np.random.default_rng(992)
    Xk = X + 0.05 * rng.normal(size=X.shape)

    result = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        method="lasso_coef_diff",
        random_state=123,
        backend="numpy",
        Xk=Xk,
        compat_mode="knockpy",
        lasso_cv_impl="sklearn",
    )

    assert result.knockoff_type == "model_x"
    assert result.backend == "numpy"
    assert result.metadata["compat_mode"] == "knockpy"
    assert result.metadata["lasso_cv_impl"] == "sklearn"
    assert result.metadata["xk_source"] == "provided"
    assert int(result.metadata["n_modelx_draws"]) == 1


@pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA not available",
)
def test_model_x_knockpy_compat_mode_with_provided_xk_statgpu_uses_cupy_backend():
    cp = pytest.importorskip("cupy")
    X, y = _make_regression_data(seed=233, n=170, p=26)
    rng = np.random.default_rng(994)
    Xk = X + 0.05 * rng.normal(size=X.shape)

    result = model_x_knockoff_filter(
        cp.asarray(X),
        cp.asarray(y),
        q=0.2,
        method="lasso_coef_diff",
        random_state=123,
        backend="cupy",
        Xk=cp.asarray(Xk),
        compat_mode="knockpy",
        lasso_cv_impl="statgpu",
    )

    assert result.knockoff_type == "model_x"
    assert result.backend == "cupy"
    assert result.metadata["compat_mode"] == "knockpy"
    assert result.metadata["lasso_cv_impl"] == "statgpu"
    assert result.metadata["xk_source"] == "provided"
    assert int(result.metadata["n_modelx_draws"]) == 1


def test_knockoff_filter_dispatcher_model_x_with_provided_xk_matches_direct_call():
    X, y = _make_regression_data(seed=232, n=165, p=24)
    rng = np.random.default_rng(993)
    Xk = X + 0.05 * rng.normal(size=X.shape)

    direct = model_x_knockoff_filter(
        X,
        y,
        q=0.2,
        random_state=9,
        backend="numpy",
        Xk=Xk,
        method="corr_diff",
    )
    routed = knockoff_filter(
        X,
        y,
        knockoff_type="model_x",
        q=0.2,
        random_state=9,
        backend="numpy",
        Xk=Xk,
        method="corr_diff",
    )

    assert np.array_equal(direct.selected_features, routed.selected_features)
    assert np.allclose(direct.W, routed.W, rtol=1e-12, atol=1e-12)
    assert np.isclose(direct.threshold, routed.threshold)


def test_model_x_knockoff_reproducible_with_seed_cpu():
    X, y = _make_regression_data(seed=24, n=165, p=22)

    r1 = model_x_knockoff_filter(X, y, q=0.2, random_state=99, backend="numpy")
    r2 = model_x_knockoff_filter(X, y, q=0.2, random_state=99, backend="numpy")

    assert np.array_equal(r1.selected_features, r2.selected_features)
    assert np.allclose(r1.W, r2.W, rtol=1e-12, atol=1e-12)
    assert np.isclose(r1.threshold, r2.threshold)


def test_fixed_x_knockoff_reproducible_with_seed_cpu():
    X, y = _make_regression_data(seed=7)

    r1 = fixed_x_knockoff_filter(X, y, q=0.2, random_state=99, backend="numpy")
    r2 = fixed_x_knockoff_filter(X, y, q=0.2, random_state=99, backend="numpy")

    assert np.array_equal(r1.selected_features, r2.selected_features)
    assert np.allclose(r1.W, r2.W, rtol=1e-12, atol=1e-12)
    assert np.isclose(r1.threshold, r2.threshold)


def test_fixed_x_knockoff_invalid_shape_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)

    with pytest.raises(ValueError):
        fixed_x_knockoff_filter(X, y, backend="numpy")


def test_knockoff_invalid_method_raises():
    X, y = _make_regression_data(seed=27)

    with pytest.raises(ValueError):
        fixed_x_knockoff_filter(X, y, method="not_a_method", backend="numpy")

    with pytest.raises(ValueError):
        model_x_knockoff_filter(X, y, method="not_a_method", backend="numpy")


def test_fixed_x_selector_fit_transform_and_support():
    X, y = _make_regression_data(seed=123)
    selector = FixedXKnockoffSelector(q=0.2, random_state=123, backend="numpy")

    Xt = selector.fit_transform(X, y)
    mask = selector.get_support()

    assert mask.shape == (X.shape[1],)
    assert mask.dtype == bool
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == int(np.sum(mask))
    assert np.array_equal(np.where(mask)[0], selector.selected_features_)


def test_unified_knockoff_selector_fixed_x_fit_transform_and_support():
    X, y = _make_regression_data(seed=124)
    selector = KnockoffSelector(knockoff_type="fixed_x", q=0.2, random_state=123, backend="numpy")

    Xt = selector.fit_transform(X, y)
    mask = selector.get_support()

    assert mask.shape == (X.shape[1],)
    assert mask.dtype == bool
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == int(np.sum(mask))
    assert selector.result_.knockoff_type == "fixed_x"


def test_unified_knockoff_selector_model_x_fit_transform_and_support():
    X, y = _make_regression_data(seed=125, n=180, p=24)
    selector = KnockoffSelector(knockoff_type="model_x", q=0.2, random_state=123, backend="numpy")

    Xt = selector.fit_transform(X, y)
    mask = selector.get_support()

    assert mask.shape == (X.shape[1],)
    assert mask.dtype == bool
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == int(np.sum(mask))
    assert selector.result_.knockoff_type == "model_x"


@pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA not available",
)
def test_fixed_x_knockoff_gpu_reproducible_with_seed():
    cp = pytest.importorskip("cupy")
    X, y = _make_regression_data(seed=11)

    X_cp = cp.asarray(X)
    y_cp = cp.asarray(y)

    r1 = fixed_x_knockoff_filter(X_cp, y_cp, q=0.2, random_state=777, backend="cupy")
    r2 = fixed_x_knockoff_filter(X_cp, y_cp, q=0.2, random_state=777, backend="cupy")

    assert np.array_equal(r1.selected_features, r2.selected_features)
    assert np.allclose(r1.W, r2.W, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA not available",
)
def test_model_x_knockoff_lasso_gpu_runs():
    cp = pytest.importorskip("cupy")
    X, y = _make_regression_data(seed=12, n=180, p=24)

    X_cp = cp.asarray(X)
    y_cp = cp.asarray(y)

    result = model_x_knockoff_filter(
        X_cp,
        y_cp,
        q=0.2,
        method="lasso_coef_diff",
        random_state=777,
        backend="cupy",
    )

    assert result.method == "lasso_coef_diff"
    assert result.backend == "cupy"
    assert result.knockoff_type == "model_x"
    assert result.W.shape == (X.shape[1],)
