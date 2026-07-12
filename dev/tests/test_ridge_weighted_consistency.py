import numpy as np
import pytest

from statgpu import Ridge
from statgpu.linear_model.cv._ridge_cv import _default_ridge_alpha_grid
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


def _weighted_closed_form(X, y, w, alpha, fit_intercept=True):
    normalizer = float(np.sum(w))
    if fit_intercept:
        x_mean = np.sum(X * w[:, None], axis=0) / normalizer
        y_mean = float(np.sum(y * w) / normalizer)
    else:
        x_mean = np.zeros(X.shape[1])
        y_mean = 0.0
    Xc = X - x_mean
    yc = y - y_mean
    XtWX = (Xc * w[:, None]).T @ Xc
    XtWy = (Xc * w[:, None]).T @ yc
    coef = np.linalg.solve(XtWX + normalizer * alpha * np.eye(X.shape[1]), XtWy)
    return coef, float(y_mean - x_mean @ coef) if fit_intercept else 0.0


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_weighted_ridge_exact_matches_average_loss_and_weight_rescaling(fit_intercept):
    rng = np.random.default_rng(1201)
    X = rng.normal(size=(240, 8))
    y = X @ rng.normal(size=8) + 0.6 + rng.normal(scale=0.4, size=240)
    w = rng.uniform(0.1, 3.0, size=240)
    alpha = 0.19

    expected_coef, expected_intercept = _weighted_closed_form(X, y, w, alpha, fit_intercept)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, device="cpu", compute_inference=False).fit(X, y, sample_weight=w)
    scaled = Ridge(alpha=alpha, fit_intercept=fit_intercept, device="cpu", compute_inference=False).fit(X, y, sample_weight=7.3 * w)

    np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(model.intercept_, expected_intercept, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(scaled.coef_, model.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(scaled.intercept_, model.intercept_, rtol=1e-11, atol=1e-11)


def test_weighted_ridge_formula_and_generic_exact_match_wrapper():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1202)
    X = rng.normal(size=(180, 4))
    y = 1.1 + X @ np.array([0.7, -0.3, 0.9, 0.2]) + rng.normal(scale=0.2, size=180)
    w = rng.uniform(0.2, 2.5, size=180)
    alpha = 0.11
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    frame["y"] = y

    direct = Ridge(alpha=alpha, compute_inference=False, device="cpu").fit(X, y, sample_weight=w)
    formula = Ridge(alpha=alpha, compute_inference=False, device="cpu").fit(
        formula="y ~ x1 + x2 + x3 + x4", data=frame, sample_weight=w
    )
    generic = PenalizedLinearRegression(
        penalty="l2", alpha=alpha, solver="exact", fit_intercept=True,
        compute_inference=False, device="cpu",
    ).fit(X, y, sample_weight=w)

    for other in (formula, generic):
        np.testing.assert_allclose(other.coef_, direct.coef_, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(other.intercept_, direct.intercept_, rtol=1e-10, atol=1e-10)


def test_weighted_ridge_fista_matches_exact_objective():
    rng = np.random.default_rng(1203)
    X = rng.normal(size=(260, 6))
    y = -0.8 + X @ rng.normal(size=6) + rng.normal(scale=0.3, size=260)
    w = rng.uniform(0.05, 4.0, size=260)
    alpha = 0.07

    exact = Ridge(alpha=alpha, solver="exact", compute_inference=False, device="cpu").fit(X, y, sample_weight=w)
    fista = Ridge(
        alpha=alpha, solver="fista", max_iter=20000, tol=1e-12,
        compute_inference=False, device="cpu",
    ).fit(X, y, sample_weight=w)

    np.testing.assert_allclose(fista.coef_, exact.coef_, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(fista.intercept_, exact.intercept_, rtol=2e-7, atol=2e-8)


def test_weighted_ridge_inference_uses_weighted_intercept_column():
    rng = np.random.default_rng(1204)
    n, p = 320, 5
    X = rng.normal(size=(n, p))
    y = 0.9 + X @ rng.normal(size=p) + rng.normal(scale=0.5, size=n)
    w = rng.uniform(0.1, 2.7, size=n)
    alpha = 0.09
    model = Ridge(alpha=alpha, compute_inference=True, device="cpu").fit(X, y, sample_weight=w)

    D = np.column_stack([np.ones(n), X])
    params = np.concatenate([[model.intercept_], model.coef_])
    resid = y - D @ params
    XtWX = D.T @ (D * w[:, None])
    penalty = np.diag(np.r_[0.0, np.repeat(np.sum(w) * alpha, p)])
    bread_inv = np.linalg.inv(XtWX + penalty)
    scale = float(np.sum(w * resid ** 2) / (n - p - 1))
    cov = scale * (bread_inv @ XtWX @ bread_inv)

    np.testing.assert_allclose(model._bse, np.sqrt(np.diag(cov)), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model._X_design[:, 0], np.sqrt(w), rtol=0, atol=0)
    np.testing.assert_allclose(model._resid, np.sqrt(w) * resid, rtol=1e-12, atol=1e-12)


def test_weighted_default_alpha_grid_uses_average_loss_scale():
    rng = np.random.default_rng(1205)
    X = rng.normal(size=(140, 4))
    y = X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.3, size=140)
    w = np.linspace(0.1, 3.0, 140)
    grid = _default_ridge_alpha_grid(X, y, n_alphas=7, sample_weight=w)
    scaled = _default_ridge_alpha_grid(X, y, n_alphas=7, sample_weight=9.0 * w)
    np.testing.assert_allclose(grid, scaled, rtol=1e-12, atol=1e-12)


def test_formula_missing_rows_aligns_full_length_sample_weights():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1206)
    n = 150
    X = rng.normal(size=(n, 3))
    y = 0.4 + X @ np.array([0.8, -0.6, 0.3]) + rng.normal(scale=0.25, size=n)
    w = rng.uniform(0.2, 3.0, size=n)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["y"] = y
    frame.loc[[4, 31, 92], "x2"] = np.nan
    frame.loc[[17, 108], "y"] = np.nan
    keep = frame[["y", "x1", "x2", "x3"]].notna().all(axis=1).to_numpy()

    formula = Ridge(alpha=0.13, compute_inference=False, device="cpu").fit(
        formula="y ~ x1 + x2 + x3", data=frame, sample_weight=w
    )
    direct = Ridge(alpha=0.13, compute_inference=False, device="cpu").fit(
        X[keep], y[keep], sample_weight=w[keep]
    )

    np.testing.assert_allclose(formula.coef_, direct.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(formula.intercept_, direct.intercept_, rtol=1e-11, atol=1e-11)


def test_ridgecv_is_invariant_to_global_weight_rescaling():
    from statgpu.linear_model import RidgeCV

    rng = np.random.default_rng(1207)
    X = rng.normal(size=(180, 6))
    y = 0.3 + X @ rng.normal(size=6) + rng.normal(scale=0.5, size=180)
    w = rng.uniform(0.1, 2.5, size=180)
    alphas = np.array([0.01, 0.04, 0.12, 0.4])

    first = RidgeCV(
        alphas=alphas, cv=4, random_state=9, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=w)
    second = RidgeCV(
        alphas=alphas, cv=4, random_state=9, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=11.0 * w)

    assert first.alpha_ == second.alpha_
    np.testing.assert_allclose(first.mean_mse_, second.mean_mse_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(first.intercept_, second.intercept_, rtol=1e-11, atol=1e-11)


def test_weighted_sklearn_mapping_uses_total_weight():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge as SklearnRidge

    rng = np.random.default_rng(1208)
    X = rng.normal(size=(210, 5))
    y = -0.2 + X @ rng.normal(size=5) + rng.normal(scale=0.35, size=210)
    w = rng.uniform(0.15, 3.2, size=210)
    alpha = 0.14

    ours = Ridge(
        alpha=alpha, fit_intercept=True, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=w)
    reference = SklearnRidge(
        alpha=float(np.sum(w)) * alpha, fit_intercept=True,
    ).fit(X, y, sample_weight=w)

    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-9, atol=1e-9)



def test_backend_weight_validation_returns_scalar_sum_without_host_vector_conversion():
    from statgpu.linear_model.penalized._fit_mixin import _validate_sample_weight_backend

    weights = np.array([0.5, 1.5, 2.0])
    assert _validate_sample_weight_backend(weights, 3, "numpy") == 4.0
    with pytest.raises(ValueError, match="non-negative"):
        _validate_sample_weight_backend(np.array([1.0, -0.1]), 2, "numpy")

    torch = pytest.importorskip("torch")
    torch_weights = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64)
    assert _validate_sample_weight_backend(torch_weights, 3, "torch") == 4.0
    with pytest.raises(ValueError, match="finite"):
        _validate_sample_weight_backend(
            torch.tensor([1.0, float("nan")], dtype=torch.float64), 2, "torch"
        )



def test_penalized_glm_cv_weighted_alpha_grid_matches_null_gradient():
    from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV

    rng = np.random.default_rng(1209)
    X = rng.normal(size=(160, 5))
    y = 0.7 + X @ rng.normal(size=5) + rng.normal(scale=0.4, size=160)
    w = rng.uniform(0.1, 3.0, size=160)
    cv = PenalizedGLM_CV(
        loss="squared_error", penalty="l2", n_alphas=6,
        cv=3, random_state=4, device="cpu",
    )
    grid = cv._generate_alpha_grid(X, y, sample_weight=w)

    total = float(np.sum(w))
    x_mean = np.sum(X * w[:, None], axis=0) / total
    y_mean = float(np.sum(y * w) / total)
    expected_max = float(
        np.max(np.abs((X - x_mean).T @ (w * (y - y_mean)) / total))
    )
    np.testing.assert_allclose(grid[0], expected_max, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        grid,
        cv._generate_alpha_grid(X, y, sample_weight=13.0 * w),
        rtol=1e-12,
        atol=1e-12,
    )


def test_penalized_glm_cv_weighted_ridge_is_weight_scale_invariant():
    from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV

    rng = np.random.default_rng(1210)
    X = rng.normal(size=(150, 5))
    y = -0.3 + X @ rng.normal(size=5) + rng.normal(scale=0.45, size=150)
    w = rng.uniform(0.15, 2.8, size=150)

    kwargs = dict(
        loss="squared_error", penalty="l2", n_alphas=7,
        cv=3, random_state=7, device="cpu", max_iter=3000, tol=1e-10,
    )
    first = PenalizedGLM_CV(**kwargs).fit(X, y, sample_weight=w)
    second = PenalizedGLM_CV(**kwargs).fit(X, y, sample_weight=8.0 * w)

    np.testing.assert_allclose(first.alpha_grid_, second.alpha_grid_, rtol=1e-12, atol=1e-12)
    assert first.alpha_ == second.alpha_
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(first.intercept_, second.intercept_, rtol=1e-10, atol=1e-10)


def test_gpu_newton_ridge_cv_does_not_request_unused_gram_cache():
    from statgpu.linear_model.penalized._penalized_cv import (
        _should_build_squared_error_cv_cache,
    )

    assert not _should_build_squared_error_cv_cache(
        "squared_error", "l2", "newton", "torch"
    )
    assert not _should_build_squared_error_cv_cache(
        "squared_error", "l2", "newton", "cuda"
    )
    assert _should_build_squared_error_cv_cache(
        "squared_error", "l2", "exact", "torch"
    )
    assert _should_build_squared_error_cv_cache(
        "squared_error", "l1", "fista", "cuda"
    )
