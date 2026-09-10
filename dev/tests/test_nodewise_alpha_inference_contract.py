import inspect
import math

import numpy as np
import pytest

from statgpu.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
)
from statgpu.linear_model.penalized._nodewise_precision import (
    build_nodewise_precision_numpy,
    resolve_effective_n,
)


def _data(seed=123, n=96, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[: min(3, p)] = [1.2, -0.7, 0.45][: min(3, p)]
    y = X @ beta + rng.normal(scale=0.35, size=n)
    return X, y


def _fit_lasso(X, y, **kwargs):
    sample_weight = kwargs.pop("sample_weight", None)
    params = dict(
        alpha=0.05,
        device="cpu",
        solver="fista",
        max_iter=2000,
        tol=1e-7,
        compute_inference=True,
        inference_method="debiased",
    )
    params.update(kwargs)
    return Lasso(**params).fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "cls",
    [
        PenalizedGeneralizedLinearModel,
        PenalizedLinearRegression,
        Lasso,
        ElasticNet,
        LassoCV,
        ElasticNetCV,
    ],
)
def test_public_signatures_expose_nodewise_alpha(cls):
    signature = inspect.signature(cls.__init__)
    assert "nodewise_alpha" in signature.parameters
    assert signature.parameters["nodewise_alpha"].default is None


@pytest.mark.parametrize(
    "value",
    [False, True, 0.0, -0.1, np.nan, np.inf, -np.inf, 1 + 2j, [0.1], np.array([0.1])],
)
def test_nodewise_alpha_rejects_invalid_public_values(value):
    with pytest.raises(ValueError, match="nodewise_alpha"):
        Lasso(nodewise_alpha=value)


def test_nodewise_alpha_preserves_numpy_scalar_for_clone_and_set_params():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    requested = np.float64(0.075)
    model = Lasso(nodewise_alpha=requested, compute_inference=False)
    assert model.get_params(deep=False)["nodewise_alpha"] is requested
    cloned = clone(model)
    assert float(cloned.nodewise_alpha) == pytest.approx(float(requested))
    assert cloned.nodewise_alpha_ is None

    model.set_params(nodewise_alpha=0.06)
    assert model.nodewise_alpha == 0.06
    assert model.nodewise_alpha_ is None


def test_auto_precision_is_response_scale_independent():
    X, y = _data(n=80, p=5)
    first = _fit_lasso(X, y)
    second = _fit_lasso(X, 100.0 * y)
    expected = math.sqrt(2.0 * math.log(X.shape[1]) / X.shape[0])
    assert first.nodewise_alpha_ == pytest.approx(expected)
    assert second.nodewise_alpha_ == pytest.approx(expected)
    np.testing.assert_allclose(first._debiased_M_cpu, second._debiased_M_cpu, rtol=2e-8, atol=2e-10)


def test_auto_and_explicit_resolved_alpha_are_equivalent():
    X, y = _data(n=84, p=5)
    auto = _fit_lasso(X, y)
    explicit = _fit_lasso(X, y, nodewise_alpha=auto.nodewise_alpha_)
    np.testing.assert_allclose(auto._debiased_M_cpu, explicit._debiased_M_cpu, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(auto._params, explicit._params, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(auto._bse, explicit._bse, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(auto._pvalues, explicit._pvalues, rtol=2e-8, atol=2e-10)
    assert explicit._inference_result.metadata["nodewise_alpha_source"] == "user"
    assert auto._inference_result.metadata["nodewise_alpha_source"] == "auto"


def test_precision_helper_has_feature_scale_equivariance():
    X, _ = _data(n=100, p=4)
    X = X - X.mean(axis=0)
    M, alpha, _ = build_nodewise_precision_numpy(
        X, requested_alpha=None, effective_n=float(X.shape[0]), weighted=False
    )
    scale = np.asarray([0.5, 2.0, 3.0, 1.25])
    M_scaled, alpha_scaled, _ = build_nodewise_precision_numpy(
        X * scale, requested_alpha=None, effective_n=float(X.shape[0]), weighted=False
    )
    expected = (1.0 / scale)[:, None] * M * (1.0 / scale)[None, :]
    assert alpha_scaled == pytest.approx(alpha)
    np.testing.assert_allclose(M_scaled, expected, rtol=5e-7, atol=5e-9)


def test_forced_bad_nodewise_solution_fails_independent_kkt_gate(monkeypatch):
    from statgpu.linear_model.wrappers import _lasso as lasso_module

    rng = np.random.default_rng(2026)
    x0 = rng.normal(size=96)
    X = np.column_stack(
        [
            x0,
            0.85 * x0 + 0.15 * rng.normal(size=x0.size),
            rng.normal(size=x0.size),
        ]
    )
    X -= X.mean(axis=0)

    def bad_solver(gram, cross, *args, **kwargs):
        p = int(np.asarray(gram).shape[-1])
        return np.zeros((1, p), dtype=np.float64), None

    monkeypatch.setattr(
        lasso_module,
        "_solve_lasso_path_cpu_from_gram",
        bad_solver,
    )
    with pytest.raises(FloatingPointError, match="KKT publication gate"):
        build_nodewise_precision_numpy(
            X,
            requested_alpha=1e-8,
            effective_n=float(X.shape[0]),
            weighted=False,
        )


def test_weight_effective_n_is_scale_and_zero_row_invariant():
    w = np.asarray([1.0, 2.0, 0.5, 0.0, 3.0])
    n_eff = resolve_effective_n(w.size, w)
    assert resolve_effective_n(w.size, 17.0 * w) == pytest.approx(n_eff)
    w_extra = np.concatenate([w, [0.0]])
    assert resolve_effective_n(w_extra.size, w_extra) == pytest.approx(n_eff)


def test_all_one_weights_match_unweighted_precision_and_alpha():
    X, y = _data(n=72, p=4)
    plain = _fit_lasso(X, y)
    weighted = _fit_lasso(X, y, sample_weight=np.ones(X.shape[0]))
    assert weighted.nodewise_alpha_ == pytest.approx(plain.nodewise_alpha_)
    np.testing.assert_allclose(weighted._debiased_M_cpu, plain._debiased_M_cpu, rtol=2e-7, atol=2e-9)


def test_global_weight_scale_does_not_change_precision():
    X, y = _data(n=76, p=4)
    rng = np.random.default_rng(8)
    w = rng.uniform(0.2, 2.0, size=X.shape[0])
    first = _fit_lasso(X, y, sample_weight=w)
    second = _fit_lasso(X, y, sample_weight=11.0 * w)
    assert first.nodewise_alpha_ == pytest.approx(second.nodewise_alpha_)
    np.testing.assert_allclose(first._debiased_M_cpu, second._debiased_M_cpu, rtol=3e-7, atol=3e-9)


def test_zero_weight_row_does_not_change_weighted_precision_problem():
    X, y = _data(n=70, p=4)
    rng = np.random.default_rng(11)
    w = rng.uniform(0.3, 1.8, size=X.shape[0])
    base = _fit_lasso(X, y, sample_weight=w)
    X2 = np.vstack([X, np.full((1, X.shape[1]), 1e6)])
    y2 = np.concatenate([y, [1e9]])
    w2 = np.concatenate([w, [0.0]])
    extra = _fit_lasso(X2, y2, sample_weight=w2)
    assert base.nodewise_alpha_ == pytest.approx(extra.nodewise_alpha_)
    np.testing.assert_allclose(base._debiased_M_cpu, extra._debiased_M_cpu, rtol=4e-7, atol=4e-9)


def test_p1_uses_analytic_precision_without_consuming_requested_alpha():
    X, y = _data(n=60, p=1)
    model = _fit_lasso(X, y, nodewise_alpha=0.123)
    assert model.nodewise_alpha == 0.123
    assert model.nodewise_alpha_ is None
    meta = model._inference_result.metadata
    assert meta["precision_method"] == "analytic_univariate"
    assert meta["nodewise_alpha_source"] == "not_applicable"
    assert meta["nodewise_alpha_requested"] == pytest.approx(0.123)
    assert np.asarray(model._debiased_M_cpu).shape == (1, 1)


def test_non_debiased_path_leaves_resolved_nodewise_state_empty():
    X, y = _data(n=70, p=4)
    model = Lasso(
        alpha=0.05,
        nodewise_alpha=0.08,
        device="cpu",
        compute_inference=True,
        inference_method="post_selection_ols",
    ).fit(X, y)
    assert model.nodewise_alpha_ is None


def test_degenerate_design_fails_closed_without_stale_nodewise_state():
    X, y = _data(n=70, p=4)
    model = _fit_lasso(X, y)
    assert model.nodewise_alpha_ is not None
    X_bad = X.copy()
    X_bad[:, 2] = 1.0
    with pytest.raises(FloatingPointError, match="node-wise design"):
        model.fit(X_bad, y)
    assert model.nodewise_alpha_ is None
    assert getattr(model, "_inference_result", None) is None


def test_lassocv_nodewise_alpha_is_final_refit_only():
    X, y = _data(n=90, p=4)
    common = dict(
        alphas=[0.03, 0.06],
        cv=3,
        device="cpu",
        solver="fista",
        cv_solver="fista",
        compute_inference=True,
        inference_method="debiased",
        max_iter=1200,
        tol=1e-6,
        random_state=7,
    )
    first = LassoCV(nodewise_alpha=0.07, **common).fit(X, y)
    second = LassoCV(nodewise_alpha=0.11, **common).fit(X, y)
    assert first.alpha_ == pytest.approx(second.alpha_)
    np.testing.assert_allclose(first.mse_path_, second.mse_path_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=0.0, atol=0.0)
    assert first.estimator_.nodewise_alpha_ == pytest.approx(0.07)
    assert second.estimator_.nodewise_alpha_ == pytest.approx(0.11)
    assert first.nodewise_alpha_ == pytest.approx(0.07)
    assert second.nodewise_alpha_ == pytest.approx(0.11)


def test_elasticnet_direct_and_cv_propagate_nodewise_alpha():
    X, y = _data(n=88, p=4)
    direct = ElasticNet(
        alpha=0.05,
        l1_ratio=0.7,
        nodewise_alpha=0.09,
        device="cpu",
        compute_inference=True,
        inference_method="debiased",
        max_iter=1600,
        tol=1e-6,
    ).fit(X, y)
    assert direct.nodewise_alpha_ == pytest.approx(0.09)

    cv = ElasticNetCV(
        l1_ratio=[0.5, 0.8],
        alphas=[0.03, 0.06],
        cv=3,
        nodewise_alpha=0.09,
        device="cpu",
        compute_inference=True,
        max_iter=1200,
        tol=1e-6,
        random_state=9,
    ).fit(X, y)
    assert cv.estimator_.nodewise_alpha_ == pytest.approx(0.09)
    assert cv.nodewise_alpha_ == pytest.approx(0.09)


def test_generic_penalized_l1_exposes_same_nodewise_contract():
    X, y = _data(n=78, p=4)
    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.05,
        nodewise_alpha=0.085,
        device="cpu",
        solver="fista",
        compute_inference=True,
        inference_method="debiased",
        max_iter=1600,
        tol=1e-6,
    ).fit(X, y)
    assert model.nodewise_alpha_ == pytest.approx(0.085)
    assert model._inference_result.metadata["nodewise_alpha"] == pytest.approx(0.085)
