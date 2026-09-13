"""Public ordinary-GLM regressions for issue #150."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu.linear_model import (
    GeneralizedLinearModel,
    OrderedLogitRegression,
    PoissonRegression,
)


def _logistic_data(seed=15101, n=120, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.8, size=(n, p)).astype(np.float64)
    beta = np.array([0.55, -0.32, 0.18])[:p]
    eta = -0.22 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X, y


def _poisson_data(seed=15102, n=120, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p)).astype(np.float64)
    beta = np.array([0.24, -0.17, 0.11])[:p]
    eta = 0.12 + X @ beta
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    return X, y


def _fit(family, solver, X, y, weights, **kwargs):
    return GeneralizedLinearModel(
        family=family,
        solver=solver,
        device="cpu",
        max_iter=600,
        tol=1e-10,
        **kwargs,
    ).fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_weighted_explicit_logistic_reaches_requested_solver_and_records_provenance(solver):
    X, y = _logistic_data()
    weights = np.linspace(0.45, 1.8, X.shape[0], dtype=np.float64)

    model = _fit("binomial", solver, X, y, weights)

    assert model._selected_solver == solver
    assert model._selected_backend_name == "numpy"
    assert model._selected_backend_device == "cpu"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_weighted_explicit_poisson_is_invariant_to_global_weight_rescaling(solver):
    X, y = _poisson_data()
    weights = np.linspace(0.35, 1.7, X.shape[0], dtype=np.float64)

    a = _fit("poisson", solver, X, y, weights)
    b = _fit("poisson", solver, X, y, 8.5 * weights)

    np.testing.assert_allclose(a.coef_, b.coef_, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(a.intercept_, b.intercept_, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_zero_weight_rows_equal_dropping_rows_for_public_logistic(solver):
    X, y = _logistic_data(seed=15103)
    weights = np.linspace(0.4, 1.6, X.shape[0], dtype=np.float64)
    weights[::9] = 0.0
    keep = weights > 0

    weighted = _fit("binomial", solver, X, y, weights)
    dropped = _fit("binomial", solver, X[keep], y[keep], weights[keep])

    np.testing.assert_allclose(weighted.coef_, dropped.coef_, rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(
        weighted.intercept_, dropped.intercept_, rtol=3e-7, atol=3e-8
    )


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_uniform_weights_match_unweighted_public_fit(solver):
    X, y = _logistic_data(seed=15104)
    unweighted = _fit("binomial", solver, X, y, None)
    weighted = _fit(
        "binomial",
        solver,
        X,
        y,
        np.full(X.shape[0], 4.0, dtype=np.float64),
    )

    np.testing.assert_allclose(unweighted.coef_, weighted.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        unweighted.intercept_, weighted.intercept_, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_weighted_explicit_solver_inference_uses_same_fit_contract(solver):
    X, y = _logistic_data(seed=15105, n=160)
    weights = np.linspace(0.5, 1.9, X.shape[0], dtype=np.float64)

    model = GeneralizedLinearModel(
        family="binomial",
        solver=solver,
        device="cpu",
        compute_inference=True,
        cov_type="hc0",
        max_iter=600,
        tol=1e-10,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == solver
    assert model._fit_metadata["solver_used"] == solver
    assert model._inference_result.metadata["solver_used"] == solver
    assert model._sample_weight_inf is not None
    assert np.all(np.isfinite(model._bse))
    assert np.all(np.isfinite(model._pvalues))
    assert np.all(np.isfinite(model._conf_int))
    assert np.isfinite(model.loglikelihood)
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_explicit_smooth_solver_preserves_existing_C_semantics(solver):
    X, y = _logistic_data(seed=15106)

    a = GeneralizedLinearModel(
        family="binomial", solver=solver, C=0.2, device="cpu", max_iter=600, tol=1e-10
    ).fit(X, y)
    b = GeneralizedLinearModel(
        family="binomial", solver=solver, C=5.0, device="cpu", max_iter=600, tol=1e-10
    ).fit(X, y)

    # Issue #150 must not silently redefine the pre-existing explicit smooth-
    # solver C contract while adding weight support.
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(a.intercept_, b.intercept_, rtol=0.0, atol=0.0)


def test_failed_refit_preserves_previous_successful_execution_provenance():
    X, y = _logistic_data(seed=15107)
    weights = np.linspace(0.5, 1.5, X.shape[0], dtype=np.float64)
    model = _fit("binomial", "lbfgs", X, y, weights)
    before = (
        model._selected_solver,
        model._selected_backend_name,
        model._selected_backend_device,
    )

    bad_weights = weights.copy()
    bad_weights[-1] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(X, y, sample_weight=bad_weights)

    assert (
        model._selected_solver,
        model._selected_backend_name,
        model._selected_backend_device,
    ) == before


def test_formula_weight_alignment_reaches_explicit_lbfgs_when_optional_deps_available():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, y = _poisson_data(seed=15108, n=90, p=2)
    data = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})
    data.loc[5, "x1"] = np.nan
    weights = np.linspace(0.5, 1.5, len(data), dtype=np.float64)

    model = GeneralizedLinearModel(
        family="poisson",
        solver="lbfgs",
        device="cpu",
        max_iter=600,
        tol=1e-10,
    ).fit(formula="y ~ x1 + x2", data=data, sample_weight=weights)

    assert model._selected_solver == "lbfgs"
    assert model._nobs == len(data) - 1
    assert model._sample_weight_inf.shape[0] == len(data) - 1


def test_typed_poisson_wrapper_inherits_weighted_explicit_lbfgs_contract():
    X, y = _poisson_data(seed=15109)
    weights = np.linspace(0.55, 1.65, X.shape[0], dtype=np.float64)

    model = PoissonRegression(
        solver="lbfgs", device="cpu", max_iter=600, tol=1e-10
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "lbfgs"
    assert np.all(np.isfinite(model.coef_))


def test_ordered_glm_weight_rejection_is_unchanged():
    rng = np.random.default_rng(15110)
    X = rng.normal(size=(40, 2))
    y = np.resize(np.array([0, 1, 2], dtype=np.int64), X.shape[0])
    weights = np.linspace(0.5, 1.5, X.shape[0])

    model = OrderedLogitRegression(n_categories=3, device="cpu")
    with pytest.raises(ValueError, match="does not support sample_weight"):
        model.fit(X, y, sample_weight=weights)


def test_weighted_explicit_solver_installer_is_idempotent_and_preserves_signatures():
    from statgpu.linear_model import _glm_weighted_explicit_solver_contract as contract

    before_fit = GeneralizedLinearModel.fit
    before_smooth = GeneralizedLinearModel._fit_smooth_solver
    fit_signature = inspect.signature(before_fit)
    smooth_signature = inspect.signature(before_smooth)

    contract.install_glm_weighted_explicit_solver_contract()

    assert GeneralizedLinearModel.fit is before_fit
    assert GeneralizedLinearModel._fit_smooth_solver is before_smooth
    assert inspect.signature(GeneralizedLinearModel.fit) == fit_signature
    assert inspect.signature(GeneralizedLinearModel._fit_smooth_solver) == smooth_signature
    assert hasattr(GeneralizedLinearModel.fit, "__wrapped__")
    assert hasattr(GeneralizedLinearModel._fit_smooth_solver, "__wrapped__")
