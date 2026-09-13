"""Hosted CPU family matrix for weighted explicit ordinary GLM solvers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.linear_model import (
    GammaRegression,
    GeneralizedLinearModel,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    TweedieRegression,
)
from statgpu.solvers._convergence import ConvergenceWarning


def _design(seed=15301, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.25, size=(n, p)).astype(np.float64)
    beta = np.array([0.18, -0.12, 0.08], dtype=np.float64)[:p]
    eta = 0.2 + X @ beta
    return rng, X, eta


_CASE_SEEDS = {
    "gaussian": 15301,
    "binomial": 15302,
    "poisson": 15303,
    "gamma_log": 15304,
    "gamma_inverse": 15305,
    "inverse_gaussian": 15306,
    "negative_binomial": 15307,
    "tweedie": 15308,
}


def _case(case, solver, *, fit_intercept=True):
    rng, X, eta = _design(seed=_CASE_SEEDS[case])
    common = dict(
        solver=solver,
        device="cpu",
        max_iter=800,
        tol=1e-8,
        fit_intercept=fit_intercept,
    )

    if case == "gaussian":
        y = eta + rng.normal(scale=0.08, size=X.shape[0])
        model = GeneralizedLinearModel(family="gaussian", **common)
    elif case == "binomial":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(np.float64)
        y[0], y[1] = 0.0, 1.0
        model = GeneralizedLinearModel(family="binomial", **common)
    elif case == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
        model = GeneralizedLinearModel(family="poisson", **common)
    elif case == "gamma_log":
        mu = np.exp(eta)
        y = (mu * rng.lognormal(0.0, 0.08, size=X.shape[0])).astype(np.float64)
        model = GammaRegression(link="log", **common)
    elif case == "gamma_inverse":
        # Keep the inverse-link optimum well inside its positive eta domain.
        mu = 1.0 / np.clip(1.0 + X @ np.array([0.08, -0.05, 0.04]), 0.6, 1.4)
        y = (mu * rng.lognormal(0.0, 0.04, size=X.shape[0])).astype(np.float64)
        model = GammaRegression(link="inverse_power", **common)
    elif case == "inverse_gaussian":
        mu = np.exp(eta)
        y = (mu * rng.lognormal(0.0, 0.06, size=X.shape[0])).astype(np.float64)
        model = InverseGaussianRegression(**common)
    elif case == "negative_binomial":
        mu = np.exp(eta)
        # A Poisson draw is in-domain and provides a stable optimization case;
        # this test is solver-contract coverage, not a generative-model test.
        y = rng.poisson(mu).astype(np.float64)
        model = NegativeBinomialRegression(alpha=0.7, **common)
    elif case == "tweedie":
        mu = np.exp(eta)
        y = (mu * rng.lognormal(0.0, 0.08, size=X.shape[0])).astype(np.float64)
        model = TweedieRegression(power=1.5, **common)
    else:  # pragma: no cover - test parameter bug
        raise AssertionError(case)

    return model, X, np.asarray(y, dtype=np.float64)


_CASES = tuple(_CASE_SEEDS)
_NO_INTERCEPT_CASES = tuple(case for case in _CASES if case != "gamma_inverse")


def _fit_without_solver_warning(model, X, y, weights):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings(
            "error",
            message="lbfgs_solver: line search failed.*",
            category=RuntimeWarning,
        )
        return model.fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("case", _CASES)
def test_weighted_explicit_ordinary_glm_family_matrix(case, solver):
    model, X, y = _case(case, solver)
    weights = np.linspace(0.55, 1.65, X.shape[0], dtype=np.float64)

    _fit_without_solver_warning(model, X, y, weights)

    assert model._selected_solver == solver
    assert model._selected_backend_name == "numpy"
    assert model._selected_backend_device == "cpu"
    assert np.all(np.isfinite(np.asarray(model.coef_)))
    assert np.isfinite(float(model.intercept_))
    assert model.n_iter_ >= 1


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("case", _CASES)
def test_weighted_explicit_family_matrix_global_weight_rescaling(case, solver):
    model_a, X, y = _case(case, solver)
    model_b, _, _ = _case(case, solver)
    weights = np.linspace(0.6, 1.7, X.shape[0], dtype=np.float64)

    _fit_without_solver_warning(model_a, X, y, weights)
    _fit_without_solver_warning(model_b, X, y, 9.0 * weights)

    np.testing.assert_allclose(model_a.coef_, model_b.coef_, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(
        model_a.intercept_, model_b.intercept_, rtol=5e-6, atol=5e-7
    )


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("case", _NO_INTERCEPT_CASES)
def test_weighted_explicit_no_intercept_family_matrix(case, solver):
    # Inverse-power Gamma is the one reviewed exception: genuine non-uniform
    # weights + no intercept fail closed because no generic family-valid interior
    # start exists.  Every other claimed ordinary row must execute both explicit
    # smooth solvers warning-free without an intercept.
    model, X, y = _case(case, solver, fit_intercept=False)
    weights = np.linspace(0.55, 1.65, X.shape[0], dtype=np.float64)

    _fit_without_solver_warning(model, X, y, weights)

    assert model._selected_solver == solver
    assert model._selected_backend_name == "numpy"
    assert model._selected_backend_device == "cpu"
    assert np.all(np.isfinite(np.asarray(model.coef_)))
    assert model.intercept_ == 0.0
    assert model.n_iter_ >= 1


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("case", _NO_INTERCEPT_CASES)
def test_weighted_explicit_no_intercept_global_weight_rescaling(case, solver):
    model_a, X, y = _case(case, solver, fit_intercept=False)
    model_b, _, _ = _case(case, solver, fit_intercept=False)
    weights = np.linspace(0.6, 1.7, X.shape[0], dtype=np.float64)

    _fit_without_solver_warning(model_a, X, y, weights)
    _fit_without_solver_warning(model_b, X, y, 9.0 * weights)

    np.testing.assert_allclose(model_a.coef_, model_b.coef_, rtol=5e-6, atol=5e-7)
    assert model_a.intercept_ == model_b.intercept_ == 0.0
