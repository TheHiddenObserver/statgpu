"""Boundary tests for inverse-power Gamma explicit smooth solvers."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.linear_model import GammaRegression


def _feasible_data(seed=15601, n=80, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.08, size=(n, p)).astype(np.float64)
    X[:, 0] = rng.uniform(0.8, 1.2, size=n)
    beta = np.array([0.9, 0.05, -0.04], dtype=np.float64)
    eta = X @ beta
    assert np.all(eta > 0)
    mu = 1.0 / eta
    y = (mu * rng.lognormal(0.0, 0.035, size=n)).astype(np.float64)
    weights = np.linspace(0.55, 1.65, n, dtype=np.float64)
    return X, y, weights


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_weighted_inverse_gamma_no_intercept_is_supported_when_domain_feasible(solver):
    X, y, weights = _feasible_data()
    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=500,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    eta = X @ model.coef_
    loss = get_glm_loss("gamma", link="inverse_power")
    lo, hi = loss._loss_domain_bounds(X)
    assert np.all(np.isfinite(model.coef_))
    assert np.all(eta > lo)
    assert np.all(eta < hi)
    assert model.intercept_ == 0.0


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_one_dimensional_weighted_solution_matches_closed_form(solver):
    x = np.array([0.7, 0.9, 1.1, 1.3, 1.6], dtype=np.float64)
    X = x[:, None]
    y = np.array([1.4, 1.1, 0.95, 0.8, 0.65], dtype=np.float64)
    weights = np.array([0.5, 1.0, 1.7, 0.8, 2.0], dtype=np.float64)
    expected = weights.sum() / np.sum(weights * y * x)

    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=500,
        tol=1e-11,
    ).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(model.coef_, [expected], rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("almost_uniform", [False, True])
def test_inverse_gamma_no_intercept_uniform_weights_equal_unweighted_objective(
    solver, almost_uniform
):
    X, y, _ = _feasible_data(seed=15602)
    weights = np.full(X.shape[0], 3.5, dtype=np.float64)
    if almost_uniform:
        weights[-1] += 1e-8

    kwargs = dict(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=500,
        tol=1e-9,
    )
    base = GammaRegression(**kwargs).fit(X, y)
    weighted = GammaRegression(**kwargs).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(base.coef_, weighted.coef_, rtol=0.0, atol=0.0)
    assert base.intercept_ == weighted.intercept_ == 0.0


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_intercept_initializer_reduces_to_inverse_mean(solver):
    X, y, weights = _feasible_data(seed=15603)
    design = np.column_stack([X, np.ones(X.shape[0])])
    loss = get_glm_loss("gamma", link="inverse_power")

    init = loss._loss_domain_initial_point(design, y, sample_weight=weights)
    expected = 1.0 / np.average(y, weights=weights)
    np.testing.assert_allclose(init[:-1], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(init[-1], expected, rtol=1e-12, atol=1e-12)

    model = GammaRegression(
        link="inverse_power",
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=500,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)
    assert np.isfinite(model.intercept_)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_zero_weight_conflict_is_equivalent_to_row_deletion(solver):
    X = np.array([[1.0], [1.4], [-1.0]], dtype=np.float64)
    y = np.array([1.0, 0.8, 1.2], dtype=np.float64)
    weights = np.array([1.0, 2.0, 0.0], dtype=np.float64)

    weighted = GammaRegression(
        link="inverse_power", fit_intercept=False, solver=solver,
        device="cpu", max_iter=500, tol=1e-10,
    ).fit(X, y, sample_weight=weights)
    deleted = GammaRegression(
        link="inverse_power", fit_intercept=False, solver=solver,
        device="cpu", max_iter=500, tol=1e-10,
    ).fit(X[:2], y[:2], sample_weight=weights[:2])

    np.testing.assert_allclose(weighted.coef_, deleted.coef_, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_active_contradiction_fails_without_false_infeasibility_claim(solver):
    X = np.array([[1.0], [-1.0]], dtype=np.float64)
    y = np.array([1.0, 1.0], dtype=np.float64)

    model = GammaRegression(
        link="inverse_power", fit_intercept=False, solver=solver,
        device="cpu", max_iter=100, tol=1e-8,
    )
    with pytest.raises(RuntimeError, match="numerically certified smooth-domain start"):
        model.fit(X, y)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_no_intercept_uses_torch_promoted_working_dtype(solver):
    torch = pytest.importorskip("torch")

    X = torch.tensor(
        [[1.0, 0.05], [1.1, -0.02], [0.9, 0.03], [1.2, 0.01]],
        dtype=torch.float16,
    )
    y = torch.tensor([0.9, 1.0, 1.1, 1.2], dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0004, 1.0, 1.0], dtype=torch.float64)
    assert bool(torch.all(X.new_tensor(weights, dtype=torch.float16) == 1.0))
    assert not bool(torch.allclose(weights, weights[0]))

    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=200,
        tol=1e-8,
    )
    model._fit_smooth_solver(X, y, weights, solver, "torch")
    assert np.all(np.isfinite(model.coef_))
