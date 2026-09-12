"""Boundary tests for weighted inverse-link Gamma smooth solvers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.linear_model import GammaRegression


def _data(seed=15601, n=80, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.2, size=(n, p)).astype(np.float64)
    eta = np.clip(1.0 + X @ np.array([0.08, -0.05, 0.04]), 0.7, 1.3)
    mu = 1.0 / eta
    y = (mu * rng.lognormal(0.0, 0.035, size=n)).astype(np.float64)
    weights = np.linspace(0.55, 1.65, n, dtype=np.float64)
    return X, y, weights


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_weighted_inverse_gamma_no_intercept_fails_closed_precisely(solver):
    X, y, weights = _data()
    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )

    with pytest.raises(ValueError, match="requires fit_intercept=True"):
        model.fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("almost_uniform", [False, True])
def test_inverse_gamma_no_intercept_uniform_weights_keep_unweighted_path(
    solver, almost_uniform
):
    X, y, _ = _data(seed=15602)
    weights = np.full(X.shape[0], 3.5, dtype=np.float64)
    if almost_uniform:
        # Preserve the established floating allclose compatibility rule used by
        # both smooth solvers rather than treating this as genuine weighting.
        weights[-1] += 1e-8

    kwargs = dict(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=5,
        tol=1e-8,
    )
    # This historical no-intercept path may emit its existing convergence or
    # line-search warning.  The regression target is that uniform/effectively
    # uniform weights execute the identical unweighted numerical path and are
    # not captured by the new genuine-nonuniform fail-closed boundary.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = GammaRegression(**kwargs).fit(X, y)
        weighted = GammaRegression(**kwargs).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(base.coef_, weighted.coef_, rtol=0.0, atol=0.0)
    assert base.intercept_ == weighted.intercept_ == 0.0


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("almost_uniform", [False, True])
def test_inverse_gamma_intercept_uniform_weights_reuse_unweighted_warm_start(
    solver, almost_uniform, monkeypatch
):
    import statgpu.solvers as solvers

    X, y, _ = _data(seed=15603)
    weights = np.full(X.shape[0], 2.75, dtype=np.float64)
    if almost_uniform:
        weights[-1] += 1e-8

    captured = []

    def capture_solver(*args, **kwargs):
        init = np.asarray(kwargs["init_coef"], dtype=np.float64).copy()
        captured.append(init)
        # Returning the supplied start isolates the public-boundary contract:
        # this test verifies the exact init_coef handed to the requested solver,
        # independent of later iterative convergence details.
        return kwargs["init_coef"], 0

    monkeypatch.setattr(solvers, f"{solver}_solver", capture_solver)
    kwargs = dict(
        link="inverse_power",
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=5,
        tol=1e-8,
    )
    GammaRegression(**kwargs).fit(X, y)
    GammaRegression(**kwargs).fit(X, y, sample_weight=weights)

    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], captured[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_no_intercept_uses_torch_promoted_working_dtype_for_boundary(
    solver,
):
    torch = pytest.importorskip("torch")

    # In float16 these weights collapse to one value; in the solver's promoted
    # float64 working dtype they are genuinely non-uniform.  The capability
    # boundary must therefore use X_work, not the pre-promotion input X.
    X = torch.ones((4, 2), dtype=torch.float16)
    y = torch.tensor([0.9, 1.0, 1.1, 1.2], dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0004, 1.0, 1.0], dtype=torch.float64)
    assert bool(torch.all(X.new_tensor(weights, dtype=torch.float16) == 1.0))
    assert not bool(torch.allclose(weights, weights[0]))

    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=5,
        tol=1e-8,
    )
    with pytest.raises(ValueError, match="requires fit_intercept=True"):
        model._fit_smooth_solver(X, y, weights, solver, "torch")


def test_unweighted_inverse_gamma_no_intercept_keeps_historical_boundary():
    # Issue #150 narrows only the newly opened genuinely weighted row.  It does
    # not turn the historical unweighted no-intercept path into an API migration.
    X, y, _ = _data(seed=15604)
    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver="lbfgs",
        device="cpu",
        max_iter=5,
        tol=1e-8,
    )
    # The historical path may converge or emit its existing numerical warning,
    # but it must not be rejected by the new weighted-only capability guard.
    try:
        model.fit(X, y)
    except ValueError as exc:
        assert "requires fit_intercept=True" not in str(exc)
