"""PR151 regressions for L-BFGS Armijo failure semantics."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.solvers import lbfgs_solver


def _data(seed=151941, n=48, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.04, size=(n, p)).astype(np.float64)
    X[:, 0] = rng.uniform(0.9, 1.1, size=n)
    beta = np.array([0.95, 0.03], dtype=np.float64)
    eta = X @ beta
    y = (1.0 / eta) * rng.lognormal(0.0, 0.02, size=n)
    weights = np.linspace(0.6, 1.6, n, dtype=np.float64)
    return X, y.astype(np.float64), weights


def test_inverse_gamma_lbfgs_armijo_exhaustion_is_not_published(monkeypatch):
    import statgpu.solvers._lbfgs as lbfgs_mod

    X, y, weights = _data()
    loss = get_glm_loss("gamma", link="inverse_power")

    # Force every numerically interior Armijo comparison to reject. The solver
    # must try its domain-aware steepest-descent recovery and then fail closed;
    # a warning plus returned coefficient vector would publish a point known not
    # to satisfy the requested convergence contract.
    monkeypatch.setattr(lbfgs_mod, "_device_leq", lambda *args, **kwargs: False)

    with pytest.raises(
        RuntimeError,
        match="Armijo line search failed inside the maintained loss domain",
    ):
        lbfgs_solver(
            loss,
            None,
            X,
            y,
            max_iter=5,
            tol=1.0e-12,
            sample_weight=weights,
        )


def test_unconstrained_lbfgs_armijo_exhaustion_warns_and_stops(monkeypatch):
    import statgpu.solvers._lbfgs as lbfgs_mod

    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loss = get_glm_loss("squared_error")

    # The historical unconstrained contract is warning + stop, not silent
    # convergence and not a hard domain error. Force all Armijo comparisons to
    # reject so the regression does not depend on floating-point step size.
    monkeypatch.setattr(lbfgs_mod, "_device_leq", lambda *args, **kwargs: False)

    with pytest.warns(RuntimeWarning, match="line search failed to find a descent step"):
        params, n_iter = lbfgs_solver(
            loss,
            None,
            X,
            y,
            max_iter=5,
            tol=1.0e-12,
        )

    np.testing.assert_allclose(params, np.zeros(1), rtol=0.0, atol=0.0)
    assert n_iter == 1
