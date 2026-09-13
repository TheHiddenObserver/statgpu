"""Execution-provenance preservation tests for issue #150."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import GeneralizedLinearModel


def _logistic_data(seed=15501, n=100, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.6, size=(n, p)).astype(np.float64)
    beta = np.array([0.45, -0.25, 0.12])[:p]
    eta = 0.08 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    weights = np.linspace(0.5, 1.5, n, dtype=np.float64)
    return X, y, weights


def _prov(model):
    return (
        model._selected_solver,
        model._selected_backend_name,
        model._selected_backend_device,
    )


def test_auto_weighted_ordinary_glm_matches_explicit_irls_and_reports_truthful_provenance():
    X, y, weights = _logistic_data()

    auto = GeneralizedLinearModel(
        family="binomial",
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)
    irls = GeneralizedLinearModel(
        family="binomial",
        solver="irls",
        device="cpu",
        max_iter=300,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    assert _prov(auto) == ("irls", "numpy", "cpu")
    assert _prov(irls) == ("irls", "numpy", "cpu")
    np.testing.assert_allclose(auto.coef_, irls.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(auto.intercept_, irls.intercept_, rtol=0.0, atol=0.0)


def test_explicit_fista_weighted_ordinary_glm_reports_fista_without_dispatch_change():
    X, y, weights = _logistic_data(seed=15502)
    model = GeneralizedLinearModel(
        family="binomial",
        solver="fista",
        device="cpu",
        max_iter=500,
        tol=1e-8,
    ).fit(X, y, sample_weight=weights)

    assert _prov(model) == ("fista", "numpy", "cpu")
    assert np.all(np.isfinite(np.asarray(model.coef_)))
    assert np.isfinite(float(model.intercept_))


def test_input_validation_failure_does_not_publish_attempted_provenance():
    X, y, weights = _logistic_data(seed=15503)
    model = GeneralizedLinearModel(
        family="binomial",
        solver="lbfgs",
        device="cpu",
        max_iter=400,
        tol=1e-8,
    ).fit(X, y, sample_weight=weights)
    before = _prov(model)

    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(X_bad, y, sample_weight=weights)

    assert _prov(model) == before


def test_solver_failure_does_not_publish_attempted_provenance(monkeypatch):
    import statgpu.solvers as solvers

    X, y, weights = _logistic_data(seed=15504)
    model = GeneralizedLinearModel(
        family="binomial",
        solver="lbfgs",
        device="cpu",
        max_iter=400,
        tol=1e-8,
    ).fit(X, y, sample_weight=weights)
    before = _prov(model)

    def fail_lbfgs(*args, **kwargs):
        raise RuntimeError("synthetic lbfgs failure")

    monkeypatch.setattr(solvers, "lbfgs_solver", fail_lbfgs)
    with pytest.raises(RuntimeError, match="synthetic lbfgs failure"):
        model.fit(X, y, sample_weight=weights)

    # The wrapper publishes execution provenance only after successful fit.
    # A failed same-solver refit therefore preserves the last completed fit's
    # provenance rather than advertising the failed attempt as new evidence.
    assert _prov(model) == before
