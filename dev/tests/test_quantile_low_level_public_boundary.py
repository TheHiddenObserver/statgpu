"""Fresh-review regressions for public low-level Quantile boundaries."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from statgpu import solvers
from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
import statgpu.losses._quantile_irls_validation_contract as _irls_contract
import statgpu.solvers._quantile_proximal_public_contract as _prox_contract
from statgpu.solvers import _proximal_irls_quantile as _prox_kernel


def _data(seed=16701):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 3))
    y = 0.2 + X @ np.array([0.5, -0.25, 0.1])
    y = y + rng.laplace(scale=0.08, size=X.shape[0])
    return X, y


def _invalid_weights(n):
    negative = np.ones(n, dtype=np.float64)
    negative[2] = -0.25
    nonfinite = np.ones(n, dtype=np.float64)
    nonfinite[3] = np.nan
    return [
        pytest.param(np.ones(n - 1), id="wrong-length"),
        pytest.param(negative, id="negative"),
        pytest.param(nonfinite, id="nonfinite"),
        pytest.param(np.zeros(n), id="zero-sum"),
    ]


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_direct_quantile_irls_rejects_invalid_weights_before_numerics(sample_weight):
    X, y = _data()
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match="sample_weight"):
        loss.irls(X, y, sample_weight=sample_weight, max_iter=3)


def test_direct_quantile_irls_validation_preserves_array_like_design_input():
    X, y = _data(seed=16704)
    loss = QuantileLoss(quantile=0.3)

    coef, n_iter = loss.irls(
        X.tolist(),
        y.tolist(),
        sample_weight=np.ones(X.shape[0]),
        max_iter=2,
    )

    assert np.asarray(coef).shape == (X.shape[1],)
    assert np.all(np.isfinite(np.asarray(coef)))
    assert 1 <= n_iter <= 2


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_public_proximal_quantile_solver_rejects_invalid_weights(sample_weight):
    X, y = _data(seed=16702)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    for solver_fn in (
        solvers.proximal_irls_quantile_solver,
        _prox_kernel.proximal_irls_quantile_solver,
    ):
        with pytest.raises(ValueError, match="sample_weight"):
            solver_fn(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.08, 0.05]),
                max_iter=3,
                sample_weight=sample_weight,
            )


def test_public_proximal_quantile_none_budget_has_defined_default():
    X, y = _data(seed=16703)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    coef, intercept, n_iter = solvers.proximal_irls_quantile_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path=np.array([0.05]),
        max_lla_per_step=1,
        sample_weight=np.ones(X.shape[0]),
    )

    assert np.all(np.isfinite(coef))
    assert np.isfinite(intercept)
    assert 0 <= n_iter <= 100


def test_public_proximal_quantile_wrapper_preserves_introspection_and_alias():
    public = solvers.proximal_irls_quantile_solver
    historical = _prox_kernel.proximal_irls_quantile_solver
    kernel = getattr(public, "__wrapped__", None)

    assert historical is public
    assert kernel is not None
    assert inspect.signature(public) == inspect.signature(kernel)
    doc = " ".join((inspect.getdoc(public) or "").split())
    assert "None`` uses 100 iterations per step" in doc


def test_quantile_irls_validation_installer_is_idempotent_under_reload():
    before = QuantileLoss.irls
    before_wrapped = getattr(before, "__wrapped__", None)

    _irls_contract.install_quantile_irls_validation_contract()
    assert QuantileLoss.irls is before

    importlib.reload(_irls_contract)
    after = QuantileLoss.irls
    assert after is before
    assert getattr(after, _irls_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped


def test_quantile_proximal_contract_is_idempotent_under_reload():
    before = solvers.proximal_irls_quantile_solver
    before_wrapped = getattr(before, "__wrapped__", None)

    assert _prox_contract.install_quantile_proximal_public_contract() is before
    importlib.reload(_prox_contract)
    after = _prox_kernel.proximal_irls_quantile_solver

    assert after is before
    assert getattr(after, _prox_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped
