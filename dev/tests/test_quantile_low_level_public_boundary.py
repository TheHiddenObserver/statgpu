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


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_public_proximal_quantile_solver_rejects_invalid_weights(sample_weight):
    X, y = _data(seed=16702)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    with pytest.raises(ValueError, match="sample_weight"):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
            sample_weight=sample_weight,
        )


def test_public_proximal_quantile_none_budget_resolves_before_kernel(monkeypatch):
    X, y = _data(seed=16703)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)
    captured = {}

    def fake_kernel(
        loss,
        penalty,
        X,
        y,
        alpha_path,
        max_lla_per_step=2,
        lla_tol=1e-6,
        max_iter=None,
        tol=1e-6,
        fit_intercept=True,
        sample_weight=None,
    ):
        captured["max_iter"] = max_iter
        captured["sample_weight"] = sample_weight
        return np.zeros(X.shape[1]), 0.0, 0

    monkeypatch.setattr(
        _prox_contract,
        "_proximal_irls_quantile_solver",
        fake_kernel,
    )

    weights = np.arange(1, X.shape[0] + 1, dtype=np.float64)
    coef, intercept, n_iter = solvers.proximal_irls_quantile_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path=np.array([0.05]),
        sample_weight=weights,
    )

    assert captured["max_iter"] == 100
    np.testing.assert_array_equal(captured["sample_weight"], weights)
    np.testing.assert_array_equal(coef, np.zeros(X.shape[1]))
    assert intercept == 0.0
    assert n_iter == 0


def test_public_proximal_quantile_wrapper_preserves_introspection():
    public = solvers.proximal_irls_quantile_solver
    kernel = _prox_kernel.proximal_irls_quantile_solver

    assert inspect.signature(public) == inspect.signature(kernel)
    assert getattr(public, "__wrapped__", None) is kernel
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
