"""Regression contract for weighted Quantile SCAD/MCP continuation starts."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from dev.benchmarks import validate_quantile_solver_provenance_gpu as _pr164_gate
from statgpu.linear_model.penalized import (
    PenalizedGLM_CV,
    PenalizedQuantileRegression,
)
from statgpu.losses import QuantileLoss
from statgpu.solvers import _quantile_continuation as _continuation
from statgpu.solvers._quantile_continuation import (
    is_auto_quantile_continuation_path,
    mark_auto_quantile_continuation_path,
    resolve_auto_quantile_continuation_path,
)
import statgpu.linear_model.penalized._quantile_continuation_contract as _path_contract
import statgpu.linear_model.penalized._fit_mixin as _fit_mixin
import statgpu.solvers._quantile_proximal_public_contract as _prox_contract


def _data(seed=16671, n=20, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 0.25 + X @ np.array([0.55, -0.30, 0.15])
    y = y + rng.exponential(scale=0.20, size=n) - 0.20
    weights = np.linspace(0.4, 2.0, n)
    rng.shuffle(weights)
    return X, y, weights


def _balanced_psi(residual, tau, weights):
    residual = np.asarray(residual, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive = residual > 0.0
    negative = residual < 0.0
    zero = ~(positive | negative)
    positive_mass = float(np.sum(weights * positive))
    negative_mass = float(np.sum(weights * negative))
    zero_mass = float(np.sum(weights * zero))
    fixed_sum = tau * positive_mass - (1.0 - tau) * negative_mass
    zero_value = -fixed_sum / zero_mass if zero_mass > 0.0 else 0.0
    return np.where(
        positive,
        tau,
        np.where(negative, -(1.0 - tau), zero_value),
    )


def _manual_weighted_start(X, y, weights, tau):
    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    w_sorted = weights[order]
    cutoff = tau * float(np.sum(w_sorted))
    index = int(np.searchsorted(np.cumsum(w_sorted), cutoff, side="left"))
    intercept = float(y_sorted[min(index, y_sorted.size - 1)])
    residual = y - intercept
    psi = _balanced_psi(residual, tau, weights)
    return float(
        np.max(np.abs(X.T @ (weights * psi) / float(np.sum(weights))))
    )


def test_weighted_auto_continuation_matches_declared_weighted_score():
    X, y, weights = _data()
    loss = QuantileLoss(quantile=0.30)
    target = 0.04
    auto = mark_auto_quantile_continuation_path(
        np.geomspace(0.2, target, 3)
    )

    observed = resolve_auto_quantile_continuation_path(
        loss,
        X,
        y,
        auto,
        sample_weight=weights,
        fit_intercept=True,
    )
    lambda_start = _manual_weighted_start(X, y, weights, 0.30)
    expected = np.geomspace(max(lambda_start, target * 1.1), target, 3)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)


def test_weighted_quantile_cutoff_roundoff_cannot_overflow_searchsorted():
    # Pairwise sum retains some unit weights that sequential cumsum loses
    # after the leading 1e16. At the largest float64 tau below one this makes
    # tau * sum(w) exceed cumsum(w)[-1], which previously returned index == n.
    n_small = 10
    y = np.arange(n_small + 1, dtype=np.float64)
    weights = np.concatenate(
        [np.asarray([1.0e16], dtype=np.float64), np.ones(n_small)]
    )
    tau = np.nextafter(1.0, 0.0)
    assert tau * np.sum(weights) > np.cumsum(weights)[-1]

    intercept = _continuation._weighted_lower_quantile_backend(
        y, weights, tau, np
    )
    assert intercept == y[-1]

    residual = y - intercept
    psi = _continuation.quantile_balanced_subgradient(
        residual,
        tau,
        sample_weight=weights,
    )
    assert np.all(psi <= tau)
    assert np.all(psi >= -(1.0 - tau))
    assert psi[-1] == pytest.approx(tau, rel=0.0, abs=0.0)

    X = np.column_stack([np.ones_like(y), np.linspace(-1.0, 1.0, y.size)])
    auto = mark_auto_quantile_continuation_path(
        np.asarray([0.2, 0.05], dtype=np.float64)
    )
    resolved = resolve_auto_quantile_continuation_path(
        QuantileLoss(quantile=tau),
        X,
        y,
        auto,
        sample_weight=weights,
        fit_intercept=True,
    )
    assert np.all(np.isfinite(resolved))
    assert resolved[-1] == pytest.approx(0.05, rel=0.0, abs=0.0)


def test_weighted_auto_continuation_is_invariant_to_weight_rescaling():
    X, y, weights = _data(seed=16672)
    loss = QuantileLoss(quantile=0.20)
    auto = mark_auto_quantile_continuation_path(
        np.geomspace(0.3, 0.025, 4)
    )

    first = resolve_auto_quantile_continuation_path(
        loss, X, y, auto, sample_weight=weights, fit_intercept=True
    )
    scaled = resolve_auto_quantile_continuation_path(
        loss, X, y, auto, sample_weight=11.0 * weights, fit_intercept=True
    )
    np.testing.assert_allclose(first, scaled, rtol=0.0, atol=1e-15)


def test_integer_weight_no_intercept_path_matches_explicit_replication():
    X = np.array(
        [[-1.0, 0.5], [0.2, 1.2], [0.8, -0.4], [1.5, 0.7]],
        dtype=np.float64,
    )
    y = np.array([-0.7, 0.1, 0.6, 1.3], dtype=np.float64)
    weights = np.array([1, 3, 2, 4], dtype=np.float64)
    loss = QuantileLoss(quantile=0.35)
    auto = mark_auto_quantile_continuation_path(
        np.geomspace(0.4, 0.03, 3)
    )

    weighted = resolve_auto_quantile_continuation_path(
        loss,
        X,
        y,
        auto,
        sample_weight=weights,
        fit_intercept=False,
    )
    repeats = weights.astype(int)
    replicated = resolve_auto_quantile_continuation_path(
        loss,
        np.repeat(X, repeats, axis=0),
        np.repeat(y, repeats),
        auto,
        sample_weight=None,
        fit_intercept=False,
    )
    np.testing.assert_allclose(weighted, replicated, rtol=0.0, atol=1e-15)


def test_equal_weights_with_intercept_preserve_historical_auto_path_exactly():
    X, y, _ = _data(seed=16673)
    loss = QuantileLoss(quantile=0.37)
    auto = mark_auto_quantile_continuation_path(
        np.array([0.317, 0.091, 0.027], dtype=np.float64)
    )

    unweighted = resolve_auto_quantile_continuation_path(
        loss, X, y, auto, sample_weight=None, fit_intercept=True
    )
    uniform = resolve_auto_quantile_continuation_path(
        loss,
        X,
        y,
        auto,
        sample_weight=np.full(X.shape[0], 7.0),
        fit_intercept=True,
    )

    assert unweighted is auto
    assert uniform is auto
    np.testing.assert_array_equal(unweighted, auto)
    np.testing.assert_array_equal(uniform, auto)


def test_no_intercept_auto_path_uses_zero_intercept_score():
    X, y, _ = _data(seed=16674)
    loss = QuantileLoss(quantile=0.25)
    target = 0.02
    auto = mark_auto_quantile_continuation_path(
        np.geomspace(0.2, target, 3)
    )

    observed = resolve_auto_quantile_continuation_path(
        loss, X, y, auto, sample_weight=None, fit_intercept=False
    )
    psi = np.where(y >= 0.0, 0.25, -0.75)
    lambda_start = float(np.max(np.abs(X.T @ psi / X.shape[0])))
    expected = np.geomspace(max(lambda_start, target * 1.1), target, 3)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)


def test_user_supplied_low_level_path_is_never_rewritten():
    X, y, weights = _data(seed=16675)
    loss = QuantileLoss(quantile=0.20)
    user_path = np.array([0.71, 0.31, 0.07], dtype=np.float64)

    observed = resolve_auto_quantile_continuation_path(
        loss,
        X,
        y,
        user_path,
        sample_weight=weights,
        fit_intercept=True,
    )
    assert observed is user_path
    np.testing.assert_array_equal(observed, user_path)


def test_pr164_physical_fixture_exercises_weighted_start_not_legacy_start():
    X, y, weights, _ = _pr164_gate._data()
    tau = float(_pr164_gate.Q)
    target = float(_pr164_gate.SCAD_ALPHA)
    legacy_residual = y - float(np.quantile(y, tau))
    legacy_psi = np.where(legacy_residual >= 0.0, tau, -(1.0 - tau))
    legacy_start = float(np.max(np.abs(X.T @ legacy_psi / X.shape[0])))

    auto = mark_auto_quantile_continuation_path(
        np.geomspace(max(legacy_start, target * 1.1), target, 3)
    )
    resolved = resolve_auto_quantile_continuation_path(
        QuantileLoss(quantile=tau),
        X,
        y,
        auto,
        sample_weight=weights,
        fit_intercept=True,
    )
    weighted_start = _manual_weighted_start(X, y, weights, tau)
    expected = np.geomspace(max(weighted_start, target * 1.1), target, 3)

    np.testing.assert_allclose(resolved, expected, rtol=0.0, atol=1e-15)
    assert not np.isclose(weighted_start, legacy_start, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
def test_high_level_quantile_nonconvex_path_is_marked_for_weight_alignment(
    monkeypatch, penalty
):
    X, y, weights = _data(seed=16676)
    captured = {}

    import statgpu.solvers as solvers

    def fake_solver(
        loss,
        penalty_obj,
        X_solver,
        y_solver,
        alpha_path,
        **kwargs,
    ):
        captured["alpha_path"] = alpha_path
        captured["sample_weight"] = kwargs.get("sample_weight")
        return np.zeros(X_solver.shape[1]), 0.0, 0

    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)

    model = PenalizedQuantileRegression(
        quantile=0.20,
        penalty=penalty,
        alpha=0.025,
        solver="auto",
        device="cpu",
        max_iter=20,
        tol=1e-6,
        compute_inference=False,
    )
    model.fit(X, y, sample_weight=weights)

    assert is_auto_quantile_continuation_path(captured["alpha_path"])
    np.testing.assert_allclose(
        np.asarray(captured["sample_weight"], dtype=np.float64),
        weights,
        rtol=0.0,
        atol=0.0,
    )


def test_weighted_direct_quantile_nonconvex_path_skips_legacy_host_snapshot(
    monkeypatch,
):
    X, y, weights = _data(seed=16678)
    import statgpu.solvers as solvers

    def forbidden_to_numpy(*args, **kwargs):
        raise AssertionError(
            "weighted Quantile continuation metadata must not snapshot full X/y"
        )

    def fake_solver(
        loss,
        penalty_obj,
        X_solver,
        y_solver,
        alpha_path,
        **kwargs,
    ):
        assert is_auto_quantile_continuation_path(alpha_path)
        return np.zeros(X_solver.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(_fit_mixin, "_to_numpy", forbidden_to_numpy)
    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)

    model = PenalizedQuantileRegression(
        quantile=0.20,
        penalty="scad",
        alpha=0.025,
        solver="auto",
        device="cpu",
        max_iter=20,
        tol=1e-6,
        compute_inference=False,
    )
    model.fit(X, y, sample_weight=weights)

    assert model._fitted is True
    assert model._selected_solver == "proximal_irls_cd"


def test_quantile_scad_cv_uses_fold_local_training_weights(monkeypatch):
    X, y, weights = _data(seed=16677, n=18)
    idx = np.arange(X.shape[0])
    folds = [
        (idx[9:], idx[:9]),
        (idx[:9], idx[9:]),
    ]
    seen_weights = []
    real_resolver = _prox_contract.resolve_auto_quantile_continuation_path

    def capture_resolver(loss, X_fit, y_fit, alpha_path, **kwargs):
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            seen_weights.append(np.asarray(sample_weight, dtype=np.float64).copy())
        return real_resolver(loss, X_fit, y_fit, alpha_path, **kwargs)

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        capture_resolver,
    )

    PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.20},
        penalty="scad",
        alpha_grid=np.asarray([0.025], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=16677,
        solver="auto",
        device="cpu",
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=1,
        max_iter=40,
        tol=1e-5,
    ).fit(X, y, sample_weight=weights)

    for train_idx, _ in folds:
        assert any(
            observed.shape == weights[train_idx].shape
            and np.array_equal(observed, weights[train_idx])
            for observed in seen_weights
        )
    assert any(
        observed.shape == weights.shape and np.array_equal(observed, weights)
        for observed in seen_weights
    )


def test_quantile_continuation_installer_is_idempotent_under_reload():
    from statgpu.linear_model.penalized._fit_mixin import _PenalizedFitMixin

    before = _PenalizedFitMixin._compute_lla_path
    before_wrapped = getattr(before, "__wrapped__", None)

    _path_contract.install_quantile_continuation_contract()
    assert _PenalizedFitMixin._compute_lla_path is before

    importlib.reload(_path_contract)
    after = _PenalizedFitMixin._compute_lla_path
    assert after is before
    assert getattr(after, _path_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped
