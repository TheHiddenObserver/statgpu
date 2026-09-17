"""Inner-solver correctness contracts for PR #166 Quantile LLA."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import GroupSCADPenalty
from statgpu.solvers import _fista as fista_module
from statgpu.solvers import _fista_lla_group_contract as lla_contract


GROUPS = [[0, 1], [2, 3]]
Q = 0.35


class _FakeQuantileLoss:
    name = "quantile"
    has_hessian = False
    _tau = Q

    def __init__(self):
        self.irls_calls = 0

    def irls(
        self,
        X,
        y,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        self.irls_calls += 1
        if init_coef is None:
            coef = np.zeros(X.shape[1], dtype=np.float64)
        else:
            coef = np.asarray(init_coef, dtype=np.float64).copy()
        return coef, 3


def test_nonzero_quantile_group_surrogate_uses_irls_weighted_squared_fista(monkeypatch):
    """Active Group LLA surrogates use IRLS WLS + Group FISTA, not pinball fixed-step FISTA."""
    loss = _FakeQuantileLoss()
    penalty = GroupSCADPenalty(alpha=0.3, a=3.7, groups=GROUPS)
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.2, -0.1, 0.3, -0.2], dtype=np.float64)
    weights = np.asarray([0.5, 0.8, 1.1, 1.6], dtype=np.float64)
    seen = {}

    def forbidden_base(*args, **kwargs):
        raise AssertionError("Quantile LLA must not use the historical fixed-step base loop")

    def fake_fista(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        seen["loss_name"] = getattr(loss_arg, "name", None)
        seen["penalty_name"] = getattr(penalty_arg, "name", None)
        seen["sample_weight"] = kwargs.get("sample_weight")
        seen["max_iter"] = kwargs.get("max_iter")
        seen["tol"] = kwargs.get("tol")
        seen["X"] = np.asarray(X_arg, dtype=np.float64).copy()
        seen["y"] = np.asarray(y_arg, dtype=np.float64).copy()
        # Keep the warm start unchanged so the outer IRLS loop closes in one step.
        return np.asarray(kwargs["init_coef"], dtype=np.float64).copy(), 4

    monkeypatch.setattr(lla_contract, "_base_fista_lla_path", forbidden_base)
    monkeypatch.setattr(fista_module, "fista_solver", fake_fista)

    coef, intercept, n_iter = lla_contract.fista_lla_path(
        loss,
        penalty,
        X,
        y,
        alpha_path=[0.3],
        max_lla_per_step=1,
        max_iter=17,
        tol=1e-7,
        fit_intercept=False,
        sample_weight=weights,
    )

    assert seen["loss_name"] == "squared_error"
    assert seen["penalty_name"] == "adaptive_group_lasso"
    # Analytic weights are absorbed into the IRLS quadratic design/response;
    # the inner squared-error FISTA therefore receives no second weight vector.
    assert seen["sample_weight"] is None
    assert seen["max_iter"] == 250
    assert seen["tol"] == pytest.approx(1e-7)
    assert not np.array_equal(seen["X"], X)
    assert not np.array_equal(seen["y"], y)
    assert loss.irls_calls == 0
    assert n_iter == 4
    np.testing.assert_array_equal(coef, np.zeros(4))
    assert intercept == 0.0


def test_zero_quantile_group_surrogate_closes_with_irls(monkeypatch):
    """A completely flat Group SCAD surrogate is ordinary Quantile regression."""
    loss = _FakeQuantileLoss()
    penalty = GroupSCADPenalty(alpha=0.04, a=3.7, groups=GROUPS)
    X = np.eye(4, dtype=np.float64)
    y = np.zeros(4, dtype=np.float64)
    init = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    def forbidden_fista(*args, **kwargs):
        raise AssertionError("zero Group LLA surrogate must close through IRLS")

    monkeypatch.setattr(fista_module, "fista_solver", forbidden_fista)

    coef, intercept, n_iter = lla_contract.fista_lla_path(
        loss,
        penalty,
        X,
        y,
        alpha_path=[0.04],
        max_lla_per_step=1,
        max_iter=23,
        tol=1e-7,
        fit_intercept=False,
        init_coef=init,
    )

    assert loss.irls_calls == 1
    assert n_iter == 3
    np.testing.assert_array_equal(coef, init)
    assert intercept == 0.0


def _fixture(seed=166401, n=64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    beta = np.array([0.85, -0.42, 0.28, 0.16], dtype=np.float64)
    y = 0.25 + X @ beta + rng.laplace(scale=0.16, size=n)
    weights = np.linspace(0.45, 1.85, n, dtype=np.float64)
    rng.shuffle(weights)
    return X, y.astype(np.float64), weights


def _pinball(y, eta, weights):
    residual = y - eta
    values = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    return float(np.average(values, weights=weights))


def _fit_group_scad_fixture(X, y, weights):
    return PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=500,
        tol=1e-7,
        max_lla_iters=18,
        lla_tol=1e-7,
    ).fit(X, y, sample_weight=weights)


def test_quantile_group_wls_surrogates_do_not_accept_max_iter_as_success():
    """The physical-style Group path must close convex WLS surrogates without warnings."""
    X, y, weights = _fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _fit_group_scad_fixture(X, y, weights)

    relevant = [
        item
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
        and "loss=squared_error, penalty=adaptive_group_lasso" in str(item.message)
    ]
    assert relevant == []


def test_flat_target_group_scad_closes_to_weighted_quantile_irls():
    """The physical-gate fixture has a flat final SCAD surrogate; close it exactly."""
    X, y, weights = _fixture()
    group = _fit_group_scad_fixture(X, y, weights)

    reference = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="l2",
        alpha=0.0,
        solver="irls",
        device="cpu",
        compute_inference=False,
        max_iter=500,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    group_norms = np.asarray(
        [np.linalg.norm(group.coef_[indices]) for indices in GROUPS],
        dtype=np.float64,
    )
    flat_threshold = 3.7 * 0.04 * np.sqrt(2.0)
    assert np.all(group_norms > flat_threshold)

    group_fit = _pinball(y, X @ group.coef_ + group.intercept_, weights)
    ref_fit = _pinball(y, X @ reference.coef_ + reference.intercept_, weights)
    assert abs(group_fit - ref_fit) <= 2e-6
    np.testing.assert_allclose(group.coef_, reference.coef_, rtol=0.0, atol=5e-5)
    assert group.intercept_ == pytest.approx(reference.intercept_, abs=5e-5)
