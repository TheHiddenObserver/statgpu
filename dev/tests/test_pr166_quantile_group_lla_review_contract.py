"""Fresh-review contracts for Quantile group-penalty routing in PR #166."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized import _fit_mixin
from statgpu.linear_model.penalized import (
    _quantile_continuation_contract as _continuation_contract,
)
from statgpu.linear_model.penalized import (
    _quantile_group_lla_contract as _group_lla_contract,
)
from statgpu.solvers import _quantile_group_proximal_irls_lla as group_solver
from statgpu.solvers._quantile_continuation import (
    is_auto_quantile_continuation_path,
)


GROUPS = [[0, 1], [2, 3]]
Q = 0.35


def _data(seed=166301, n=24):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    beta = np.array([0.8, -0.45, 0.3, 0.15], dtype=np.float64)
    y = (0.2 + X @ beta + rng.laplace(scale=0.18, size=n)).astype(np.float64)
    weights = np.linspace(0.4, 1.9, n, dtype=np.float64)
    rng.shuffle(weights)
    return X, y, weights


def _penalty_kwargs(kind):
    result = {"groups": GROUPS}
    if kind == "group_scad":
        result["a"] = 3.7
    else:
        result["gamma"] = 3.0
    return result


def _balanced_psi(residual, tau, weights=None):
    residual = np.asarray(residual, dtype=np.float64)
    positive = residual > 0.0
    negative = residual < 0.0
    zero = ~(positive | negative)
    if weights is None:
        positive_mass = float(np.sum(positive))
        negative_mass = float(np.sum(negative))
        zero_mass = float(np.sum(zero))
    else:
        weights = np.asarray(weights, dtype=np.float64)
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


def _manual_weighted_path(X, y, weights, target_alpha, n_cont=3):
    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    w_sorted = weights[order]
    cutoff = Q * float(np.sum(w_sorted))
    idx = int(np.searchsorted(np.cumsum(w_sorted), cutoff, side="left"))
    intercept = float(y_sorted[min(idx, y_sorted.size - 1)])
    residual = y - intercept
    psi = _balanced_psi(residual, Q, weights)
    score = X.T @ (weights * psi) / float(np.sum(weights))
    lam = max(
        float(np.linalg.norm(score[np.asarray(group, dtype=int)]))
        / np.sqrt(len(group))
        for group in GROUPS
    )
    return np.geomspace(max(lam, target_alpha * 1.1), target_alpha, n_cont)


def test_quantile_group_lasso_bypasses_gaussian_block_cd(monkeypatch):
    """Convex Quantile Group Lasso remains on loss-gradient FISTA."""
    X, y, weights = _data()
    import statgpu.solvers as solvers

    seen = {}

    def forbidden_block(*args, **kwargs):
        raise AssertionError("Quantile Group Lasso must not use Gaussian block CD")

    def fake_fista(loss, penalty, X_fit, y_fit, **kwargs):
        seen["loss"] = getattr(loss, "name", None)
        seen["penalty"] = getattr(penalty, "name", None)
        return np.zeros(X_fit.shape[1], dtype=np.float64), 1

    monkeypatch.setattr(_fit_mixin._PenalizedFitMixin, "_block_cd_group_lasso", forbidden_block)
    monkeypatch.setattr(
        _fit_mixin._PenalizedFitMixin, "_block_cd_group_lasso_gpu", forbidden_block
    )
    monkeypatch.setattr(solvers, "fista_solver", fake_fista)

    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_lasso",
        penalty_kwargs={"groups": GROUPS},
        alpha=0.04,
        solver="auto",
        device="cpu",
        fit_intercept=False,
        max_iter=20,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "fista"
    assert seen["loss"] == "quantile"
    assert seen["penalty"] == "_group_lasso_generic"


@pytest.mark.parametrize("kind", ["group_scad", "group_mcp"])
def test_quantile_group_nonconvex_auto_uses_group_proximal_irls_lla(monkeypatch, kind):
    X, y, weights = _data(seed=166302)
    captured = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        captured["loss"] = getattr(loss, "name", None)
        captured["penalty"] = getattr(penalty, "name", None)
        captured["alpha_path"] = np.asarray(alpha_path, dtype=np.float64).copy()
        captured["sample_weight"] = np.asarray(
            kwargs.get("sample_weight"), dtype=np.float64
        ).copy()
        captured["fit_intercept"] = bool(kwargs.get("fit_intercept"))
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", fake_solver
    )

    target = 0.04
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha=target,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        max_iter=80,
        tol=1e-7,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "group_proximal_irls_lla"
    assert captured["loss"] == "quantile"
    assert captured["penalty"] == kind
    assert captured["fit_intercept"] is True
    np.testing.assert_array_equal(captured["sample_weight"], weights)
    np.testing.assert_allclose(
        captured["alpha_path"],
        _manual_weighted_path(X, y, weights, target),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize("kind", ["group_scad", "group_mcp"])
def test_quantile_group_auto_does_not_build_discarded_scalar_continuation(
    monkeypatch, kind
):
    X, y, _ = _data(seed=166313)

    def forbidden_scalar_path(*args, **kwargs):
        raise AssertionError("Group Quantile auto route must not build scalar continuation")

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_compute_lla_path",
        forbidden_scalar_path,
    )
    monkeypatch.setattr(
        group_solver,
        "quantile_group_proximal_irls_lla_solver",
        fake_solver,
    )

    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha=0.04,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        max_iter=80,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "group_proximal_irls_lla"


@pytest.mark.parametrize("kind", ["group_scad", "group_mcp"])
def test_quantile_group_unweighted_continuation_uses_group_alpha_scale(
    monkeypatch, kind
):
    X, y, _ = _data(seed=166312)
    captured = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        captured["alpha_path"] = np.asarray(alpha_path, dtype=np.float64).copy()
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", fake_solver
    )

    target = 0.04
    PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha=target,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        max_iter=80,
        tol=1e-7,
    ).fit(X, y)

    intercept = float(np.sort(y, kind="stable")[max(int(np.ceil(Q * len(y))) - 1, 0)])
    residual = y - intercept
    psi = _balanced_psi(residual, Q)
    score = X.T @ psi / float(X.shape[0])
    lam = max(
        float(np.linalg.norm(score[np.asarray(group, dtype=int)]))
        / np.sqrt(len(group))
        for group in GROUPS
    )
    expected = np.geomspace(max(lam, target * 1.1), target, 3)
    np.testing.assert_allclose(
        captured["alpha_path"], expected, rtol=0.0, atol=1e-15
    )


@pytest.mark.parametrize("kind", ["group_scad", "group_mcp"])
def test_quantile_group_nonconvex_explicit_fista_stays_explicit(monkeypatch, kind):
    """An explicit solver request must not be converted into the auto route."""
    X, y, weights = _data(seed=166307)
    import statgpu.solvers as solvers

    seen = {"fista": 0}

    def forbidden_auto(*args, **kwargs):
        raise AssertionError("explicit Quantile group FISTA must not enter Proximal IRLS-LLA")

    def fake_fista(loss, penalty, X_fit, y_fit, **kwargs):
        seen["fista"] += 1
        assert getattr(loss, "name", None) == "quantile"
        assert getattr(penalty, "name", None) == kind
        return np.zeros(X_fit.shape[1], dtype=np.float64), 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", forbidden_auto
    )
    monkeypatch.setattr(solvers, "fista_solver", fake_fista)

    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha=0.04,
        solver="fista",
        device="cpu",
        fit_intercept=False,
        max_iter=20,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "fista"
    assert seen["fista"] == 1


def test_group_quantile_public_fit_intercept_replacement_reaches_solver(monkeypatch):
    X, y, weights = _data(seed=166309)
    captured = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        captured["fit_intercept"] = kwargs["fit_intercept"]
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", fake_solver
    )
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_scad",
        penalty_kwargs=_penalty_kwargs("group_scad"),
        alpha=0.04,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        max_iter=40,
        tol=1e-6,
    )
    model.fit_intercept = False
    model.fit(X, y, sample_weight=weights)

    assert captured["fit_intercept"] is False
    assert model._effective_intercept is False
    assert model.intercept_ == 0.0


@pytest.mark.parametrize("kind", ["group_scad", "group_mcp"])
def test_quantile_group_nonconvex_actual_cpu_fit_runs_full_auto_route(kind):
    X, y, weights = _data(seed=166305, n=20)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha=0.04,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=80,
        tol=1e-5,
        max_lla_iters=9,
        lla_tol=1e-5,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "group_proximal_irls_lla"
    assert model.n_iter_ >= 1
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert np.all(np.isfinite(model.predict(X)))


def test_quantile_group_scad_cv_uses_fold_local_weights_and_auto_route(monkeypatch):
    X, y, weights = _data(seed=166303, n=20)
    idx = np.arange(X.shape[0])
    folds = [(idx[10:], idx[:10]), (idx[:10], idx[10:])]
    seen_weights = []

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            seen_weights.append(np.asarray(sample_weight, dtype=np.float64).copy())
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", fake_solver
    )

    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha_grid=np.asarray([0.04], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="auto",
        device="cpu",
        max_iter=60,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    assert cv.alpha_ == pytest.approx(0.04)
    assert cv.estimator_._selected_solver == "group_proximal_irls_lla"
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


def test_quantile_group_scad_explicit_fista_cv_stays_explicit(monkeypatch):
    """Explicit-FISTA CV children and final refit stay explicit FISTA."""
    X, y, weights = _data(seed=166308, n=18)
    idx = np.arange(X.shape[0])
    folds = [(idx[9:], idx[:9]), (idx[:9], idx[9:])]
    import statgpu.solvers as solvers

    seen = {"fista": 0}

    def forbidden_auto(*args, **kwargs):
        raise AssertionError("explicit Quantile group FISTA CV must not enter auto LLA")

    def fake_fista(loss, penalty, X_fit, y_fit, **kwargs):
        seen["fista"] += 1
        return np.zeros(X_fit.shape[1], dtype=np.float64), 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", forbidden_auto
    )
    monkeypatch.setattr(solvers, "fista_solver", fake_fista)

    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha_grid=np.asarray([0.04], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="fista",
        device="cpu",
        max_iter=30,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    assert cv.alpha_ == pytest.approx(0.04)
    assert cv.estimator_._selected_solver == "fista"
    assert seen["fista"] >= 3


def test_quantile_group_scad_actual_cpu_cv_runs_full_auto_route():
    X, y, weights = _data(seed=166306, n=18)
    idx = np.arange(X.shape[0])
    folds = [(idx[9:], idx[:9]), (idx[:9], idx[9:])]
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha_grid=np.asarray([0.04], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="auto",
        device="cpu",
        max_iter=60,
        tol=1e-5,
    ).fit(X, y, sample_weight=weights)

    assert cv.alpha_ == pytest.approx(0.04)
    assert cv.estimator_._selected_solver == "group_proximal_irls_lla"
    assert np.all(np.isfinite(cv.coef_))
    assert np.isfinite(cv.intercept_)
    assert np.all(np.isfinite(cv.cv_results_["all_scores"]))


def test_nonuniform_quantile_path_metadata_avoids_legacy_full_host_snapshot(monkeypatch):
    X, y, weights = _data(seed=166304)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="scad",
        alpha=0.04,
        solver="auto",
        device="cpu",
        max_iter=40,
    )
    model._penalty = model._resolve_penalty()
    model._loss = model._resolve_loss()

    original = getattr(
        model._compute_lla_path, "_statgpu_original", None
    ) or getattr(model._compute_lla_path, "__wrapped__", None)
    assert original is not None

    def forbidden_legacy(*args, **kwargs):
        raise AssertionError("legacy full-host Quantile path generator was called")

    monkeypatch.setattr(_fit_mixin, "_to_numpy", forbidden_legacy)
    token = _continuation_contract._QUANTILE_SAMPLE_WEIGHT.set(weights)
    try:
        path, _, _ = model._compute_lla_path(X, y, X.shape[1], "quantile")
    finally:
        _continuation_contract._QUANTILE_SAMPLE_WEIGHT.reset(token)

    assert is_auto_quantile_continuation_path(path)


def test_quantile_group_lla_installer_is_idempotent_under_reload():
    before = _fit_mixin._PenalizedFitMixin._fit_loss_backend
    wrapped = getattr(before, "__wrapped__", None)

    importlib.reload(_group_lla_contract)
    after = _fit_mixin._PenalizedFitMixin._fit_loss_backend

    assert after is before
    assert getattr(after, _group_lla_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is wrapped
