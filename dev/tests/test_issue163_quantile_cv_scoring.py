"""Issue #163 regressions for Quantile CV scoring at the requested tau."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    AdaptiveL1Penalty,
    ElasticNetPenalty,
    L2Penalty,
    SCADPenalty,
)


def _data(seed=16321, n=64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)).astype(np.float64)
    noise = rng.exponential(scale=0.22, size=n) - 0.22
    y = (0.3 + X @ np.array([0.7, -0.25]) + noise).astype(np.float64)
    folds = [
        (np.arange(0, n // 2), np.arange(n // 2, n)),
        (np.arange(n // 2, n), np.arange(0, n // 2)),
    ]
    return X, y, folds


def _balanced_psi(residual, tau, sample_weight=None):
    residual = np.asarray(residual, dtype=np.float64)
    positive = residual > 0.0
    negative = residual < 0.0
    zero = ~(positive | negative)
    if sample_weight is None:
        positive_mass = float(np.sum(positive))
        negative_mass = float(np.sum(negative))
        zero_mass = float(np.sum(zero))
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
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


def _pinball(y, eta, tau, sample_weight=None):
    u = np.asarray(y) - np.asarray(eta)
    values = np.where(u >= 0.0, tau * u, (tau - 1.0) * u)
    if sample_weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=np.asarray(sample_weight)))


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.array([1.0, -0.1, 1.0, 1.0]),
        np.array([1.0, np.nan, 1.0, 1.0]),
        np.zeros(4, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    ],
)
def test_quantile_cv_score_rejects_invalid_sample_weight(sample_weight):
    X = np.array(
        [[-1.0, 0.2], [0.0, -0.1], [0.5, 0.4], [1.0, -0.3]],
        dtype=np.float64,
    )
    y = np.array([-0.4, 0.1, 0.35, 0.8], dtype=np.float64)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=np.asarray([0.02], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    with pytest.raises(ValueError, match="sample_weight"):
        model.score(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("weighted", [False, True])
def test_quantile_cv_general_scores_use_requested_tau(weighted):
    X, y, folds = _data()
    tau = 0.2
    alpha = 0.035
    weights = (
        np.linspace(0.45, 1.8, X.shape[0], dtype=np.float64)
        if weighted
        else None
    )

    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="l2",
        alpha_grid=np.array([alpha], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=500,
        tol=1e-9,
    )
    observed = cv._compute_cv_scores(
        X,
        y,
        np.array([alpha], dtype=np.float64),
        Device.CPU,
        folds,
        sample_weight=weights,
        max_iter=500,
        tol=1e-9,
        strict=True,
    )

    expected = []
    median_scores = []
    for train_idx, val_idx in folds:
        train_weight = weights[train_idx] if weights is not None else None
        val_weight = weights[val_idx] if weights is not None else None
        model = PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": tau},
            penalty="l2",
            alpha=alpha,
            solver="irls",
            device="cpu",
            max_iter=500,
            tol=1e-9,
        ).fit(X[train_idx], y[train_idx], sample_weight=train_weight)
        eta = X[val_idx] @ model.coef_ + model.intercept_
        expected.append(_pinball(y[val_idx], eta, tau, val_weight))
        median_scores.append(_pinball(y[val_idx], eta, 0.5, val_weight))

    np.testing.assert_allclose(
        observed[:, 0], np.asarray(expected), rtol=2e-8, atol=2e-10
    )
    assert float(np.max(np.abs(observed[:, 0] - np.asarray(median_scores)))) > 1e-4

    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    assert contract._QUANTILE_CV_LEVEL.get() is None


@pytest.mark.parametrize("penalty", ["none", "null", "", None])
def test_quantile_cv_rejects_non_tunable_no_penalty_alias_before_grid_work(
    monkeypatch, penalty
):
    X, y, _ = _data(seed=16346, n=32)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty=penalty,
        alpha_grid=np.asarray([0.1, 0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
    )

    def forbidden_grid(*args, **kwargs):
        raise AssertionError("no-penalty CV must fail before grid work")

    monkeypatch.setattr(model, "_generate_alpha_grid", forbidden_grid)
    with pytest.raises(ValueError, match="non-tunable"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.alpha_ is None
    assert model.estimator_ is None


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        ([], "at least one fold"),
        ([(np.array([], dtype=int), np.array([0, 1]))], "non-empty"),
        ([(np.array([0, 1]), np.array([], dtype=int))], "non-empty"),
        ([(np.array([0, 0, 1]), np.array([2, 3]))], "duplicates"),
        ([(np.array([0, 1]), np.array([1, 2]))], "disjoint"),
        ([(np.array([-1, 0]), np.array([1, 2]))], "out of bounds"),
        ([(np.array([0, 1]), np.array([2, 99]))], "out of bounds"),
        ([(np.array([0.5, 1.0]), np.array([2, 3]))], "integers"),
        ([(np.array([True, False]), np.array([2, 3]))], "integers"),
        ([(np.array(["0", "1"]), np.array([2, 3]))], "integers"),
    ],
)
def test_quantile_cv_rejects_malformed_custom_folds_before_candidate_work(
    monkeypatch, folds, message
):
    X, y, _ = _data(seed=16341, n=24)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
    )

    def forbidden_device(*args, **kwargs):
        raise AssertionError("CV numerical routing must not start")

    monkeypatch.setattr(model, "_effective_cv_device", forbidden_device)
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)

    assert model._fitted is False
    assert model.alpha_ is None
    assert model.estimator_ is None


def test_quantile_cv_reuses_one_shot_custom_splits_across_refits():
    X, y, _ = _data(seed=16338, n=60)
    idx = np.arange(X.shape[0])
    folds = [
        (np.concatenate([idx[:start], idx[stop:]]), idx[start:stop])
        for start, stop in ((0, 20), (20, 40), (40, 60))
    ]
    iterations = []

    def one_shot():
        iterations.append(1)
        if len(iterations) > 1:
            raise AssertionError("custom split generator was consumed twice")
        yield from folds

    generator = one_shot()
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=3,
        cv_splits=generator,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )

    cv.fit(X, y)
    first_coef = np.asarray(cv.coef_, dtype=np.float64).copy()
    first_score = float(cv.best_score_)
    cv.fit(X, y)

    assert iterations == [1]
    assert cv.cv_splits is generator
    np.testing.assert_allclose(cv.coef_, first_coef, rtol=0.0, atol=1e-12)
    assert cv.best_score_ == pytest.approx(first_score, rel=0.0, abs=1e-12)
    assert cv.cv_results_["device_sizing_fold_count"] == len(folds)


def test_quantile_cv_pickle_materializes_one_shot_custom_splits_once():
    import pickle

    X, _, _ = _data(seed=16352, n=48)
    idx = np.arange(X.shape[0])
    folds = [
        (np.setdiff1d(idx, val, assume_unique=True), val)
        for val in np.array_split(idx, 3)
    ]
    iterations = []

    def one_shot():
        iterations.append(1)
        if len(iterations) > 1:
            raise AssertionError("custom split generator was consumed twice")
        yield from folds

    generator = one_shot()
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=3,
        cv_splits=generator,
        solver="auto",
        device="cpu",
    )

    restored = pickle.loads(pickle.dumps(cv))

    assert iterations == [1]
    assert cv.cv_splits is generator
    assert isinstance(restored.cv_splits, list)
    assert len(restored.cv_splits) == len(folds)
    assert restored._cv_split_source is None
    assert restored._cv_split_snapshot is None
    assert restored._fitted is False


def test_quantile_cv_clone_materializes_one_shot_custom_splits_once():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    X, _, _ = _data(seed=16339, n=48)
    idx = np.arange(X.shape[0])
    folds = [
        (np.setdiff1d(idx, val, assume_unique=True), val)
        for val in np.array_split(idx, 3)
    ]
    iterations = []

    def one_shot():
        iterations.append(1)
        if len(iterations) > 1:
            raise AssertionError("custom split generator was consumed twice")
        yield from folds

    generator = one_shot()
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=3,
        cv_splits=generator,
        solver="auto",
        device="cpu",
    )
    cloned = clone(cv)

    assert sklearn is not None
    assert iterations == [1]
    assert cv.cv_splits is generator
    assert isinstance(cloned.cv_splits, list)
    assert len(cloned.cv_splits) == len(folds)
    assert cloned._fitted is False


def test_quantile_two_stage_does_not_repeat_strict_grid_when_all_candidates_refined(
    monkeypatch,
):
    X, y, folds = _data(seed=16355, n=48)
    alpha_grid = np.asarray([0.04, 0.02], dtype=np.float64)
    calls = []

    def fake_scores(
        self,
        X_arg,
        y_arg,
        alpha_grid_arg,
        cv_device,
        folds_arg,
        *,
        sample_weight=None,
        max_iter=None,
        tol=None,
        strict=True,
    ):
        alpha_grid_arg = np.asarray(alpha_grid_arg, dtype=np.float64)
        calls.append((bool(strict), alpha_grid_arg.copy()))
        if not strict:
            return np.zeros((len(folds_arg), len(alpha_grid_arg)), dtype=np.float64)
        return np.full((len(folds_arg), len(alpha_grid_arg)), np.nan)

    monkeypatch.setattr(PenalizedGLM_CV, "_compute_cv_scores", fake_scores)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="scad",
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=len(alpha_grid),
        max_iter=80,
        tol=1e-6,
    )

    with pytest.raises(RuntimeError, match="no finite candidate score"):
        model.fit(X, y)

    assert [strict for strict, _ in calls] == [False, True]
    np.testing.assert_array_equal(calls[1][1], alpha_grid)


def test_quantile_two_stage_falls_back_to_full_strict_grid_when_refined_set_fails(
    monkeypatch,
):
    from types import SimpleNamespace

    X, y, folds = _data(seed=16340, n=56)
    alpha_grid = np.asarray([0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
    calls = []

    def fake_scores(
        self,
        X_arg,
        y_arg,
        alpha_grid_arg,
        cv_device,
        folds_arg,
        *,
        sample_weight=None,
        max_iter=None,
        tol=None,
        strict=True,
    ):
        alpha_grid_arg = np.asarray(alpha_grid_arg, dtype=np.float64)
        calls.append((bool(strict), alpha_grid_arg.copy()))
        out = np.full((len(folds_arg), len(alpha_grid_arg)), np.nan)
        if not strict:
            stage1_mean = {
                0.07: 5.0,
                0.06: 4.0,
                0.05: 3.0,
                0.04: 0.0,
                0.03: 3.0,
                0.02: 4.0,
                0.01: 5.0,
            }
            for j, alpha in enumerate(alpha_grid_arg):
                out[:, j] = stage1_mean[float(alpha)]
            return out

        if len(alpha_grid_arg) == len(alpha_grid):
            # Only an alpha omitted by the approximate shortlist is strictly
            # usable; full-grid recovery must discover it.
            j = int(np.flatnonzero(np.isclose(alpha_grid_arg, 0.06))[0])
            out[:, j] = 0.125
        return out

    def fake_refit(self, X_arg, y_arg, best_alpha, sample_weight=None):
        return SimpleNamespace(
            coef_=np.zeros(X_arg.shape[1], dtype=np.float64),
            intercept_=0.0,
        )

    monkeypatch.setattr(PenalizedGLM_CV, "_compute_cv_scores", fake_scores)
    monkeypatch.setattr(PenalizedGLM_CV, "_refit_best", fake_refit)

    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="scad",
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=1,
        max_iter=80,
        tol=1e-6,
    )

    with pytest.warns(
        RuntimeWarning,
        match="retrying the full alpha grid with strict solves",
    ):
        model.fit(X, y)

    assert [strict for strict, _ in calls] == [False, True, True]
    assert len(calls[1][1]) < len(alpha_grid)
    np.testing.assert_array_equal(calls[2][1], alpha_grid)
    assert model.alpha_ == pytest.approx(0.06)
    assert np.all(model.cv_results_["refined_mask"])
    assert model.cv_results_["mean_score"][1] == pytest.approx(0.125)


def test_quantile_auto_alpha_grid_uses_empirical_pinball_intercept():
    X = np.arange(5, dtype=np.float64)[:, None]
    y = np.asarray([0.0, 1.0, 4.0, 8.0, 9.0], dtype=np.float64)
    tau = 0.3
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="l1",
        alpha_grid=None,
        n_alphas=3,
        cv=2,
        solver="auto",
        device="cpu",
    )

    grid = model._generate_alpha_grid(X, y)

    lower_intercept = 1.0
    residual = y - lower_intercept
    psi = _balanced_psi(residual, tau)
    expected = float(np.max(np.abs(X.T @ psi / X.shape[0])))

    linear_intercept = float(np.quantile(y, tau))
    linear_residual = y - linear_intercept
    linear_psi = np.where(
        linear_residual >= 0.0,
        tau,
        -(1.0 - tau),
    )
    legacy_linear = float(np.max(np.abs(X.T @ linear_psi / X.shape[0])))

    assert lower_intercept != pytest.approx(linear_intercept)
    assert grid[0] == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert abs(grid[0] - legacy_linear) > 1e-3


def test_quantile_auto_alpha_grid_is_translation_invariant_with_intercept():
    X = np.array(
        [[-2.0, 0.5], [-1.0, 1.5], [0.0, -0.5], [1.0, 2.0], [3.0, -1.0]],
        dtype=np.float64,
    )
    y = np.asarray([0.0, 1.0, 4.0, 8.0, 9.0], dtype=np.float64)
    kwargs = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l1",
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )
    base = PenalizedGLM_CV(**kwargs)._generate_alpha_grid(X, y)
    shifted = PenalizedGLM_CV(**kwargs)._generate_alpha_grid(
        X + np.asarray([17.0, -9.0]),
        y,
    )
    np.testing.assert_allclose(shifted, base, rtol=0.0, atol=1e-15)


def test_quantile_auto_alpha_grid_uniform_weights_is_bitwise_unweighted():
    X, y, _ = _data(seed=16347, n=40)
    kwargs = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.23},
        penalty="l1",
        alpha_grid=None,
        n_alphas=5,
        cv=2,
        solver="auto",
        device="cpu",
    )
    unweighted = PenalizedGLM_CV(**kwargs)._generate_alpha_grid(X, y)
    weighted = PenalizedGLM_CV(**kwargs)._generate_alpha_grid(
        X,
        y,
        sample_weight=np.full(X.shape[0], 3.5, dtype=np.float64),
    )

    np.testing.assert_array_equal(weighted, unweighted)


def test_quantile_auto_alpha_grid_uses_weighted_pinball_zero_score():
    X, y, _ = _data(seed=16342, n=40)
    tau = 0.23
    weights = np.linspace(0.35, 1.9, X.shape[0], dtype=np.float64)

    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="l2",
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )

    grid = model._generate_alpha_grid(X, y, sample_weight=weights)

    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    w_sorted = weights[order]
    cutoff = tau * float(np.sum(w_sorted))
    q_index = min(
        int(np.searchsorted(np.cumsum(w_sorted), cutoff, side="left")),
        len(y_sorted) - 1,
    )
    intercept = float(y_sorted[q_index])
    residual = y - intercept
    psi = _balanced_psi(residual, tau, weights)
    score = X.T @ (weights * psi) / float(np.sum(weights))
    expected = float(np.max(np.abs(score)))

    assert grid[0] == pytest.approx(expected, rel=0.0, abs=1e-14)
    assert not np.isclose(
        grid[0],
        float(np.max(np.abs(X.T @ (weights * residual) / np.sum(weights)))),
        rtol=1e-4,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("penalty_name", "penalty_kwargs"),
    [
        ("group_scad", {"a": 3.7}),
        ("group_mcp", {"gamma": 3.0}),
    ],
)
def test_quantile_group_nonconvex_auto_alpha_grid_uses_group_public_scale(
    penalty_name, penalty_kwargs
):
    X, y, _ = _data(seed=16343, n=44)
    tau = 0.31
    groups = [[0, 1]]
    kwargs = {"groups": groups, **penalty_kwargs}
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty=penalty_name,
        penalty_kwargs=kwargs,
        alpha_grid=None,
        n_alphas=3,
        cv=2,
        solver="auto",
        device="cpu",
    )

    grid = model._generate_alpha_grid(X, y)

    intercept = float(np.sort(y, kind="stable")[max(int(np.ceil(tau * len(y))) - 1, 0)])
    residual = y - intercept
    psi = _balanced_psi(residual, tau)
    score = X.T @ psi / float(X.shape[0])
    expected = float(np.linalg.norm(score)) / np.sqrt(score.size)
    assert grid[0] == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_quantile_cv_public_fold_count_replacement_is_authoritative():
    X, y, _ = _data(seed=16326, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    cv.cv = 3
    cv.fit(X, y)

    assert cv._cv == 3
    assert np.asarray(cv.cv_results_["all_scores"]).shape[0] == 3


def test_quantile_cv_public_n_alphas_replacement_controls_generated_grid():
    X, y, _ = _data(seed=16327, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=None,
        n_alphas=5,
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    cv.n_alphas = 3
    cv.fit(X, y)

    assert cv._n_alphas == 3
    assert len(cv.alpha_grid_) == 3


def test_quantile_cv_public_two_stage_controls_are_authoritative():
    X, y, folds = _data(seed=16328, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=np.asarray([0.04, 0.02], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        cv_strategy="strict",
        acknowledge_approx=False,
        refine_top_k=2,
        max_iter=300,
        tol=1e-8,
    )
    cv.cv_strategy = "two_stage"
    cv.acknowledge_approx = True
    cv.refine_top_k = 1
    cv.fit(X, y)

    assert cv._cv_strategy == "two_stage"
    assert cv._acknowledge_approx is True
    assert cv._refine_top_k == 1
    assert cv.cv_strategy_ == "two_stage"
    assert "all_scores_stage1" in cv.cv_results_


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "l1_ratio must be a finite real number in \\[0, 1\\]"),
        ("0.4", "l1_ratio must be a finite real number in \\[0, 1\\]"),
        (np.nan, "l1_ratio must be a finite real number in \\[0, 1\\]"),
        (-0.1, "l1_ratio must be a finite real number in \\[0, 1\\]"),
        (1.1, "l1_ratio must be a finite real number in \\[0, 1\\]"),
    ],
)
def test_quantile_elasticnet_public_l1_ratio_rejects_invalid_refit_before_grid(
    monkeypatch, value, message
):
    X, y, _ = _data(seed=16353, n=48)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="elasticnet",
        l1_ratio=0.5,
        alpha_grid=None,
        n_alphas=3,
        cv=2,
        solver="auto",
        device="cpu",
    )
    cv.l1_ratio = value

    def forbidden_grid(*args, **kwargs):
        raise AssertionError("invalid l1_ratio must fail before grid work")

    monkeypatch.setattr(cv, "_generate_alpha_grid", forbidden_grid)
    with pytest.raises(ValueError, match=message):
        cv.fit(X, y)

    assert cv._fitted is False
    assert cv.alpha_ is None
    assert cv.estimator_ is None


def test_quantile_elasticnet_public_l1_ratio_replacement_is_authoritative():
    X, y, _ = _data(seed=16354, n=56)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="elasticnet",
        l1_ratio=0.8,
        alpha_grid=np.asarray([0.04], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-7,
    )
    cv.l1_ratio = 0.25
    cv.fit(X, y)

    assert cv.l1_ratio == pytest.approx(0.25)
    assert cv.estimator_._penalty.l1_ratio == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("cv", True, "cv must be a positive integer"),
        ("cv", 2.5, "cv must be a positive integer"),
        ("cv", 1, "cv must be an integer greater than or equal to 2"),
        ("n_alphas", True, "n_alphas must be a positive integer"),
        ("n_alphas", "3", "n_alphas must be a positive integer"),
        ("cv_strategy", 2, "cv_strategy must be either"),
        ("cv_strategy", "fast", "cv_strategy must be either"),
        ("acknowledge_approx", "False", "acknowledge_approx must be boolean"),
        ("refine_top_k", True, "refine_top_k must be a positive integer"),
        ("refine_top_k", 0, "refine_top_k must be a positive integer"),
    ],
)
def test_quantile_cv_public_search_controls_reject_coercion(name, value, message):
    X, y, _ = _data(seed=16329, n=48)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    setattr(cv, name, value)

    with pytest.raises(ValueError, match=message):
        cv.fit(X, y)

    assert cv._fitted is False
    assert cv.estimator_ is None
    assert cv.cv_results_ is None


def test_quantile_cv_public_loss_kwargs_replacement_is_authoritative():
    X, y, folds = _data(seed=16324, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    cv.loss_kwargs = {"quantile": 0.8}
    cv.fit(X, y)

    assert cv._loss_kwargs == {"quantile": 0.8}
    assert getattr(cv.estimator_._loss, "_tau", None) == pytest.approx(0.8)


def test_quantile_cv_public_alpha_grid_replacement_is_authoritative():
    X, y, folds = _data(seed=16325, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=np.asarray([0.09], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    cv.alpha_grid = np.asarray([0.03], dtype=np.float64)
    cv.fit(X, y)

    np.testing.assert_array_equal(cv.alpha_grid_, np.asarray([0.03]))
    assert cv.alpha_ == pytest.approx(0.03)


def test_quantile_eval_dispatch_uses_call_local_tau_without_fast_registries():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    # No residual/validation fast registry is added: this PR fixes scoring
    # semantics while keeping the numerical implementation on maintained paths.
    assert "quantile" not in cv_mod._LOSS_RESIDUAL_FNS
    assert "quantile" not in cv_mod._LOSS_VALLOSS_FNS

    eta = np.array([-0.2, 0.1, 0.8], dtype=np.float64)
    y = np.array([0.4, -0.1, 1.2], dtype=np.float64)
    tau = 0.23

    token = contract._QUANTILE_CV_LEVEL.set(tau)
    try:
        eval_fn, _ = cv_mod._LOSS_EVAL_DISPATCH["quantile"]
        observed = eval_fn(eta, y)
    finally:
        contract._QUANTILE_CV_LEVEL.reset(token)

    u = y - eta
    expected = np.where(u >= 0.0, tau * u, (tau - 1.0) * u)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert contract._QUANTILE_CV_LEVEL.get() is None


def test_quantile_sparse_fold_batch_is_fail_safe_to_general_path():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod

    result = cv_mod._glm_sparse_cv_folds(
        None,
        None,
        [],
        np.array([0.03]),
        "l1",
        1.0,
        5,
        1e-4,
        "quantile",
        "torch",
    )
    assert result is None


def test_quantile_scad_fast_helper_is_fail_safe_to_general_path():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod

    result = cv_mod._scad_mcp_cv_path(
        "quantile",
        None,
        None,
        np.array([0.03]),
        "scad",
        1.0,
        5,
        1e-4,
        Device.CPU,
    )
    assert result is None


def test_quantile_scalar_scad_two_stage_uses_maintained_per_alpha_path_for_object_and_string(
    monkeypatch,
):
    from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

    X, y, folds = _data(seed=16336, n=72)
    alpha_grid = np.asarray([0.04, 0.025], dtype=np.float64)
    seen = {"object_path": 0, "string_path": 0}
    original_fit = PenalizedGeneralizedLinearModel.fit
    active_label = {"value": None}

    def tracking_fit(self, *args, **kwargs):
        if (
            str(getattr(self, "loss", "")).lower() == "quantile"
            and getattr(self, "_cv_alpha_path", None) is not None
        ):
            seen[active_label["value"]] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(PenalizedGeneralizedLinearModel, "fit", tracking_fit)

    common = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.25},
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=1,
        max_iter=300,
        tol=1e-6,
    )

    active_label["value"] = "object_path"
    object_cv = PenalizedGLM_CV(
        penalty=SCADPenalty(alpha=0.9, a=3.7),
        **common,
    ).fit(X, y)

    active_label["value"] = "string_path"
    string_cv = PenalizedGLM_CV(
        penalty="scad",
        penalty_kwargs={"a": 3.7},
        **common,
    ).fit(X, y)

    assert seen == {"object_path": 0, "string_path": 0}
    np.testing.assert_allclose(
        object_cv.cv_results_["all_scores_stage1"],
        string_cv.cv_results_["all_scores_stage1"],
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_array_equal(
        object_cv.cv_results_["refined_mask"],
        string_cv.cv_results_["refined_mask"],
    )


def test_quantile_l2_penalty_object_matches_string_cv_and_refit():
    X, y, folds = _data(seed=16344, n=72)
    alpha_grid = np.asarray([0.05, 0.02], dtype=np.float64)

    penalty_object = L2Penalty(alpha=0.9)
    object_cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty=penalty_object,
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=400,
        tol=1e-8,
    ).fit(X, y)
    string_cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="l2",
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=400,
        tol=1e-8,
    ).fit(X, y)

    np.testing.assert_allclose(
        object_cv.cv_results_["all_scores"],
        string_cv.cv_results_["all_scores"],
        rtol=2e-8,
        atol=2e-10,
    )
    assert object_cv.alpha_ == pytest.approx(string_cv.alpha_)
    np.testing.assert_allclose(
        object_cv.coef_, string_cv.coef_, rtol=2e-8, atol=2e-10
    )
    assert object_cv.intercept_ == pytest.approx(
        string_cv.intercept_, rel=2e-8, abs=2e-10
    )
    assert penalty_object.alpha == pytest.approx(0.9)
    assert object_cv.penalty is penalty_object
    assert object_cv.estimator_.penalty is not penalty_object
    assert object_cv.estimator_.penalty.alpha == pytest.approx(object_cv.alpha_)
    assert object_cv.estimator_._penalty.alpha == pytest.approx(object_cv.alpha_)


def test_quantile_adaptive_l1_fixed_weights_auto_grid_uses_public_scale():
    X, y, _ = _data(seed=16349, n=64)
    tau = 0.27
    raw_weights = np.asarray([0.5, 2.0], dtype=np.float64)

    object_penalty = AdaptiveL1Penalty(
        alpha=0.9,
        weights=raw_weights,
        normalize=True,
    )
    object_model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty=object_penalty,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )
    string_model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="adaptive_l1",
        penalty_kwargs={"weights": raw_weights, "normalize": True},
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )

    object_grid = object_model._generate_alpha_grid(X, y)
    string_grid = string_model._generate_alpha_grid(X, y)

    intercept = float(np.sort(y, kind="stable")[max(int(np.ceil(tau * len(y))) - 1, 0)])
    residual = y - intercept
    psi = _balanced_psi(residual, tau)
    score = X.T @ psi / float(X.shape[0])
    effective_weights = np.asarray(object_penalty._weights, dtype=np.float64)
    expected = float(np.max(np.abs(score) / effective_weights))

    assert object_grid[0] == pytest.approx(expected, rel=0.0, abs=1e-14)
    np.testing.assert_allclose(object_grid, string_grid, rtol=0.0, atol=1e-14)


def test_quantile_adaptive_group_fixed_weights_auto_grid_uses_public_scale():
    X, y, _ = _data(seed=16350, n=64)
    tau = 0.27
    groups = [[0], [1]]
    group_weights = np.asarray([0.5, 2.0], dtype=np.float64)
    penalty = AdaptiveGroupLassoPenalty(
        groups=groups,
        alpha=0.9,
        weights=group_weights,
    )
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty=penalty,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )

    grid = model._generate_alpha_grid(X, y)

    intercept = float(np.sort(y, kind="stable")[max(int(np.ceil(tau * len(y))) - 1, 0)])
    residual = y - intercept
    psi = _balanced_psi(residual, tau)
    score = X.T @ psi / float(X.shape[0])
    expected = max(
        abs(float(score[0])) / group_weights[0],
        abs(float(score[1])) / group_weights[1],
    )
    assert grid[0] == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_quantile_adaptive_group_zero_weight_keeps_finite_heuristic_grid():
    X, y, _ = _data(seed=16351, n=64)
    penalty = AdaptiveGroupLassoPenalty(
        groups=[[0], [1]],
        alpha=0.9,
        weights=np.asarray([0.0, 1.0], dtype=np.float64),
    )
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.27},
        penalty=penalty,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )

    grid = model._generate_alpha_grid(X, y)

    assert np.all(np.isfinite(grid))
    assert np.all(grid > 0.0)


def test_quantile_elasticnet_zero_l1_ratio_uses_l2_grid_scale():
    X, y, _ = _data(seed=16348, n=64)
    tau = 0.27

    l2_grid = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="l2",
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )._generate_alpha_grid(X, y)

    string_grid = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="elasticnet",
        l1_ratio=0.0,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )._generate_alpha_grid(X, y)

    object_grid = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty=ElasticNetPenalty(alpha=0.9, l1_ratio=0.0),
        l1_ratio=0.8,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )._generate_alpha_grid(X, y)

    np.testing.assert_array_equal(string_grid, l2_grid)
    np.testing.assert_array_equal(object_grid, l2_grid)


def test_quantile_elasticnet_penalty_object_auto_grid_uses_object_l1_ratio():
    X, y, _ = _data(seed=16345, n=64)
    tau = 0.27
    ratio = 0.25

    object_model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty=ElasticNetPenalty(alpha=0.9, l1_ratio=ratio),
        l1_ratio=0.8,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )
    string_model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="elasticnet",
        l1_ratio=ratio,
        alpha_grid=None,
        n_alphas=4,
        cv=2,
        solver="auto",
        device="cpu",
    )

    object_grid = object_model._generate_alpha_grid(X, y)
    string_grid = string_model._generate_alpha_grid(X, y)

    np.testing.assert_allclose(object_grid, string_grid, rtol=0.0, atol=1e-14)


def test_quantile_scalar_scad_penalty_object_matches_string_cv_and_refit():
    X, y, folds = _data(seed=16331, n=72)
    weights = np.linspace(0.5, 1.7, X.shape[0], dtype=np.float64)
    alpha_grid = np.asarray([0.04, 0.025], dtype=np.float64)

    penalty_object = SCADPenalty(alpha=0.9, a=3.7)
    object_cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.25},
        penalty=penalty_object,
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)
    string_cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.25},
        penalty="scad",
        penalty_kwargs={"a": 3.7},
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(
        object_cv.cv_results_["all_scores"],
        string_cv.cv_results_["all_scores"],
        rtol=2e-8,
        atol=2e-10,
    )
    assert object_cv.alpha_ == pytest.approx(string_cv.alpha_)
    np.testing.assert_allclose(
        object_cv.coef_, string_cv.coef_, rtol=2e-8, atol=2e-10
    )
    assert object_cv.intercept_ == pytest.approx(
        string_cv.intercept_, rel=2e-8, abs=2e-10
    )

    assert penalty_object.alpha == pytest.approx(0.9)
    assert object_cv.penalty is penalty_object
    assert object_cv.estimator_.penalty is not penalty_object
    assert object_cv.estimator_.penalty.alpha == pytest.approx(object_cv.alpha_)
    assert object_cv.estimator_._penalty.alpha == pytest.approx(object_cv.alpha_)


@pytest.mark.parametrize(
    "penalty,alpha_grid,expected_solver",
    [
        ("l1", np.array([0.04, 0.02], dtype=np.float64), "fista"),
        ("scad", np.array([0.025], dtype=np.float64), "proximal_irls_cd"),
    ],
)
def test_quantile_public_two_stage_falls_back_to_maintained_per_fold_path(
    penalty, alpha_grid, expected_solver
):
    """Public approximate CV remains usable when incomplete fast helpers decline."""
    X, y, folds = _data(seed=16322, n=72)
    tau = 0.2
    weights = np.linspace(0.55, 1.65, X.shape[0], dtype=np.float64)

    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty=penalty,
        alpha_grid=alpha_grid,
        cv=2,
        cv_splits=folds,
        random_state=163,
        solver="auto",
        device="cpu",
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=1,
        max_iter=500,
        tol=1e-7 if penalty == "l1" else 1e-6,
    ).fit(X, y, sample_weight=weights)

    assert cv.cv_strategy_ == "two_stage"
    assert cv.estimator_._selected_solver == expected_solver
    assert cv.estimator_._selected_backend_name == "numpy"
    assert cv.alpha_ in set(alpha_grid.tolist())

    stage1 = np.asarray(cv.cv_results_["all_scores_stage1"], dtype=np.float64)
    strict = np.asarray(cv.cv_results_["all_scores"], dtype=np.float64)
    assert stage1.shape == strict.shape == (len(folds), len(alpha_grid))
    assert np.all(np.isfinite(stage1))
    assert np.all(np.isfinite(strict))
    assert np.all(np.isfinite(np.asarray(cv.coef_, dtype=np.float64)))
    assert np.isfinite(float(cv.intercept_))

    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    assert contract._QUANTILE_CV_LEVEL.get() is None


def test_strict_scalar_nonconvex_quantile_cv_marks_target_nonconvergence_failed(
    monkeypatch,
):
    import statgpu.solvers as solvers
    from statgpu.solvers import _proximal_irls_quantile as kernel

    X, y, folds = _data(seed=16323, n=48)
    seen = []

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        strict = kernel._STRICT_CV_TARGET.get()
        seen.append(strict)
        if strict:
            raise FloatingPointError("scalar target did not converge")
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="scad",
        alpha_grid=np.asarray([0.025], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=20,
        tol=1e-6,
    )
    scores = cv._compute_cv_scores(
        X,
        y,
        np.asarray([0.025], dtype=np.float64),
        Device.CPU,
        folds,
        sample_weight=None,
        max_iter=20,
        tol=1e-6,
        strict=True,
    )

    assert seen and all(seen)
    assert np.all(np.isnan(scores))


def test_strict_scalar_nonconvex_quantile_cv_invalidates_partial_fold_candidate(
    monkeypatch,
):
    import statgpu.solvers as solvers
    from statgpu.solvers import _proximal_irls_quantile as kernel

    X, y, folds = _data(seed=16330, n=48)
    calls = {"strict": 0}

    def fail_first_strict_fold(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        if kernel._STRICT_CV_TARGET.get():
            calls["strict"] += 1
            if calls["strict"] == 1:
                raise FloatingPointError("first strict fold did not converge")
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        solvers, "proximal_irls_quantile_solver", fail_first_strict_fold
    )
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="scad",
        alpha_grid=np.asarray([0.025], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=20,
        tol=1e-6,
    )
    scores = cv._compute_cv_scores(
        X,
        y,
        np.asarray([0.025], dtype=np.float64),
        Device.CPU,
        folds,
        sample_weight=None,
        max_iter=20,
        tol=1e-6,
        strict=True,
    )

    assert calls["strict"] == len(folds)
    assert scores.shape == (len(folds), 1)
    assert np.all(np.isnan(scores[:, 0]))


def test_quantile_eval_preserves_historical_default_outside_cv_context():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod

    eta = np.array([0.0, 0.5], dtype=np.float64)
    y = np.array([1.0, 0.0], dtype=np.float64)
    eval_fn, _ = cv_mod._LOSS_EVAL_DISPATCH["quantile"]
    observed = eval_fn(eta, y)
    expected = np.array([0.5, 0.25], dtype=np.float64)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
