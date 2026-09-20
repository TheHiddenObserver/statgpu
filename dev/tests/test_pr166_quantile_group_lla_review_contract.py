"""Fresh-review contracts for Quantile group-penalty routing in PR #166."""

from __future__ import annotations

import importlib
import types

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
from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
)
import statgpu.penalties._group_lasso as _group_lasso_impl
import statgpu.backends._utils as _backend_utils


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


def test_public_group_penalty_cupy_caches_migrate_with_operand_device(monkeypatch):
    penalty = AdaptiveGroupLassoPenalty(
        groups=GROUPS,
        alpha=0.1,
        weights=[0.75, 1.25],
    )
    targets = []

    class FakeDevice:
        def __init__(self, device_id):
            self.id = int(device_id)

    class FakeArray:
        __module__ = "cupy._core.core"

        def __init__(self, device_id, dtype=np.float64):
            self.device = FakeDevice(device_id)
            self.dtype = np.dtype(dtype)

        def astype(self, dtype, copy=False):
            self.dtype = np.dtype(dtype)
            return self

    def fake_cupy_align(value, target_device, dtype=None):
        targets.append(int(target_device))
        return FakeArray(target_device, dtype or np.float64)

    monkeypatch.setattr(
        _backend_utils,
        "_cupy_asarray_on_device",
        fake_cupy_align,
    )

    xp = types.SimpleNamespace(__name__="cupy")
    w = FakeArray(1)

    # Seed stale caches as if the same public penalty had previously executed
    # on cuda:0, then reuse it for a cuda:1 fit.
    penalty._sqrt_pg_cupy = FakeArray(0)
    penalty._group_feat_idx_cupy = FakeArray(0)
    penalty._group_weights_cupy = FakeArray(0)

    sqrt_pg = penalty._get_sqrt_pg(xp, w)
    group_index = penalty._get_cached("_group_feat_idx", xp, w)
    group_weights = penalty._get_group_weights(xp, w)

    assert sqrt_pg.device.id == 1
    assert group_index.device.id == 1
    assert group_weights.device.id == 1
    assert targets == [1, 1, 1]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GroupMCPPenalty(alpha=0.1, gamma=3.0, groups=GROUPS),
        lambda: GroupSCADPenalty(alpha=0.1, a=3.7, groups=GROUPS),
    ],
)
def test_public_group_nonconvex_cupy_caches_migrate_with_operand_device(
    monkeypatch, factory
):
    penalty = factory()
    targets = []

    class FakeDevice:
        def __init__(self, device_id):
            self.id = int(device_id)

    class FakeArray:
        __module__ = "cupy._core.core"

        def __init__(self, device_id):
            self.device = FakeDevice(device_id)
            self.dtype = np.dtype("float64")

    def fake_cupy_align(value, target_device, dtype=None):
        targets.append(int(target_device))
        return FakeArray(target_device)

    monkeypatch.setattr(
        _backend_utils,
        "_cupy_asarray_on_device",
        fake_cupy_align,
    )

    xp = types.SimpleNamespace(__name__="cupy")
    w = FakeArray(1)
    penalty._sqrt_pg_cupy = FakeArray(0)
    penalty._group_feat_idx_cupy = FakeArray(0)

    sqrt_pg = penalty._get_sqrt_pg(xp, w)
    group_index = penalty._get_cached("_group_feat_idx", xp, w)

    assert sqrt_pg.device.id == 1
    assert group_index.device.id == 1
    assert targets == [1, 1]


def test_noncontiguous_group_routes_use_backend_flat_indices(monkeypatch):
    groups = [[0, 2], [1, 3]]
    coef = np.asarray([0.8, -0.4, 0.25, 0.15], dtype=np.float64)

    adaptive = AdaptiveGroupLassoPenalty(
        groups=groups,
        alpha=0.1,
        weights=[0.75, 1.25],
    )
    adaptive_calls = []
    adaptive_original = adaptive._get_flat_indices

    def adaptive_flat(xp, ref):
        adaptive_calls.append(ref)
        return adaptive_original(xp, ref)

    monkeypatch.setattr(adaptive, "_get_flat_indices", adaptive_flat)
    result = adaptive.proximal(coef.copy(), 0.2, backend="numpy")
    assert np.all(np.isfinite(result))
    assert adaptive_calls

    adaptive_calls.clear()
    value = adaptive.value(coef)
    gradient = adaptive.gradient(coef)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert len(adaptive_calls) >= 2

    for penalty in (
        GroupSCADPenalty(alpha=0.1, a=3.7, groups=groups),
        GroupMCPPenalty(alpha=0.1, gamma=3.0, groups=groups),
    ):
        calls = []
        original = penalty._get_flat_indices

        def recording_flat(xp, ref, *, _calls=calls, _original=original):
            _calls.append(ref)
            return _original(xp, ref)

        monkeypatch.setattr(penalty, "_get_flat_indices", recording_flat)
        weights = penalty.lla_weights(coef)
        assert np.all(np.isfinite(np.asarray(weights)))
        assert calls


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


def test_automatic_quantile_group_solver_keeps_lla_derivatives_backend_native(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    X_np, y_np, _ = _data(seed=166320, n=12)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    penalty = GroupSCADPenalty(alpha=0.08, a=3.7, groups=GROUPS)
    captured = {}

    def fake_factory(penalty_arg):
        def factory(derivatives):
            captured["derivatives"] = derivatives
            return object()

        return factory

    def fake_admm(
        loss,
        inner_penalty,
        X_quad,
        y_quad,
        *,
        init_coef=None,
        **kwargs,
    ):
        return init_coef.clone(), 1

    monkeypatch.setattr(group_solver, "_group_surrogate_factory", fake_factory)
    monkeypatch.setattr(group_solver, "admm_solver", fake_admm)

    coef, intercept, n_iter = group_solver.quantile_group_proximal_irls_lla_solver(
        loss=__import__("statgpu.losses", fromlist=["QuantileLoss"]).QuantileLoss(Q),
        penalty=penalty,
        X=X,
        y=y,
        alpha_path=np.asarray([0.08], dtype=np.float64),
        max_lla_per_step=1,
        max_iter=1,
        tol=1e-8,
        lla_tol=1e-8,
        fit_intercept=False,
    )

    assert torch.is_tensor(captured["derivatives"])
    assert captured["derivatives"].device == X.device
    assert captured["derivatives"].dtype == X.dtype
    assert np.all(np.isfinite(coef))
    assert intercept == 0.0
    assert n_iter == 1


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
