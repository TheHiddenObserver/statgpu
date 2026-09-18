"""Issue #163 regressions for truthful penalized Quantile solver identity."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


def _data(seed=16301, n=96, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.75, -0.3], dtype=np.float64)[:p]
    y = (0.25 + X @ beta + rng.laplace(scale=0.18, size=n)).astype(np.float64)
    return X, y


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.5},
            penalty="l2",
            alpha=0.03,
            solver="auto",
            device="cpu",
            max_iter=300,
            tol=1e-8,
        ),
        lambda: PenalizedQuantileRegression(
            quantile=0.5,
            penalty="l2",
            alpha=0.03,
            solver="auto",
            device="cpu",
            max_iter=300,
            tol=1e-8,
        ),
    ],
)
def test_l2_quantile_auto_reports_and_executes_irls(factory):
    X, y = _data()
    model = factory().fit(X, y)

    assert model._selected_solver == "irls"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("max_iter", True, "max_iter must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
    ],
)
def test_quantile_direct_public_stopping_controls_fail_closed(name, value, message):
    X, y = _data(seed=16311)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    setattr(model, name, value)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_rejected_quantile_refit_clears_all_fit_derived_state():
    X, y = _data(seed=16333, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    assert model._loss is not None
    assert model._penalty is not None
    assert model._nobs == X.shape[0]
    assert model._df_resid is not None

    model.tol = 0.0
    with pytest.raises(ValueError, match="tol must be a finite positive number"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.n_iter_ == 0
    assert model._selected_solver is None
    assert model._selected_backend_name is None
    assert model._selected_backend_device is None
    assert model._loss is None
    assert model._penalty is None
    assert model._nobs is None
    assert model._df_resid is None
    assert model._X_design is None
    assert model._y is None
    assert model._resid is None
    assert model._scale is None
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("tol", True, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
    ],
)
def test_quantile_cv_public_stopping_controls_fail_closed(name, value, message):
    X, y = _data(seed=16312, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
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


def test_quantile_direct_public_device_replacement_is_authoritative_before_backend(
    monkeypatch,
):
    X, y = _data(seed=16320)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    model.device = "cuda"

    def capture_backend(backend="auto"):
        assert model._device == Device.CUDA
        raise RuntimeError("device sync sentinel")

    monkeypatch.setattr(model, "_get_backend", capture_backend)
    with pytest.raises(RuntimeError, match="device sync sentinel"):
        model.fit(X, y)


def test_quantile_cv_public_device_replacement_is_authoritative_before_routing(
    monkeypatch,
):
    X, y = _data(seed=16321, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    cv.device = "torch"
    seen = {}

    def capture_device(self, *args, **kwargs):
        seen["device"] = self._device
        raise RuntimeError("device routing sentinel")

    monkeypatch.setattr(PenalizedGLM_CV, "_effective_cv_device", capture_device)
    with pytest.raises(RuntimeError, match="device routing sentinel"):
        cv.fit(X, y)

    assert seen["device"] == Device.TORCH
    assert cv._device == Device.TORCH


@pytest.mark.parametrize("owner_kind", ["direct", "cv"])
def test_quantile_invalid_public_device_replacement_fails_closed(owner_kind):
    X, y = _data(seed=16322, n=72)
    if owner_kind == "direct":
        owner = PenalizedQuantileRegression(
            quantile=0.4,
            penalty="l2",
            alpha=0.02,
            solver="auto",
            device="cpu",
        )
    else:
        owner = PenalizedGLM_CV(
            loss="quantile",
            loss_kwargs={"quantile": 0.4},
            penalty="l2",
            alpha_grid=np.asarray([0.03], dtype=np.float64),
            cv=2,
            solver="auto",
            device="cpu",
        )

    owner.device = "not-a-device"
    with pytest.raises(ValueError, match="device must be one of"):
        owner.fit(X, y)


def test_generic_quantile_direct_public_loss_kwargs_replacement_is_authoritative():
    X, y = _data(seed=16335)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    model.loss_kwargs = {"quantile": 0.8}
    model.fit(X, y)

    assert model._loss_kwargs == {"quantile": 0.8}
    assert getattr(model._loss, "_tau", None) == pytest.approx(0.8)


def test_quantile_direct_public_fit_intercept_replacement_is_authoritative():
    X, y = _data(seed=16315)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        fit_intercept=True,
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )
    model.fit_intercept = False
    model.fit(X, y)

    assert model._fit_intercept is False
    assert model._effective_intercept is False
    assert model.intercept_ == 0.0


def test_quantile_direct_invalid_public_fit_intercept_fails_before_backend(monkeypatch):
    X, y = _data(seed=16316)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        fit_intercept=True,
        device="cpu",
    )
    model.fit_intercept = "False"

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="fit_intercept must be boolean"):
        model.fit(X, y)


def test_invalid_direct_quantile_refit_control_clears_prior_fit_state():
    X, y = _data(seed=16317)
    model = PenalizedQuantileRegression(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    ).fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None

    model.max_iter = 0
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._selected_solver is None


def test_invalid_quantile_cv_refit_control_clears_prior_selection_state():
    X, y = _data(seed=16318, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.asarray([0.03], dtype=np.float64),
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=300,
        tol=1e-8,
    ).fit(X, y)
    assert cv._fitted is True
    assert cv.estimator_ is not None

    cv.tol = "1e-6"
    with pytest.raises(ValueError, match="tol must be a finite positive number"):
        cv.fit(X, y)

    assert cv._fitted is False
    assert cv.alpha_ is None
    assert cv.estimator_ is None
    assert cv.coef_ is None
    assert cv.intercept_ is None
    assert cv.cv_results_ is None


def test_explicit_irls_l2_matches_auto_quantile_fit():
    X, y = _data(seed=16302)
    common = dict(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        device="cpu",
        max_iter=400,
        tol=1e-9,
    )
    auto = PenalizedQuantileRegression(solver="auto", **common).fit(X, y)
    explicit = PenalizedQuantileRegression(solver="irls", **common).fit(X, y)

    assert auto._selected_solver == explicit._selected_solver == "irls"
    np.testing.assert_allclose(auto.coef_, explicit.coef_, rtol=0.0, atol=0.0)
    assert auto.intercept_ == pytest.approx(explicit.intercept_, rel=0.0, abs=0.0)


def test_explicit_fista_l2_quantile_executes_true_fista(monkeypatch):
    from statgpu.losses import QuantileLoss

    X, y = _data(seed=16303)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("explicit solver='fista' must not execute Quantile IRLS")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista",
        device="cpu",
        max_iter=1200,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


def test_explicit_fista_bb_l2_quantile_fails_before_backend_fit(monkeypatch):
    X, y = _data(seed=16303)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista_bb",
        device="cpu",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend fit must not run for rejected explicit solver")

    monkeypatch.setattr(model, "_fit_cpu", forbidden)
    with pytest.raises(ValueError, match="not supported for L2/no-penalty Quantile objectives"):
        model.fit(X, y)


def test_sparse_quantile_auto_remains_fista_family():
    X, y = _data(seed=16304)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l1",
        alpha=0.03,
        solver="auto",
        device="cpu",
        max_iter=500,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert np.all(np.isfinite(model.coef_))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("lla", False, "require lla=True"),
        ("lla", "False", "lla must be boolean"),
        ("max_lla_iters", 0, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", False, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", 2, "max_lla_iters must be at least 3"),
        ("lla_tol", 0.0, "lla_tol must be a finite positive number"),
        ("lla_tol", "1e-6", "lla_tol must be a finite positive number"),
    ],
)
def test_scalar_nonconvex_quantile_public_lla_controls_fail_closed(
    name, value, message
):
    X, y = _data(seed=16313, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
    )
    setattr(model, name, value)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_scalar_quantile_small_max_iter_is_never_expanded_by_continuation(
    monkeypatch,
):
    import statgpu.solvers as solvers

    X, y = _data(seed=16319, n=64)
    seen = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["max_iter"] = list(kwargs["max_iter"])
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)
    PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=7,
        tol=1e-6,
        max_lla_iters=6,
    ).fit(X, y)

    assert seen["max_iter"] == [1, 1, 7]


def test_scalar_quantile_three_lla_budget_maps_to_one_update_per_continuation(
    monkeypatch,
):
    import statgpu.solvers as solvers

    X, y = _data(seed=16314, n=64)
    seen = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["n_steps"] = len(alpha_path)
        seen["max_lla_per_step"] = kwargs["max_lla_per_step"]
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(solvers, "proximal_irls_quantile_solver", fake_solver)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=100,
        tol=1e-6,
        max_lla_iters=3,
    ).fit(X, y)

    assert seen == {"n_steps": 3, "max_lla_per_step": 1}
    assert model._selected_solver == "proximal_irls_cd"


def test_nonconvex_quantile_auto_reports_dedicated_proximal_irls_cd():
    X, y = _data(seed=16307, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=160,
        tol=1e-6,
    ).fit(X, y)

    assert model._selected_solver == "proximal_irls_cd"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)


def test_explicit_scalar_scad_solver_error_precedes_irrelevant_lla_controls():
    X, y = _data(seed=16337, n=48)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="fista",
        device="cpu",
        lla=False,
        max_lla_iters=1,
        lla_tol="unused-for-rejected-explicit-route",
    )

    with pytest.raises(ValueError, match="dedicated Proximal IRLS-CD"):
        model.fit(X, y)


@pytest.mark.parametrize("solver", ["fista", "fista_bb", "admm"])
def test_explicit_solver_does_not_silently_replace_quantile_scad(
    monkeypatch, solver
):
    X, y = _data(seed=16308, n=48)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver=solver,
        device="cpu",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend fit must not run for rejected explicit solver")

    monkeypatch.setattr(model, "_fit_cpu", forbidden)
    with pytest.raises(ValueError, match="dedicated Proximal IRLS-CD"):
        model.fit(X, y)


def test_quantile_cv_l2_uses_irls_for_candidates_and_final_refit():
    X, y = _data(seed=16305, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="l2",
        alpha_grid=np.array([0.04, 0.02], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=250,
        tol=1e-8,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "irls"
    assert cv.estimator_._selected_solver == "irls"
    assert cv.alpha_ in {0.04, 0.02}
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_cv_sparse_path_and_final_refit_remain_fista():
    X, y = _data(seed=16306, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="l1",
        alpha_grid=np.array([0.04], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=350,
        tol=1e-7,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "fista"
    assert cv.estimator_._selected_solver == "fista"
    assert cv.alpha_ == pytest.approx(0.04)
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_cv_scad_reports_dedicated_solver_for_candidates_and_refit():
    X, y = _data(seed=16309, n=60)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="scad",
        alpha_grid=np.array([0.025], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=140,
        tol=1e-6,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "proximal_irls_cd"
    assert cv.estimator_._selected_solver == "proximal_irls_cd"
    assert cv.alpha_ == pytest.approx(0.025)
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_formula_route_preserves_auto_irls_identity_and_numerics():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")

    X, y = _data(seed=16310, n=84)
    df = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})
    common = dict(
        quantile=0.45,
        penalty="l2",
        alpha=0.025,
        solver="auto",
        device="cpu",
        max_iter=350,
        tol=1e-9,
    )

    array_model = PenalizedQuantileRegression(**common).fit(X, y)
    formula_model = PenalizedQuantileRegression(**common).fit(
        formula="y ~ x1 + x2", data=df
    )

    assert array_model._selected_solver == "irls"
    assert formula_model._selected_solver == "irls"
    assert array_model._selected_backend_name == formula_model._selected_backend_name == "numpy"
    np.testing.assert_allclose(
        formula_model.coef_, array_model.coef_, rtol=0.0, atol=2e-10
    )
    assert formula_model.intercept_ == pytest.approx(
        array_model.intercept_, rel=0.0, abs=2e-10
    )


def test_quantile_solver_contract_installer_is_idempotent_and_signature_safe():
    from statgpu.linear_model.penalized import _fit_mixin
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    before_policy = _fit_mixin._preferred_penalized_glm_solver
    before_validate = PenalizedGeneralizedLinearModel._validate_solver_penalty
    before_direct_fit = PenalizedGeneralizedLinearModel.fit
    before_cv_fit = PenalizedGLM_CV.fit
    before_resolve_penalty = PenalizedGeneralizedLinearModel._resolve_penalty
    policy_signature = inspect.signature(before_policy)
    validate_signature = inspect.signature(before_validate)

    contract.install_quantile_solver_contract()

    assert _fit_mixin._preferred_penalized_glm_solver is before_policy
    assert PenalizedGeneralizedLinearModel._validate_solver_penalty is before_validate
    assert PenalizedGeneralizedLinearModel.fit is before_direct_fit
    assert PenalizedGLM_CV.fit is before_cv_fit
    assert PenalizedGeneralizedLinearModel._resolve_penalty is before_resolve_penalty
    assert inspect.signature(before_policy) == policy_signature
    assert inspect.signature(before_validate) == validate_signature
    assert hasattr(before_policy, "__wrapped__")
    assert hasattr(before_validate, "__wrapped__")
    assert getattr(
        before_direct_fit, contract._DIRECT_FIT_SOLVER_SYNC_MARKER, False
    )
    assert getattr(before_cv_fit, contract._CV_FIT_SOLVER_SYNC_MARKER, False)
    assert getattr(
        before_resolve_penalty, contract._RESOLVE_PENALTY_MARKER, False
    )


def test_quantile_solver_contract_reload_preserves_layered_explicit_solver_semantics(
    monkeypatch,
):
    import importlib

    from statgpu.losses import QuantileLoss
    from statgpu.linear_model.penalized import _fit_mixin
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    before_direct_fit = PenalizedGeneralizedLinearModel.fit
    before_cv_fit = PenalizedGLM_CV.fit
    before_backend_fit = _fit_mixin._PenalizedFitMixin._fit_loss_backend

    reloaded = importlib.reload(contract)

    assert PenalizedGeneralizedLinearModel.fit is before_direct_fit
    assert PenalizedGLM_CV.fit is before_cv_fit
    assert _fit_mixin._PenalizedFitMixin._fit_loss_backend is before_backend_fit

    X, y = _data(seed=16338, n=48)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError(
            "reload must not make explicit smooth Quantile FISTA fall back to IRLS"
        )

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista",
        device="cpu",
        max_iter=300,
        tol=1e-6,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert model._selected_backend_name == "numpy"

    rejected = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista_bb",
        device="cpu",
    )
    with pytest.raises(ValueError, match="not supported|does not support"):
        rejected.fit(X, y)

    assert getattr(
        reloaded._validate_quantile_solver_request,
        "_statgpu_quantile_smooth_fista_validator_contract",
        False,
    )


def test_quantile_solver_installer_is_import_order_safe_in_fresh_interpreter():
    import json
    import subprocess
    import sys

    code = r'''
import json
from statgpu.linear_model.penalized import _fit_mixin
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
payload = {
    "l2_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "l2", backend_name="numpy"
    ),
    "sparse_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "l1", backend_name="numpy"
    ),
    "nonconvex_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "scad", backend_name="numpy"
    ),
    "policy_wrapped": hasattr(
        _fit_mixin._preferred_penalized_glm_solver, "__wrapped__"
    ),
    "validate_wrapped": hasattr(
        PenalizedGeneralizedLinearModel._validate_solver_penalty, "__wrapped__"
    ),
}
print(json.dumps(payload, sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {
        "l2_auto": "irls",
        "nonconvex_auto": "proximal_irls_cd",
        "policy_wrapped": True,
        "sparse_auto": "fista",
        "validate_wrapped": True,
    }
