"""Contracts for truthful explicit Quantile FISTA on L2/no-penalty objectives."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)
from statgpu.glm_core._squared import SquaredErrorLoss
from statgpu.losses import QuantileLoss
from statgpu.penalties import L2Penalty
from statgpu.solvers import fista_solver
from statgpu.solvers._convergence import ConvergenceWarning


def _data(seed=16681, n=72, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.8, -0.35], dtype=np.float64)[:p]
    y = (0.3 + X @ beta + rng.laplace(scale=0.22, size=n)).astype(np.float64)
    weights = np.linspace(0.6, 1.8, n, dtype=np.float64)
    return X, y, weights


def _objective(model, X, y, weights, quantile, alpha):
    pred = X @ np.asarray(model.coef_, dtype=np.float64) + float(model.intercept_)
    u = y - pred
    loss = np.where(u >= 0.0, quantile * u, (quantile - 1.0) * u)
    data_fit = float(np.average(loss, weights=weights))
    penalty = 0.5 * float(alpha) * float(np.dot(model.coef_, model.coef_))
    return data_fit + penalty


@pytest.mark.parametrize("penalty,alpha", [("l2", 0.025), ("none", 0.0)])
@pytest.mark.parametrize("typed", [False, True])
def test_explicit_smooth_quantile_fista_is_true_fista_and_weighted(
    monkeypatch, penalty, alpha, typed
):
    X, y, weights = _data()

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("explicit Quantile FISTA must not call QuantileLoss.irls")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    if typed:
        model = PenalizedQuantileRegression(
            quantile=0.35,
            penalty=penalty,
            alpha=alpha,
            solver="fista",
            device="cpu",
            max_iter=3000,
            tol=1e-7,
        )
    else:
        model = PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.35},
            penalty=penalty,
            alpha=alpha,
            solver="fista",
            device="cpu",
            max_iter=3000,
            tol=1e-7,
        )

    model.fit(X, y, sample_weight=weights)
    assert model._selected_solver == "fista"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


def test_direct_public_solver_replacement_from_auto_to_fista_is_authoritative(monkeypatch):
    X, y, weights = _data(seed=16685, n=64)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("public solver='fista' replacement must not execute IRLS")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    model = PenalizedQuantileRegression(
        quantile=0.35,
        penalty="l2",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=2500,
        tol=1e-7,
    )
    model.solver = "fista"
    model.fit(X, y, sample_weight=weights)

    assert model._solver == "fista"
    assert model._selected_solver == "fista"


def test_cv_public_solver_replacement_from_auto_to_fista_is_authoritative(monkeypatch):
    X, y, weights = _data(seed=16686, n=60)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("CV public solver='fista' replacement must not execute IRLS")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.4},
        penalty="l2",
        alpha_grid=np.array([0.03], dtype=np.float64),
        cv=2,
        random_state=166,
        solver="auto",
        device="cpu",
        max_iter=2500,
        tol=1e-7,
    )
    cv.solver = "fista"
    cv.fit(X, y, sample_weight=weights)

    assert cv._solver == "fista"
    assert cv._solver_for_cv("cpu", X=X) == "fista"
    assert cv.estimator_._selected_solver == "fista"


def test_quantile_fista_nonconvergence_warning_recommends_supported_routes():
    X, y, _ = _data(seed=16687, n=48)

    with pytest.warns(ConvergenceWarning) as caught:
        fista_solver(
            QuantileLoss(quantile=0.35),
            L2Penalty(alpha=0.02),
            X,
            y,
            max_iter=1,
            tol=1e-30,
        )

    messages = [str(item.message) for item in caught]
    message = next(
        text for text in messages if "did not converge within 1 iterations" in text
    )
    assert "IRLS is also supported" in message
    assert "newton" not in message.lower()
    assert "lbfgs" not in message.lower()


def test_smooth_quantile_torch_batches_armijo_and_convergence_sync(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.solvers._fista as fista_mod

    X_np = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )
    y_np = np.zeros(X_np.shape[0], dtype=np.float64)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)

    calls = {"sync": 0, "to_float": 0}
    original_sync = fista_mod._sync_scalars
    original_to_float = fista_mod._to_float_scalar

    def recording_sync(*args, **kwargs):
        calls["sync"] += 1
        return original_sync(*args, **kwargs)

    def recording_to_float(*args, **kwargs):
        calls["to_float"] += 1
        return original_to_float(*args, **kwargs)

    monkeypatch.setattr(fista_mod, "_sync_scalars", recording_sync)
    monkeypatch.setattr(fista_mod, "_to_float_scalar", recording_to_float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, n_iter = fista_mod.fista_solver(
            QuantileLoss(quantile=0.5),
            L2Penalty(alpha=0.0),
            X,
            y,
            max_iter=10,
            tol=1e-8,
        )

    assert n_iter == 1
    torch.testing.assert_close(coef, torch.zeros_like(coef), rtol=0.0, atol=0.0)
    # The accepted Armijo trial and convergence decision share one batched
    # synchronization; there is no separate host scalar conversion.
    assert calls == {"sync": 1, "to_float": 0}


def test_smooth_quantile_fista_numpy_torch_point_parity_matches_public_fixture():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(16692 + 31)
    X = rng.normal(size=(80, 1)).astype(np.float64)
    y = (
        0.35
        + 0.75 * X[:, 0]
        + rng.laplace(scale=0.22, size=80)
    ).astype(np.float64)
    X_aug = np.column_stack([X, np.ones(X.shape[0], dtype=np.float64)])
    loss = QuantileLoss(quantile=0.20)
    penalty = L2Penalty(alpha=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cpu_params, cpu_n_iter = fista_solver(
            loss,
            penalty,
            X_aug,
            y,
            max_iter=1600,
            tol=1e-7,
        )
        torch_params, torch_n_iter = fista_solver(
            QuantileLoss(quantile=0.20),
            L2Penalty(alpha=0.0),
            torch.as_tensor(X_aug, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64),
            max_iter=1600,
            tol=1e-7,
        )

    torch_params_np = torch_params.detach().cpu().numpy()
    np.testing.assert_allclose(
        torch_params_np,
        np.asarray(cpu_params, dtype=np.float64),
        rtol=0.0,
        atol=2e-5,
    )
    assert 1 <= cpu_n_iter <= 1600
    assert 1 <= torch_n_iter <= 1600


def test_fista_last_allowed_iteration_convergence_is_not_false_exhaustion():
    X = np.eye(2, dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, n_iter = fista_solver(
            SquaredErrorLoss(),
            L2Penalty(alpha=0.1),
            X,
            y,
            max_iter=1,
            tol=1e-12,
        )

    assert n_iter == 1
    np.testing.assert_array_equal(coef, np.zeros(2, dtype=np.float64))


def test_explicit_l2_quantile_fista_matches_irls_objective():
    X, y, weights = _data(seed=16682, n=96)
    common = dict(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        device="cpu",
        max_iter=5000,
        tol=1e-8,
    )
    fista = PenalizedQuantileRegression(solver="fista", **common).fit(
        X, y, sample_weight=weights
    )
    irls = PenalizedQuantileRegression(solver="irls", **common).fit(
        X, y, sample_weight=weights
    )

    fista_obj = _objective(fista, X, y, weights, 0.4, 0.02)
    irls_obj = _objective(irls, X, y, weights, 0.4, 0.02)
    assert abs(fista_obj - irls_obj) <= 5e-4


def test_explicit_no_penalty_quantile_fista_matches_sklearn_objective():
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    X, y, _ = _data(seed=16683, n=64)
    q = 0.3

    model = PenalizedQuantileRegression(
        quantile=q,
        penalty="none",
        alpha=0.0,
        solver="fista",
        device="cpu",
        max_iter=5000,
        tol=1e-8,
    ).fit(X, y)
    reference = sklearn_linear.QuantileRegressor(
        quantile=q,
        alpha=0.0,
        fit_intercept=True,
        solver="highs",
    ).fit(X, y)

    pred = X @ model.coef_ + model.intercept_
    pred_ref = X @ reference.coef_ + reference.intercept_
    u = y - pred
    u_ref = y - pred_ref
    obj = float(np.mean(np.where(u >= 0.0, q * u, (q - 1.0) * u)))
    obj_ref = float(
        np.mean(np.where(u_ref >= 0.0, q * u_ref, (q - 1.0) * u_ref))
    )
    assert obj - obj_ref <= 5e-4


def test_quantile_cv_explicit_l2_fista_uses_fista_for_children_and_refit(monkeypatch):
    X, y, weights = _data(seed=16684, n=60)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("explicit Quantile FISTA CV must not call QuantileLoss.irls")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.45},
        penalty="l2",
        alpha_grid=np.array([0.03], dtype=np.float64),
        cv=2,
        random_state=166,
        solver="fista",
        device="cpu",
        max_iter=2500,
        tol=1e-7,
        cv_strategy="strict",
    ).fit(X, y, sample_weight=weights)

    assert cv._solver_for_cv("cpu", X=X) == "fista"
    assert cv.estimator_._selected_solver == "fista"
    assert cv.alpha_ == pytest.approx(0.03)
    assert np.all(np.isfinite(cv.coef_))


def test_smooth_quantile_fista_installer_is_idempotent_and_signature_safe():
    from statgpu.linear_model.penalized import _fit_mixin
    from statgpu.linear_model.penalized import _quantile_solver_contract
    from statgpu.linear_model.penalized import _quantile_smooth_fista_contract as contract

    before_fit = _fit_mixin._PenalizedFitMixin._fit_loss_backend
    before_validate = _quantile_solver_contract._validate_quantile_solver_request
    fit_signature = inspect.signature(before_fit)
    validate_signature = inspect.signature(before_validate)

    contract.install_quantile_smooth_fista_contract()

    assert _fit_mixin._PenalizedFitMixin._fit_loss_backend is before_fit
    assert _quantile_solver_contract._validate_quantile_solver_request is before_validate
    assert inspect.signature(before_fit) == fit_signature
    assert inspect.signature(before_validate) == validate_signature
    assert hasattr(before_fit, "__wrapped__")
    assert hasattr(before_validate, "__wrapped__")
