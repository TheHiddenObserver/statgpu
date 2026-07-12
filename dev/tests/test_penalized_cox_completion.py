"""Completion tests for the penalized Cox loss/estimator integration."""

from __future__ import annotations

import inspect
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.backends._utils import _to_numpy
from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.losses import CoxPartialLikelihoodLoss


@pytest.fixture(scope="module")
def survival_data():
    """Censored survival data with deterministic ties and a sparse signal."""
    rng = np.random.default_rng(20260712)
    n, p = 160, 6
    X = rng.normal(size=(n, p))
    beta = np.array([0.7, -0.6, 0.35, 0.0, 0.0, 0.0])
    event_time = rng.exponential(scale=np.exp(-(X @ beta)))
    censor_time = rng.exponential(scale=1.8, size=n)
    time = np.round(np.minimum(event_time, censor_time), 1) + 0.1
    event = (event_time <= censor_time).astype(np.float64)
    assert 0 < event.sum() < n
    return X.astype(np.float64), np.column_stack([time, event])


def _objective(X, y, coef, penalty, ties="breslow"):
    loss = CoxPartialLikelihoodLoss(ties=ties)
    return loss.value(X, y, coef) + penalty.value(coef)


def _kkt_violation(model, X, y):
    """Infinity-norm first-order residual for the five tested penalties."""
    coef = np.asarray(model.coef_, dtype=np.float64)
    grad = np.asarray(model._loss.gradient(X, y, coef), dtype=np.float64)
    penalty_name = str(model._penalty.name).lower()
    active = np.abs(coef) > 1e-7

    if penalty_name == "l2":
        residual = grad + model._penalty.gradient(coef)
        return float(np.max(np.abs(residual)))

    if penalty_name == "elasticnet":
        l1_threshold = model.alpha * model.l1_ratio
        smooth_grad = grad + model.alpha * (1.0 - model.l1_ratio) * coef
        active_residual = np.abs(smooth_grad + l1_threshold * np.sign(coef))
        zero_residual = np.maximum(np.abs(smooth_grad) - l1_threshold, 0.0)
    else:
        # L1, SCAD and MCP all have one-sided derivative alpha at zero.
        active_residual = np.abs(grad + model._penalty.gradient(coef))
        zero_residual = np.maximum(np.abs(grad) - model.alpha, 0.0)
    return float(np.max(np.where(active, active_residual, zero_residual)))


@pytest.mark.parametrize("penalty", ["l1", "l2", "elasticnet", "scad", "mcp"])
def test_penalized_cox_cpu_objective_and_convergence(survival_data, penalty):
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty=penalty,
        alpha=0.03,
        l1_ratio=0.4,
        ties="breslow",
        device="cpu",
        max_iter=500,
        tol=1e-7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(X, y)

    coef = np.asarray(model.coef_)
    objective = _objective(X, y, coef, model._penalty)
    objective_at_zero = _objective(X, y, np.zeros(X.shape[1]), model._penalty)

    assert model.n_iter_ > 0
    assert coef.shape == (X.shape[1],)
    assert np.all(np.isfinite(coef))
    assert np.isfinite(objective)
    assert objective < objective_at_zero - 1e-4
    assert _kkt_violation(model, X, y) < 1e-3


def test_efron_negative_loglik_hessian_matches_gradient_finite_difference(
    survival_data,
):
    X, y = survival_data
    coef = np.array([0.2, -0.1, 0.05, 0.0, 0.03, -0.02])
    loss = CoxPartialLikelihoodLoss(ties="efron")
    analytic = np.asarray(loss.hessian(X, y, coef), dtype=np.float64)
    epsilon = 1e-5
    directions = np.eye(X.shape[1])
    finite_difference = np.column_stack(
        [
            (
                np.asarray(loss.gradient(X, y, coef + epsilon * direction))
                - np.asarray(loss.gradient(X, y, coef - epsilon * direction))
            )
            / (2.0 * epsilon)
            for direction in directions
        ]
    )
    assert_allclose(analytic, finite_difference, rtol=2e-5, atol=2e-7)
    symmetric = 0.5 * (analytic + analytic.T)
    assert np.min(np.linalg.eigvalsh(symmetric)) >= -1e-10


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_penalized_cox_loss_is_finite_and_shift_invariant(survival_data, ties):
    X, y = survival_data
    coef = np.array([0.2, -0.1, 0.05, 0.0, 0.03, -0.02])
    reference_loss = CoxPartialLikelihoodLoss(ties=ties)
    shifted_loss = CoxPartialLikelihoodLoss(ties=ties)
    reference = (
        reference_loss.value(X, y, coef),
        np.asarray(reference_loss.gradient(X, y, coef)),
        np.asarray(reference_loss.hessian(X, y, coef)),
    )
    shifted_X = X + 1e9
    shifted = (
        shifted_loss.value(shifted_X, y, coef),
        np.asarray(shifted_loss.gradient(shifted_X, y, coef)),
        np.asarray(shifted_loss.hessian(shifted_X, y, coef)),
    )
    assert np.isfinite(shifted[0])
    assert np.all(np.isfinite(shifted[1]))
    assert np.all(np.isfinite(shifted[2]))
    assert_allclose(shifted[0], reference[0], rtol=2e-7, atol=2e-8)
    assert_allclose(shifted[1], reference[1], rtol=2e-7, atol=2e-8)
    assert_allclose(shifted[2], reference[2], rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_efron_heavy_tie_gradient_hessian_gpu_parity(survival_data, device):
    if not _gpu_available(device):
        pytest.skip(f"{device} GPU backend is unavailable")
    X, y = survival_data
    coef = np.array([0.2, -0.1, 0.05, 0.0, 0.03, -0.02])
    cpu_loss = CoxPartialLikelihoodLoss(ties="efron")
    expected_gradient = np.asarray(cpu_loss.gradient(X, y, coef))
    expected_hessian = np.asarray(cpu_loss.hessian(X, y, coef))
    if device == "cuda":
        import cupy as cp

        X_device = cp.asarray(X)
        y_device = cp.asarray(y)
        coef_device = cp.asarray(coef)
    else:
        import torch

        X_device = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        y_device = torch.as_tensor(y, dtype=torch.float64, device="cuda")
        coef_device = torch.as_tensor(coef, dtype=torch.float64, device="cuda")
    gpu_loss = CoxPartialLikelihoodLoss(ties="efron")
    actual_gradient = np.asarray(
        _to_numpy(gpu_loss.gradient(X_device, y_device, coef_device))
    )
    actual_hessian = np.asarray(
        _to_numpy(gpu_loss.hessian(X_device, y_device, coef_device))
    )
    assert_allclose(actual_gradient, expected_gradient, rtol=2e-9, atol=2e-10)
    assert_allclose(actual_hessian, expected_hessian, rtol=2e-8, atol=2e-9)


def test_penalized_cox_has_no_intercept_and_prediction_ignores_it(survival_data):
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty="l2", alpha=0.03, device="cpu", tol=1e-8, max_iter=200
    ).fit(X, y)

    assert model.fit_intercept is False
    assert model._effective_intercept is False
    assert model.intercept_ == 0.0
    assert model._params.shape == (X.shape[1],)

    expected = np.exp(np.clip(X @ model.coef_, -500.0, 500.0))
    assert_allclose(model.predict(X), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(model.predict_hazard_ratio(X), expected, rtol=1e-12, atol=1e-12)

    # Even corrupted legacy state cannot leak an unidentified intercept into
    # predictions after loading an older serialized estimator.
    model.intercept_ = 100.0
    assert_allclose(model.predict(X), expected, rtol=1e-12, atol=1e-12)
    invalid_X = X[:2].copy()
    invalid_X[0, 0] = np.nan
    with pytest.raises(ValueError, match="X must contain only finite values"):
        model.predict(invalid_X)


def test_penalized_cox_efron_fit_converges(survival_data):
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.03,
        ties="efron",
        device="cpu",
        max_iter=300,
        tol=1e-7,
    ).fit(X, y)

    objective = _objective(X, y, model.coef_, model._penalty, ties="efron")
    objective_at_zero = _objective(
        X, y, np.zeros(X.shape[1]), model._penalty, ties="efron"
    )
    assert objective < objective_at_zero - 1e-4
    assert _kkt_violation(model, X, y) < 1e-3


def test_penalized_cox_rejects_intercept():
    with pytest.raises(ValueError, match="does not fit an intercept"):
        PenalizedCoxPHModel(fit_intercept=True)
    model = PenalizedCoxPHModel()
    with pytest.raises(ValueError, match="does not fit an intercept"):
        model.set_params(fit_intercept=True)
    assert model.fit_intercept is False


@pytest.mark.parametrize(
    "penalty,inference_method",
    [("l2", "debiased"), ("l1", "bootstrap"), ("scad", "oracle")],
)
def test_penalized_cox_inference_is_explicitly_estimation_only(
    survival_data, penalty, inference_method
):
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty=penalty,
        alpha=0.03,
        device="cpu",
        compute_inference=True,
        inference_method=inference_method,
    )
    with pytest.raises(NotImplementedError, match="currently estimation-only"):
        model.fit(X, y)


def test_native_torch_efron_helpers_match_numpy(survival_data):
    """Exercise the Torch-only Efron math locally even without CUDA."""
    torch = pytest.importorskip("torch")
    X, y = survival_data
    coef = np.array([0.2, -0.1, 0.05, 0.0, 0.03, -0.02])
    loss = CoxPartialLikelihoodLoss(ties="efron")
    loss.preprocess(X, y)

    X_sorted = np.asarray(loss._X_sorted)
    eta = X_sorted @ coef
    expected_loglik = loss._cpu_loglik(eta, loss._time_np, loss._event_np)
    expected_grad, expected_hess = loss._cpu_grad_hess(
        eta, loss._time_np, loss._event_np
    )

    X_t = torch.as_tensor(X_sorted, dtype=torch.float64)
    eta_t = torch.as_tensor(eta, dtype=torch.float64)
    actual_loglik = loss._efron_loglik_backend(eta_t, X_t, torch)
    actual_grad, actual_hess = loss._efron_grad_hess_backend(
        eta_t - eta_t.max(), X_t, torch
    )

    assert_allclose(actual_loglik.numpy(), expected_loglik, rtol=1e-12, atol=1e-12)
    assert_allclose(actual_grad.numpy(), expected_grad, rtol=1e-11, atol=1e-11)
    assert_allclose(actual_hess.numpy(), expected_hess, rtol=1e-11, atol=1e-11)


def _gpu_available(device):
    if device == "cuda":
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.parametrize("device", ["cuda", "torch"])
@pytest.mark.parametrize("penalty", ["l1", "l2", "elasticnet", "scad", "mcp"])
def test_penalized_cox_available_backend_parity(survival_data, penalty, device):
    if not _gpu_available(device):
        pytest.skip(f"{device} GPU backend is unavailable")

    X, y = survival_data
    kwargs = dict(
        penalty=penalty,
        alpha=0.03,
        l1_ratio=0.4,
        ties="breslow",
        max_iter=500,
        tol=1e-7,
    )
    cpu = PenalizedCoxPHModel(device="cpu", **kwargs).fit(X, y)
    gpu = PenalizedCoxPHModel(device=device, **kwargs).fit(X, y)

    coef_tol = 2e-3 if penalty in ("scad", "mcp") else 2e-5
    assert_allclose(gpu.coef_, cpu.coef_, rtol=coef_tol, atol=coef_tol)
    assert_allclose(gpu.predict(X), cpu.predict(X), rtol=coef_tol, atol=coef_tol)
    assert gpu.intercept_ == 0.0
    assert gpu._effective_intercept is False

    gpu_objective = _objective(X, y, gpu.coef_, gpu._penalty)
    cpu_objective = _objective(X, y, cpu.coef_, cpu._penalty)
    assert_allclose(gpu_objective, cpu_objective, rtol=2e-5, atol=2e-6)


def test_torch_cuda_efron_does_not_import_cupy(survival_data, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA backend is unavailable")

    X, y = survival_data
    X_t = torch.as_tensor(X, dtype=torch.float64, device="cuda")
    y_t = torch.as_tensor(y, dtype=torch.float64, device="cuda")
    coef_t = torch.tensor(
        [0.2, -0.1, 0.05, 0.0, 0.03, -0.02],
        dtype=torch.float64,
        device="cuda",
    )
    loss = CoxPartialLikelihoodLoss(ties="efron")

    # An import attempt now fails the test.  The public Torch-CUDA loss path
    # must still evaluate value, gradient and Hessian successfully.
    monkeypatch.setitem(sys.modules, "cupy", None)
    value = loss.value(X_t, y_t, coef_t)
    gradient = loss.gradient(X_t, y_t, coef_t)
    hessian = loss.hessian(X_t, y_t, coef_t)

    assert np.isfinite(value)
    assert gradient.is_cuda and hessian.is_cuda
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian).all()


def test_penalized_cox_rejects_fractional_events(survival_data):
    X, y = survival_data
    invalid_y = y.copy()
    invalid_y[0, 1] = 0.5
    with pytest.raises(ValueError, match="event"):
        CoxPartialLikelihoodLoss(ties="breslow").preprocess(X, invalid_y)
    with pytest.raises(ValueError, match="event"):
        PenalizedCoxPHModel(
            penalty="l2", alpha=0.03, device="cpu", compute_inference=False
        ).fit(X, invalid_y)


@pytest.mark.parametrize("invalid_time", [0.0, -0.1])
def test_cox_partial_likelihood_loss_rejects_nonpositive_time(
    survival_data, invalid_time
):
    X, y = survival_data
    invalid_y = y.copy()
    invalid_y[0, 0] = invalid_time
    with pytest.raises(ValueError, match="time must contain only positive values"):
        CoxPartialLikelihoodLoss(ties="breslow").preprocess(X, invalid_y)


def test_penalized_cox_sklearn_clone_and_grid_search_smoke(survival_data):
    sklearn_base = pytest.importorskip("sklearn.base")
    sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.03,
        ties="efron",
        device="cpu",
        n_jobs=2,
        cpu_solver="fista_bb",
        lipschitz_L=4.5,
        gpu_memory_cleanup=True,
        inference_method="bootstrap",
        cov_type="hc1",
        hac_maxlags=3,
        stopping="objective",
        lla=False,
        max_lla_iters=7,
        lla_tol=2e-5,
        max_iter=150,
        tol=1e-6,
    )
    assert list(inspect.signature(model.fit).parameters) == [
        "X",
        "y",
        "sample_weight",
        "formula",
        "data",
    ]
    cloned = sklearn_base.clone(model)
    clone_params = cloned.get_params()
    expected_inherited_params = {
        "ties": "efron",
        "n_jobs": 2,
        "cpu_solver": "fista_bb",
        "lipschitz_L": 4.5,
        "gpu_memory_cleanup": True,
        "inference_method": "bootstrap",
        "cov_type": "hc1",
        "hac_maxlags": 3,
        "stopping": "objective",
        "lla": False,
        "max_lla_iters": 7,
        "lla_tol": 2e-5,
    }
    for name, expected in expected_inherited_params.items():
        assert clone_params[name] == expected
    search = sklearn_model_selection.GridSearchCV(
        model,
        {"alpha": [0.02, 0.04]},
        cv=2,
        error_score="raise",
    ).fit(X[:80], y[:80])
    assert search.best_estimator_.coef_ is not None


def test_penalized_cox_score_counts_prediction_ties_and_same_time_censoring():
    tied = PenalizedCoxPHModel(device="cpu")
    tied.coef_ = np.zeros(1)
    X = np.array([[1.0], [0.0], [-1.0]])
    y = np.array([[1.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    assert tied.score(X, y) == pytest.approx(0.5)

    ranked = PenalizedCoxPHModel(device="cpu")
    ranked.coef_ = np.ones(1)
    assert ranked.score(X, y) == pytest.approx(1.0)


def test_penalized_cox_set_params_updates_effective_tie_method():
    model = PenalizedCoxPHModel(ties="breslow", device="cpu")
    model.set_params(ties="EFRON")
    assert model.ties == "efron"
    assert model._resolve_loss().ties == "efron"
    with pytest.raises(ValueError, match="different tie methods"):
        model.set_params(loss_kwargs={"ties": "breslow"})


def test_penalized_cox_failed_refit_clears_previous_coefficients(survival_data):
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty="l2", alpha=0.03, device="cpu", max_iter=150
    ).fit(X, y)
    assert model.coef_ is not None
    invalid_y = y.copy()
    invalid_y[0, 1] = 0.5
    with pytest.raises(ValueError, match="event"):
        model.fit(X, invalid_y)
    assert model.coef_ is None
    assert model._fitted is False
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(X[:2])


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_penalized_cox_score_accepts_device_response_arrays(survival_data, device):
    X, y = survival_data
    model = PenalizedCoxPHModel(
        penalty="l2", alpha=0.03, device="cpu", max_iter=150
    ).fit(X, y)
    expected = model.score(X, y)
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA backend is unavailable: {exc}")
        actual = model.score(cp.asarray(X), cp.asarray(y))
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device is unavailable")
        actual = model.score(
            torch.as_tensor(X, device="cuda"),
            torch.as_tensor(y, device="cuda"),
        )
    assert actual == pytest.approx(expected)
