"""Weighted squared-error Group Lasso must not use unweighted block CD."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


_GROUPS = [[0, 3], [1, 2]]


def _data(seed=10301):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    y = 0.2 + X @ np.array([0.9, -0.5, 0.25, 0.65])
    y += rng.normal(scale=0.07, size=X.shape[0])
    y[:6] += np.array([15.0, -13.0, 11.0, -9.0, 8.0, -7.0])
    weights = np.ones(X.shape[0])
    weights[:6] = 0.02
    weights[60:] = 1.7
    return X, y, weights


def _backend_inputs(backend_name, X, y, weights):
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device unavailable")
        except Exception:
            pytest.skip("CuPy CUDA runtime unavailable")
        return (
            "cuda",
            cp.asarray(X),
            cp.asarray(y),
            cp.asarray(weights),
        )

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        torch.as_tensor(weights, dtype=torch.float64, device="cuda"),
    )


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _model(device="cpu"):
    return PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha=0.11,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=2500,
        tol=1e-9,
    )


def _weighted_objective(model, X, y, weights):
    X_work = np.column_stack([X, np.ones(X.shape[0])])
    params = np.append(np.asarray(model.coef_), float(model.intercept_))
    smooth = float(
        model._loss.value(
            X_work,
            y,
            params,
            sample_weight=weights,
        )
    )
    return smooth + float(model._penalty.value(np.asarray(model.coef_)))


def _weighted_kkt_residual(model, X, y, weights):
    X_work = np.column_stack([X, np.ones(X.shape[0])])
    params = np.append(np.asarray(model.coef_), float(model.intercept_))
    gradient = np.asarray(
        model._loss.gradient(
            X_work,
            y,
            params,
            sample_weight=weights,
        )
    )
    residuals = [abs(float(gradient[-1]))]
    for group in _GROUPS:
        idx = np.asarray(group, dtype=np.int64)
        beta_g = np.asarray(model.coef_)[idx]
        grad_g = gradient[idx]
        norm = np.linalg.norm(beta_g)
        threshold = model.alpha * np.sqrt(idx.size)
        if norm > 1e-8:
            residuals.append(
                np.linalg.norm(grad_g + threshold * beta_g / norm)
            )
        else:
            residuals.append(max(np.linalg.norm(grad_g) - threshold, 0.0))
    return float(max(residuals))


def test_weighted_group_lasso_satisfies_weighted_kkt_and_beats_unweighted_fit():
    X, y, weights = _data()
    weighted = _model().fit(X, y, sample_weight=weights)
    unweighted = _model().fit(X, y)

    assert weighted._selected_solver == "fista"
    assert weighted._penalty.name == "group_lasso"
    assert _weighted_kkt_residual(weighted, X, y, weights) < 3e-4
    assert _weighted_objective(weighted, X, y, weights) < (
        _weighted_objective(unweighted, X, y, weights) - 1e-3
    )
    assert np.linalg.norm(weighted.coef_ - unweighted.coef_) > 1e-3


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
def test_weighted_group_lasso_gpu_matches_cpu(backend_name):
    X, y, weights = _data(seed=10302)
    reference = _model().fit(X, y, sample_weight=weights)
    device, Xb, yb, wb = _backend_inputs(backend_name, X, y, weights)
    actual = _model(device=device).fit(Xb, yb, sample_weight=wb)

    np.testing.assert_allclose(
        _as_numpy(actual.coef_), reference.coef_, rtol=4e-5, atol=4e-6
    )
    assert actual.intercept_ == pytest.approx(
        reference.intercept_, rel=4e-5, abs=4e-6
    )
    np.testing.assert_allclose(
        _as_numpy(actual.predict(Xb)),
        reference.predict(X),
        rtol=4e-5,
        atol=4e-6,
    )


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
def test_weighted_group_lasso_gpu_cv_matches_cpu(backend_name):
    X, y, weights = _data(seed=10303)
    common = dict(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha_grid=[0.18, 0.09],
        cv=2,
        random_state=31,
        max_iter=1800,
        tol=1e-8,
    )
    reference = PenalizedGLM_CV(device="cpu", **common).fit(
        X, y, sample_weight=weights
    )
    device, Xb, yb, wb = _backend_inputs(backend_name, X, y, weights)
    actual = PenalizedGLM_CV(device=device, **common).fit(
        Xb, yb, sample_weight=wb
    )

    np.testing.assert_allclose(
        actual.cv_results_["all_scores"],
        reference.cv_results_["all_scores"],
        rtol=5e-5,
        atol=5e-6,
    )
    assert actual.alpha_ == pytest.approx(reference.alpha_)
    assert actual.estimator_.alpha == pytest.approx(actual.alpha_)
    np.testing.assert_allclose(
        _as_numpy(actual.coef_), reference.coef_, rtol=5e-5, atol=5e-6
    )
