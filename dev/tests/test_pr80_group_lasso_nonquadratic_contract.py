"""Non-quadratic Group Lasso must optimize the advertised loss."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


_GROUPS = [[0, 3], [1, 2]]


def _data(seed=10101):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(140, 4))
    y = 0.25 + X @ np.array([0.9, -0.55, 0.3, 0.7])
    y += rng.normal(scale=0.08, size=X.shape[0])
    # Deliberate high-leverage response contamination so the Huber and
    # Gaussian group solutions are observably different.
    y[:8] += np.array([18.0, -16.0, 15.0, -14.0, 13.0, -12.0, 11.0, -10.0])
    return X, y


def _backend_inputs(backend_name, X, y):
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device unavailable")
        except Exception:
            pytest.skip("CuPy CUDA runtime unavailable")
        return "cuda", cp.asarray(X), cp.asarray(y)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
    )


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _model(loss, *, device="cpu", alpha=0.12):
    return PenalizedGeneralizedLinearModel(
        loss=loss,
        loss_kwargs={"delta": 1.0} if loss == "huber" else None,
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha=alpha,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=2500,
        tol=1e-9,
    )


def _huber_composite_objective(model, X, y):
    X_work = np.column_stack([X, np.ones(X.shape[0])])
    params = np.append(np.asarray(model.coef_), float(model.intercept_))
    smooth = float(model._loss.value(X_work, y, params))
    penalty = float(model._penalty.value(np.asarray(model.coef_)))
    return smooth + penalty


def _group_kkt_residual(model, X, y):
    X_work = np.column_stack([X, np.ones(X.shape[0])])
    params = np.append(np.asarray(model.coef_), float(model.intercept_))
    gradient = np.asarray(model._loss.gradient(X_work, y, params))
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


def test_huber_group_lasso_satisfies_composite_kkt_and_beats_gaussian_bcd():
    X, y = _data()
    huber = _model("huber").fit(X, y)
    gaussian = _model("squared_error").fit(X, y)

    assert huber._selected_solver == "fista"
    assert huber._penalty.name == "group_lasso"
    assert _group_kkt_residual(huber, X, y) < 3e-4
    assert _huber_composite_objective(huber, X, y) < (
        _huber_composite_objective(gaussian, X, y) - 1e-3
    )
    assert np.linalg.norm(huber.coef_ - gaussian.coef_) > 1e-3


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
def test_huber_group_lasso_gpu_matches_cpu_kkt_solution(backend_name):
    X, y = _data(seed=10102)
    reference = _model("huber").fit(X, y)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    actual = _model("huber", device=device).fit(Xb, yb)

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
def test_huber_group_lasso_gpu_cv_scores_selection_and_refit_match_cpu(
    backend_name,
):
    X, y = _data(seed=10103)
    common = dict(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha_grid=[0.2, 0.1],
        cv=2,
        random_state=29,
        max_iter=1800,
        tol=1e-8,
    )
    reference = PenalizedGLM_CV(device="cpu", **common).fit(X, y)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    actual = PenalizedGLM_CV(device=device, **common).fit(Xb, yb)

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
