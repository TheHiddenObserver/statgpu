"""Regression tests for the final PR #80 review fixes."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.cross_validation._base import CVCache
from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.survival import CoxPH
from statgpu.survival import _cox_counting as counting_module
from statgpu.survival import _cox_cv as cox_cv_module
from statgpu.survival._cox_counting import fit_counting_process_cox
from statgpu.survival._cox_score import score as cox_score
from statgpu.survival._risk_sets import cox_counting_process_objective


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_penalized_cox_uses_failure_time_local_risk_scaling(ties):
    # The maximum linear predictor leaves before the tied failures.  A single
    # global max shift makes every later risk weight underflow to zero.
    X = np.array([[1000.0], [0.0], [-1.0], [-2.0]])
    time = np.array([1.0, 2.0, 2.0, 3.0])
    event = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.column_stack([time, event])
    coef = np.array([1.0])

    reference = cox_counting_process_objective(
        coef, X, time, event, ties=ties
    )
    loss = CoxPartialLikelihoodLoss(ties=ties)

    value = loss.value(X, y, coef)
    gradient = np.asarray(loss.gradient(X, y, coef))
    hessian = np.asarray(loss.hessian(X, y, coef))

    n = X.shape[0]
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert value == pytest.approx(
        -float(reference["log_likelihood"]) / n, rel=1e-12, abs=1e-12
    )
    assert_allclose(
        gradient,
        -np.asarray(reference["score"]) / n,
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        hessian,
        np.asarray(reference["information"]) / n,
        # The stable suffix implementation changes summation order while
        # retaining substantially tighter accuracy than backend parity needs.
        rtol=5e-11,
        atol=5e-12,
    )


def test_penalized_cox_loss_avoids_quadratic_shared_risk_scans(monkeypatch):
    import statgpu.losses._cox_ph as cox_loss_module

    X = np.array([[1000.0], [0.0], [-1.0], [-2.0]])
    y = np.array([[1.0, 0.0], [2.0, 1.0], [2.0, 1.0], [3.0, 0.0]])
    coef = np.array([1.0])
    loss = CoxPartialLikelihoodLoss(ties="efron")

    def fail_shared_derivatives(*args, **kwargs):
        raise AssertionError("first-order path requested the shared p-by-p information")

    monkeypatch.setattr(
        cox_loss_module,
        "cox_counting_process_objective",
        fail_shared_derivatives,
    )
    value, gradient = loss.fused_value_and_gradient(X, y, coef)
    hessian = loss.hessian(X, y, coef)
    assert np.isfinite(value)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.isfinite(np.asarray(hessian)))


def test_all_censored_loss_validates_coefficient_contract():
    X = np.ones((4, 1), dtype=np.float64)
    y = np.column_stack(
        [np.arange(1.0, 5.0), np.zeros(4, dtype=np.float64)]
    )
    loss = CoxPartialLikelihoodLoss(ties="breslow")

    with pytest.raises(ValueError, match="coef must have shape"):
        loss.hessian(X, y, np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match="coef must contain only finite"):
        loss.hessian(X, y, np.array([np.nan]))
    with pytest.raises(ValueError, match="coef must have shape"):
        loss.lipschitz(X, np.zeros(2, dtype=np.float64), y=y)


def test_counting_solver_reports_line_search_failure_without_discarding_iterate(
    monkeypatch,
):
    def objective(beta, X, stop, event, **kwargs):
        beta_value = float(np.asarray(beta)[0])
        return {
            "log_likelihood": np.asarray(0.0 if beta_value == 0.0 else -1.0),
            "score": np.array([1.0]),
            "information": np.array([[1.0]]),
        }

    monkeypatch.setattr(
        counting_module, "cox_counting_process_objective", objective
    )
    result = fit_counting_process_cox(
        np.ones((3, 1)),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0]),
        ties="breslow",
        max_iter=2,
        compute_baseline=False,
        compute_score_residuals=False,
    )

    assert result["converged"] is False
    assert result["stop_reason"] == "line_search_failed"
    assert_allclose(result["coef"], np.zeros(1))
    assert len(result["objective_history"]) == 1


def test_cox_cv_reuses_thread_safe_shared_cache():
    assert isinstance(cox_cv_module._COXPH_CV_CACHE, CVCache)
    cox_cv_module._COXPH_CV_CACHE.clear()
    cox_cv_module._COXPH_CV_CACHE.put("key", {"value": 1})
    assert cox_cv_module._COXPH_CV_CACHE.get("key") == {"value": 1}
    assert cox_cv_module._COXPH_CV_CACHE.pop("key") == {"value": 1}


def test_cox_inference_uses_unified_distribution_backend():
    import statgpu.survival._cox as cox_module

    source = inspect.getsource(cox_module)
    assert "from scipy import stats" not in source
    assert "stats.norm" not in source
    assert "stats.chi2" not in source


def test_cox_public_facade_preserves_historical_class_path():
    assert CoxPH.__module__ == "statgpu.survival._cox"
    assert CoxPH.__name__ == "CoxPH"


def test_cox_score_packed_target_uses_active_backend_source():
    source = inspect.getsource(cox_score)
    assert "np.asarray(self._to_numpy(time)" not in source
    assert "target = backend.asarray(time" in source


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_cox_score_packed_target_preserves_explicit_gpu_backend(device, monkeypatch):
    X_np = np.array([[1.0], [0.0], [-1.0], [-2.0]], dtype=np.float64)
    y_np = np.array(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0]],
        dtype=np.float64,
    )
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA backend is unavailable: {exc}")
        X = cp.asarray(X_np)
        y = cp.asarray(y_np)
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device is unavailable")
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")

    model = CoxPH(
        device=device,
        compute_inference=False,
        compute_cindex=False,
        max_iter=50,
    ).fit(X, time=y[:, 0], event=y[:, 1])

    def reject_host_transfer(*args, **kwargs):
        raise AssertionError("packed GPU survival target was transferred to NumPy")

    monkeypatch.setattr(model, "_to_numpy", reject_host_transfer)
    score = model.score(X, y)
    assert np.isfinite(score)


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_penalized_cox_score_preserves_explicit_gpu_backend(device, monkeypatch):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA backend is unavailable: {exc}")
        X = cp.asarray([[1.0], [0.0], [-1.0]], dtype=cp.float64)
        y = cp.asarray([[1.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
        backend_name = "cupy"
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device is unavailable")
        X = torch.tensor(
            [[1.0], [0.0], [-1.0]], dtype=torch.float64, device="cuda"
        )
        y = torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=torch.float64,
            device="cuda",
        )
        backend_name = "torch"

    model = PenalizedCoxPHModel(
        device=device, compute_inference=False
    )
    model.coef_ = np.ones(1)
    model._selected_backend_name = backend_name

    import statgpu.linear_model.penalized._penalized_cox as module

    def reject_host_transfer(*args, **kwargs):
        raise AssertionError("full GPU score input was transferred to NumPy")

    monkeypatch.setattr(module, "_to_numpy", reject_host_transfer)
    assert model.score(X, y) == pytest.approx(1.0)
