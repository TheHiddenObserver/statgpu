"""Regression tests for the final PR #80 statistical and GPU-boundary fixes."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.survival import CoxPH
from statgpu.survival._cox_counting import fit_counting_process_cox
from statgpu.survival._cox_score_residuals import cox_score_residuals
from statgpu.survival._risk_sets import cox_counting_process_objective


def _require_backend(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA unavailable")
        return cp
    if device == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        return torch
    return np


def _on_backend(device, value):
    xp = _require_backend(device)
    if device == "cuda":
        return xp.asarray(value)
    if device == "torch":
        return xp.as_tensor(value, dtype=xp.float64, device="cuda")
    return np.asarray(value)


def _to_numpy(value):
    if hasattr(value, "get"):
        return value.get()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_efron_tied_residuals_sum_to_efron_score_not_breslow_score():
    X = np.array([[0.0], [1.0], [3.0]])
    stop = np.array([1.0, 1.0, 2.0])
    event = np.array([1.0, 1.0, 0.0])
    beta = np.array([0.0])

    efron = cox_counting_process_objective(
        beta, X, stop, event, ties="efron"
    )
    breslow = cox_counting_process_objective(
        beta, X, stop, event, ties="breslow"
    )
    residuals = cox_score_residuals(
        beta, X, stop, event, ties="efron"
    )

    assert_allclose(residuals.sum(axis=0), efron["score"], atol=1e-14, rtol=0)
    assert not np.allclose(efron["score"], breslow["score"])


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_score_residuals_sum_to_score_with_entry_and_strata(ties):
    X = np.array(
        [
            [-1.0, 0.2],
            [0.5, -0.4],
            [1.2, 0.7],
            [-0.3, 1.1],
            [0.8, -1.0],
            [1.5, 0.3],
        ]
    )
    start = np.array([0.0, 0.0, 0.5, 0.0, 0.4, 0.0])
    stop = np.array([1.0, 1.0, 2.0, 1.5, 1.5, 2.5])
    event = np.array([1, 1, 0, 1, 1, 0])
    strata = np.array([0, 0, 0, 1, 1, 1])
    beta = np.array([0.25, -0.15])

    objective = cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
    )
    residuals = cox_score_residuals(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
    )
    assert_allclose(
        residuals.sum(axis=0), objective["score"], atol=2e-13, rtol=2e-13
    )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_counting_solver_returns_tie_consistent_score_residuals(ties):
    X = np.array([[0.0], [1.0], [3.0], [-0.5]])
    stop = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0, 0.0])
    result = fit_counting_process_cox(
        X,
        stop,
        event,
        ties=ties,
        compute_baseline=False,
        compute_score_residuals=True,
        max_iter=40,
    )
    assert_allclose(
        result["score_residuals"].sum(axis=0),
        result["score"],
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cuda", "torch"])
@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_gpu_score_residuals_sum_to_backend_score(device, ties):
    X_np = np.array(
        [[-1.0, 0.2], [0.5, -0.4], [1.2, 0.7], [-0.3, 1.1], [0.8, -1.0]]
    )
    stop_np = np.array([1.0, 1.0, 2.0, 1.5, 1.5])
    event_np = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    strata_np = np.array([0, 0, 0, 1, 1])
    beta_np = np.array([0.25, -0.15])

    X = _on_backend(device, X_np)
    stop = _on_backend(device, stop_np)
    event = _on_backend(device, event_np)
    strata = _on_backend(device, strata_np)
    beta = _on_backend(device, beta_np)
    objective = cox_counting_process_objective(
        beta, X, stop, event, strata=strata, ties=ties
    )
    residuals = cox_score_residuals(
        beta, X, stop, event, strata=strata, ties=ties
    )
    assert_allclose(
        _to_numpy(residuals.sum(axis=0)),
        _to_numpy(objective["score"]),
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
def test_packed_target_fit_avoids_public_to_numpy_and_clears_ordinary_entry(
    device, monkeypatch
):
    if device != "cpu":
        _require_backend(device)
    X_np = np.array([[-1.0], [0.0], [1.0], [2.0]])
    target_np = np.array(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0]]
    )
    X = _on_backend(device, X_np)
    target = _on_backend(device, target_np)
    model = CoxPH(
        device=device,
        compute_inference=False,
        compute_cindex=False,
        max_iter=40,
    )

    def reject_to_numpy(*_args, **_kwargs):
        raise AssertionError("packed survival target crossed the public host boundary")

    monkeypatch.setattr(model, "_to_numpy", reject_to_numpy)
    model.fit(X, target)
    assert model._entry is None
    assert np.all(np.isfinite(model.coef_))


def test_three_column_packed_target_preserves_real_entry_state():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    target = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.2, 2.0, 1.0],
            [0.5, 3.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, target)
    assert model._is_counting_process is True
    assert_allclose(model._entry, target[:, 0])


def test_scalar_X_has_public_validation_error():
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        CoxPH(device="cpu", compute_inference=False).fit(
            np.asarray(1.0), np.array([[1.0, 1.0]])
        )
