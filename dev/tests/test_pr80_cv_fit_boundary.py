"""Regression tests for CoxPHCV fit-time control boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv_module


def _require_backend(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA unavailable: {exc}")
        return cp
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")
    return torch


def _on_backend(device, value):
    xp = _require_backend(device)
    if device == "cuda":
        return xp.asarray(value)
    return xp.as_tensor(value, dtype=xp.float64, device="cuda")


def _cv_sample(seed=2290, n=48, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.3, -0.15, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=2.0, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[:4] = 1.0
    return X, stop, event


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("ties", "not-a-tie-method", "ties must be"),
        ("cov_type", "not-a-covariance", "cov_type must be"),
        ("inference_mode", "not-an-inference-mode", "inference_mode must be"),
        ("device", "not-a-device", "device must be"),
        ("compute_inference", "False", "compute_inference must be"),
        ("gpu_memory_cleanup", "False", "gpu_memory_cleanup must be"),
    ],
)
def test_invalid_cv_control_fails_before_selector_and_clears_state(
    parameter, value, message, monkeypatch
):
    X, stop, event = _cv_sample()
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        device="cpu",
        compute_inference=False,
    ).fit(X, stop, event)
    assert model._fitted is True
    setattr(model, parameter, value)

    def forbidden_selector(*_args, **_kwargs):
        raise AssertionError("invalid CoxPHCV control reached fold selection")

    monkeypatch.setattr(
        cox_cv_module, "_select_coxph_penalty_cv", forbidden_selector
    )
    with pytest.raises(ValueError, match=message):
        model.fit(X, stop, event)

    assert model._fitted is False
    assert model.estimator_ is None
    assert model.coef_ is None
    assert model.cv_results_ is None


def test_exact_robust_cv_fails_before_selector(monkeypatch):
    X, stop, event = _cv_sample(seed=2291)
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        ties="exact",
        cov_type="hc0",
        compute_inference=True,
        device="cpu",
    )

    def forbidden_selector(*_args, **_kwargs):
        raise AssertionError("unsupported Exact robust CV reached fold selection")

    monkeypatch.setattr(
        cox_cv_module, "_select_coxph_penalty_cv", forbidden_selector
    )
    with pytest.raises(NotImplementedError, match="robust covariance"):
        model.fit(X, stop, event)

    assert model._fitted is False
    assert model.estimator_ is None


def test_cv_controls_are_canonicalized_before_fitting():
    X, stop, event = _cv_sample(seed=2292)
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        ties="EFRON",
        cov_type="NONROBUST",
        inference_mode="STRICT",
        compute_inference=0,
        gpu_memory_cleanup=0,
        device="cpu",
        max_iter=60,
        tol=1e-7,
    ).fit(X, stop, event)

    assert model.ties == "efron"
    assert model.cov_type == "nonrobust"
    assert model.inference_mode == "strict"
    assert model.compute_inference is False
    assert model.gpu_memory_cleanup is False
    assert model.device is Device.CPU
    assert model.estimator_ is not None
    assert model.estimator_._bse is None
    assert np.all(np.isfinite(model.coef_))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda", Device.CUDA), ("torch", Device.TORCH)],
)
def test_cv_gpu_device_normalization_reaches_final_refit(device, expected):
    _require_backend(device)
    X_np, stop_np, event_np = _cv_sample(seed=2293, n=36, p=2)
    X = _on_backend(device, X_np)
    stop = _on_backend(device, stop_np)
    event = _on_backend(device, event_np)
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=50,
    )
    model.set_params(device=device)
    model.fit(X, stop, event)

    assert model.device is expected
    assert model.estimator_ is not None
    assert model.estimator_.device is expected
    assert model.effective_device_ == device
    assert model._fitted is True
    assert np.all(np.isfinite(model.coef_))
