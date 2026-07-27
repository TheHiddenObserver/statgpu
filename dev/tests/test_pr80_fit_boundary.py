"""Regression tests for the final PR #80 public CoxPH input boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.survival import CoxPH


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
        dtype = xp.complex128 if np.iscomplexobj(value) else xp.float64
        return xp.as_tensor(value, dtype=dtype, device="cuda")
    return np.asarray(value)


def _stable_sample(seed=2280, n=80, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.35, -0.2, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=1.8, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[0] = 1.0
    return X, stop, event


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


def test_pandas_packed_target_remains_supported():
    pd = pytest.importorskip("pandas")
    X, stop, event = _stable_sample(p=1)
    target = pd.DataFrame({"time": stop, "event": event})
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, target)
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


@pytest.mark.parametrize("invalid", ["scalar_x", "bad_packed_target"])
def test_adapter_validation_failure_clears_stale_fit_state(invalid):
    X, stop, event = _stable_sample(p=1)
    target = np.column_stack((stop, event))
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, target)
    assert model._fitted is True
    assert model.coef_ is not None

    with pytest.raises(ValueError):
        if invalid == "scalar_x":
            model.fit(np.asarray(1.0), np.array([[1.0, 1.0]]))
        else:
            model.fit(X, stop)

    assert model._fitted is False
    assert model.coef_ is None
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X[:2])


def test_scalar_X_has_public_validation_error():
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        CoxPH(device="cpu", compute_inference=False).fit(
            np.asarray(1.0), np.array([[1.0, 1.0]])
        )


def test_zero_feature_design_has_public_validation_error():
    target = np.array(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0]]
    )
    with pytest.raises(ValueError, match="at least one feature"):
        CoxPH(device="cpu", compute_inference=False).fit(
            np.empty((4, 0)), target
        )


@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
@pytest.mark.parametrize(
    "method", ["predict_risk_score", "predict_hazard_ratio", "predict_survival"]
)
def test_public_predictions_reject_complex_X_before_cast(device, method):
    if device != "cpu":
        _require_backend(device)
    X_np, stop_np, event_np = _stable_sample()
    X = _on_backend(device, X_np)
    stop = _on_backend(device, stop_np)
    event = _on_backend(device, event_np)
    model = CoxPH(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=80,
    ).fit(X, stop, event)
    complex_X = _on_backend(
        device, X_np[:3].astype(np.complex128) + 1j
    )

    with pytest.raises(ValueError, match="X must be real-valued"):
        getattr(model, method)(complex_X)


@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
def test_predict_survival_rejects_complex_times_before_cast(device):
    if device != "cpu":
        _require_backend(device)
    X_np, stop_np, event_np = _stable_sample(seed=2281)
    X = _on_backend(device, X_np)
    stop = _on_backend(device, stop_np)
    event = _on_backend(device, event_np)
    model = CoxPH(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=80,
    ).fit(X, stop, event)
    complex_times = _on_backend(
        device, np.array([0.5 + 1j], dtype=np.complex128)
    )

    with pytest.raises(ValueError, match="times must be real-valued"):
        model.predict_survival(X[:2], times=complex_times)
