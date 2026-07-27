"""Regression tests for the final PR #80 GPU fit-boundary fixes."""

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
        return xp.as_tensor(value, dtype=xp.float64, device="cuda")
    return np.asarray(value)


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
