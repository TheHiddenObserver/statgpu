"""Tests for top-level exports and Ridge/Lasso penalty behavior."""

import numpy as np
import pytest

import statgpu
from statgpu.linear_model import Ridge, Lasso
from statgpu._config import set_device, Device


def test_top_level_logistic_export():
    """Top-level package should expose LogisticRegression."""
    assert hasattr(statgpu, "LogisticRegression")


@pytest.mark.parametrize("model_cls", [Ridge, Lasso])
def test_penalty_models_basic_predict_shape(model_cls):
    """Penalty models should predict 1D output matching sample size."""
    set_device("cpu")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8))
    beta = rng.normal(size=8)
    y = X @ beta + rng.normal(scale=0.2, size=120)

    model = model_cls(device="cpu")
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (120,)
    assert np.all(np.isfinite(pred))


@pytest.mark.skipif(
    not Ridge(device="auto")._get_compute_device() == Device.CUDA,
    reason="CUDA not available",
)
@pytest.mark.parametrize("model_cls", [Ridge, Lasso])
def test_penalty_models_gpu_cpu_prediction_consistency(model_cls):
    """GPU and CPU predictions should be numerically close."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(256, 16)).astype(np.float64)
    beta = rng.normal(size=16)
    y = X @ beta + rng.normal(scale=0.3, size=256)

    cpu_model = model_cls(device="cpu")
    cpu_model.fit(X, y)
    cpu_pred = cpu_model.predict(X)

    gpu_model = model_cls(device="cuda")
    gpu_model.fit(X, y)
    gpu_pred = gpu_model.predict(X)
    if hasattr(gpu_pred, "get"):
        gpu_pred = gpu_pred.get()
    gpu_pred = np.asarray(gpu_pred)

    assert np.allclose(cpu_pred, gpu_pred, rtol=1e-4, atol=1e-4)
