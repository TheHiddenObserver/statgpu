
import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device


class DummyEstimator(BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        self._fitted = True
        return self

    def predict(self, X):
        return X


def test_model_context_resolves_torch_backend():
    model = DummyEstimator(device=Device.TORCH)
    assert model._resolve_inference_backend("auto") == "torch"
    assert model._resolve_inference_backend("numpy") == "numpy"


def test_inference_cast_helper_preserves_numpy(monkeypatch):
    model = DummyEstimator(device=Device.CPU)
    value = np.array([0.1, 0.2])
    out = model._cast_inference_array(value, "numpy")
    assert out is value

    calls = []
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda x, device, backend=None: calls.append((device, backend)) or x,
    )
    model._cast_inference_array(value, "torch")
    model._cast_inference_array(value, "cupy")
    assert calls == [
        (Device.TORCH, "torch"),
        (Device.CUDA, "cupy"),
    ]
