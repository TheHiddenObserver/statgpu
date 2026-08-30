"""Formula-path regression for estimation-only accelerator cleanup ownership."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def test_formula_no_inference_gpu_fit_does_not_repeat_backend_cleanup(monkeypatch):
    pd = pytest.importorskip("pandas")
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin_module
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []

    class FakeArray:
        def __init__(self, shape, device=1):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.device = types.SimpleNamespace(id=int(device))

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=FakeDevice))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        fit_mixin_module,
        "_cupy_asarray_on_device",
        lambda value, target_device, dtype=None: value,
    )

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=True,
        device="cpu",
        solver="exact",
        compute_inference=False,
        gpu_memory_cleanup=True,
    )
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model, "_auto_backend_override", lambda backend_name, X: backend_name
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "exact",
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, device=None, backend=None: FakeArray(np.asarray(value).shape),
    )
    monkeypatch.setattr(
        model, "_cleanup_cuda_memory", lambda: events.append("cleanup")
    )

    def fake_fit_gpu(X, y, sample_weight=None):
        events.append("fit")
        model.coef_ = np.zeros(int(X.shape[1]), dtype=float)
        model.intercept_ = 0.0
        model._params = model.coef_.copy()
        model._cleanup_cuda_memory()

    monkeypatch.setattr(model, "_fit_gpu", fake_fit_gpu)

    frame = pd.DataFrame(
        {
            "y": [-1.2, 0.1, 0.4, 2.2, 1.6, 3.8],
            "x1": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "x2": [0.5, 1.5, -0.5, 2.0, 0.25, 1.0],
        }
    )
    model.fit(formula="y ~ x1 + x2", data=frame)

    assert events == ["fit", "cleanup"]
    assert model._fitted is True
