from types import SimpleNamespace

import numpy as np

from statgpu._config import Device
from statgpu.linear_model import LassoCV
from statgpu.linear_model import _lassocv_device_affinity_contract as affinity
from statgpu.linear_model.wrappers import _lasso as lasso_impl


class _FakeArray:
    def __init__(self, device_id, label):
        self.device = SimpleNamespace(id=device_id)
        self.label = label


def test_lassocv_cupy_alignment_uses_design_device_for_y_and_weights(monkeypatch):
    X_cv = _FakeArray(3, "X")
    y_cv = _FakeArray(0, "y")
    w_cv = _FakeArray(0, "w")
    calls = []

    def fake_align(value, device_id):
        calls.append((value.label, device_id))
        return _FakeArray(device_id, value.label)

    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: value is X_cv)
    monkeypatch.setattr(affinity, "_cupy_asarray_on_device", fake_align)
    X_out, y_out, w_out = affinity._align_cupy_cv_inputs(X_cv, y_cv, w_cv)

    assert X_out is X_cv
    assert y_out.device.id == 3
    assert w_out.device.id == 3
    assert calls == [("y", 3), ("w", 3)]


def test_lassocv_cupy_alignment_handles_missing_weights(monkeypatch):
    X_cv = _FakeArray(2, "X")
    y_cv = _FakeArray(0, "y")
    calls = []

    def fake_align(value, device_id):
        calls.append((value.label, device_id))
        return _FakeArray(device_id, value.label)

    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: value is X_cv)
    monkeypatch.setattr(affinity, "_cupy_asarray_on_device", fake_align)
    X_out, y_out, w_out = affinity._align_cupy_cv_inputs(X_cv, y_cv, None)

    assert X_out is X_cv
    assert y_out.device.id == 2
    assert w_out is None
    assert calls == [("y", 2)]


def test_lassocv_affinity_is_transparent_for_synthetic_backend_double(monkeypatch):
    X_cv = _FakeArray(4, "synthetic-X")
    y_cv = _FakeArray(0, "synthetic-y")
    w_cv = _FakeArray(0, "synthetic-w")

    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: False)

    X_out, y_out, w_out = affinity._align_cupy_cv_inputs(X_cv, y_cv, w_cv)
    assert X_out is X_cv
    assert y_out is y_cv
    assert w_out is w_cv


def test_lassocv_resolved_cpu_converts_all_operands_to_numpy_backend(monkeypatch):
    model = LassoCV(device="cpu", compute_inference=False)
    X = object()
    y = object()
    weight = object()
    converted = {}
    calls = []

    def fake_to_array(value, device=None, backend=None):
        calls.append((value, device, backend))
        result = object()
        converted[value] = result
        return result

    monkeypatch.setattr(model, "_to_array", fake_to_array)

    X_out, y_out, w_out = model._prepare_cv_inputs_for_resolved_device(
        X,
        y,
        weight,
        "cpu",
    )

    assert X_out is converted[X]
    assert y_out is converted[y]
    assert w_out is converted[weight]
    assert calls == [
        (X, Device.CPU, "numpy"),
        (y, Device.CPU, "numpy"),
        (weight, Device.CPU, "numpy"),
    ]


def test_lassocv_auto_resolved_backend_is_pinned_through_final_refit(monkeypatch):
    rng = np.random.default_rng(13817)
    X = rng.normal(size=(36, 4))
    y = X @ np.array([1.0, -0.5, 0.25, 0.0]) + rng.normal(scale=0.2, size=36)
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="auto",
        compute_inference=False,
        random_state=7,
    )
    observed = {}

    monkeypatch.setattr(model, "_get_compute_device", lambda: Device.CUDA)
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, device=None, backend=None: np.asarray(value),
    )

    def synthetic_selection(X_cv, y_cv, **kwargs):
        assert kwargs["device"] == "cuda"
        return {
            "alpha": 0.03,
            "alphas": np.asarray([0.03, 0.08], dtype=np.float64),
            "mse_path": np.asarray(
                [[0.2, 0.21, 0.19], [0.3, 0.31, 0.29]], dtype=np.float64
            ),
            "mean_mse": np.asarray([0.2, 0.3], dtype=np.float64),
        }

    def successful_refit(self, *args, **kwargs):
        observed["refit_device"] = self._device
        self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self.n_iter_ = 1
        self._fitted = True
        return self

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", synthetic_selection)
    monkeypatch.setattr(lasso_impl.Lasso, "fit", successful_refit)

    model.fit(X, y)

    assert observed["refit_device"] is Device.CUDA
    assert model._device is Device.AUTO
    assert model.device == "auto"
