import sys
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


class _FakeTorchArray:
    def __init__(self, device, label):
        self.device = device
        self.label = label


class _FakeDeviceContext:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __enter__(self):
        self.events.append(("enter", self.label))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.append(("exit", self.label))
        return False


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


def test_lassocv_torch_alignment_uses_exact_design_device_for_y_and_weights(monkeypatch):
    X_cv = _FakeTorchArray("cuda:3", "X")
    y_cv = _FakeTorchArray("cuda:0", "y")
    w_cv = _FakeTorchArray("cuda:0", "w")
    calls = []

    def fake_move(value, device=None, dtype=None, pin_memory=False):
        calls.append((value.label, device))
        return _FakeTorchArray(device, value.label)

    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: value is X_cv)
    monkeypatch.setattr(affinity, "_move_torch_tensor", fake_move)
    X_out, y_out, w_out = affinity._align_torch_cv_inputs(X_cv, y_cv, w_cv)

    assert X_out is X_cv
    assert y_out.device == "cuda:3"
    assert w_out.device == "cuda:3"
    assert calls == [("y", "cuda:3"), ("w", "cuda:3")]


def test_lassocv_torch_alignment_handles_missing_weights(monkeypatch):
    X_cv = _FakeTorchArray("cuda:2", "X")
    y_cv = _FakeTorchArray("cuda:0", "y")
    calls = []

    def fake_move(value, device=None, dtype=None, pin_memory=False):
        calls.append((value.label, device))
        return _FakeTorchArray(device, value.label)

    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: value is X_cv)
    monkeypatch.setattr(affinity, "_move_torch_tensor", fake_move)
    X_out, y_out, w_out = affinity._align_torch_cv_inputs(X_cv, y_cv, None)

    assert X_out is X_cv
    assert y_out.device == "cuda:2"
    assert w_out is None
    assert calls == [("y", "cuda:2")]


def test_lassocv_affinity_is_transparent_for_synthetic_backend_double(monkeypatch):
    X_cv = _FakeArray(4, "synthetic-X")
    y_cv = _FakeArray(0, "synthetic-y")
    w_cv = _FakeArray(0, "synthetic-w")

    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: False)
    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: False)

    X_out, y_out, w_out = affinity._align_cupy_cv_inputs(X_cv, y_cv, w_cv)
    assert X_out is X_cv
    assert y_out is y_cv
    assert w_out is w_cv
    X_out, y_out, w_out = affinity._align_torch_cv_inputs(X_cv, y_cv, w_cv)
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


def test_lassocv_resolved_torch_aligns_side_arrays_to_design_device(monkeypatch):
    model = LassoCV(device="torch", compute_inference=False)
    X = object()
    y = object()
    weight = object()
    X_cv = _FakeTorchArray("cuda:4", "X")
    y_cv = _FakeTorchArray("cuda:0", "y")
    w_cv = _FakeTorchArray("cuda:0", "w")
    converted = {X: X_cv, y: y_cv, weight: w_cv}
    conversion_calls = []
    move_calls = []

    def fake_to_array(value, device=None, backend=None):
        conversion_calls.append((value, device, backend))
        return converted[value]

    def fake_move(value, device=None, dtype=None, pin_memory=False):
        move_calls.append((value.label, device))
        return _FakeTorchArray(device, value.label)

    monkeypatch.setattr(model, "_to_array", fake_to_array)
    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: value is X_cv)
    monkeypatch.setattr(affinity, "_move_torch_tensor", fake_move)

    X_out, y_out, w_out = model._prepare_cv_inputs_for_resolved_device(
        X,
        y,
        weight,
        "torch",
    )

    assert X_out is X_cv
    assert y_out.device == "cuda:4"
    assert w_out.device == "cuda:4"
    assert conversion_calls == [
        (X, Device.TORCH, "torch"),
        (y, Device.TORCH, "torch"),
        (weight, Device.TORCH, "torch"),
    ]
    assert move_calls == [("y", "cuda:4"), ("w", "cuda:4")]


def test_lassocv_cupy_selector_binds_entire_transaction_to_design_device(monkeypatch):
    X = _FakeArray(3, "X")
    y = object()
    events = []

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda device_id: _FakeDeviceContext(("cupy", int(device_id)), events)
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: value is X)
    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: False)

    def delegate(X_arg, y_arg, *args, **kwargs):
        assert X_arg is X
        assert y_arg is y
        assert events == [("enter", ("cupy", 3))]
        return "selected"

    monkeypatch.setattr(affinity, "_ORIGINAL_SELECT", delegate)
    result = affinity._select_lasso_alpha_cv_on_design_device(X, y, cv_folds=3)

    assert result == "selected"
    assert events == [
        ("enter", ("cupy", 3)),
        ("exit", ("cupy", 3)),
    ]


def test_lassocv_torch_selector_binds_entire_transaction_to_design_device(monkeypatch):
    X = _FakeTorchArray("cuda:5", "X")
    y = object()
    events = []

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device=lambda device: _FakeDeviceContext(("torch", str(device)), events)
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(affinity, "_is_cupy_array", lambda value: False)
    monkeypatch.setattr(affinity, "_is_torch_array", lambda value: value is X)

    def delegate(X_arg, y_arg, *args, **kwargs):
        assert X_arg is X
        assert y_arg is y
        assert events == [("enter", ("torch", "cuda:5"))]
        return "selected"

    monkeypatch.setattr(affinity, "_ORIGINAL_SELECT", delegate)
    result = affinity._select_lasso_alpha_cv_on_design_device(X, y, cv_folds=3)

    assert result == "selected"
    assert events == [
        ("enter", ("torch", "cuda:5")),
        ("exit", ("torch", "cuda:5")),
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