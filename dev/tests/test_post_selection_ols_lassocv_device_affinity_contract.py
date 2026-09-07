from types import SimpleNamespace

from statgpu._config import Device
from statgpu.linear_model import LassoCV
from statgpu.linear_model import _lassocv_device_affinity_contract as affinity


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
