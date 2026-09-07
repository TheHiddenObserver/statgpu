from types import SimpleNamespace

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

    monkeypatch.setattr(affinity, "_cupy_asarray_on_device", fake_align)
    X_out, y_out, w_out = affinity._align_cupy_cv_inputs(X_cv, y_cv, None)

    assert X_out is X_cv
    assert y_out.device.id == 2
    assert w_out is None
    assert calls == [("y", 2)]
