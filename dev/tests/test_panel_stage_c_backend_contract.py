"""Stage-C backend contracts that can be enforced without a physical GPU."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from statgpu import backends
from statgpu.backends._cupy import CuPyBackend
from statgpu.panel import _linalg, _utils


def test_cupy_scatter_add_never_routes_values_through_host(monkeypatch):
    """The CuPy scatter path must remain backend-native even if cupyx is absent."""

    def _forbid_host_conversion(*args, **kwargs):
        raise AssertionError("CuPy scatter_add attempted a host conversion")

    monkeypatch.setattr(_utils, "_to_numpy", _forbid_host_conversion)
    fake_cupy = SimpleNamespace(
        __name__="cupy",
        zeros=np.zeros,
        add=np.add,
    )
    indices = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    values = np.array([1.5, -2.0, 0.5, 4.0, 3.0], dtype=np.float64)

    actual = _utils._scatter_add(fake_cupy, indices, values, n_groups=3)
    assert_allclose(actual, np.array([2.0, 1.0, 4.0]))


def test_cupy_availability_probe_does_not_switch_current_device(monkeypatch):
    """Backend discovery must not force CUDA device 0 as a side effect."""

    class ForbiddenDevice:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("availability probe attempted to construct/switch a device")

    class Runtime:
        @staticmethod
        def getDeviceCount():
            return 2

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(Device=ForbiddenDevice, runtime=Runtime()),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert CuPyBackend().is_available() is True


def test_cupy_reference_creation_helpers_enter_reference_device_context():
    """Backend creation helpers must not allocate on an unrelated current GPU."""

    class FakeDevice:
        def __init__(self):
            self.active = False
            self.entries = 0

        def __enter__(self):
            assert not self.active
            self.active = True
            self.entries += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            self.active = False
            return False

    class FakeCuPyRef:
        __module__ = "cupy.fake"

        def __init__(self, device):
            self.device = device

    class FakeCuPyNamespace:
        __name__ = "cupy"

        def __init__(self, device):
            self.device = device
            self.calls = []

        def _record(self, name, value):
            assert self.device.active, f"{name} allocated outside reference device"
            self.calls.append(name)
            return value

        def zeros(self, shape, dtype=None):
            return self._record("zeros", np.zeros(shape, dtype=dtype))

        def eye(self, n, dtype=None):
            return self._record("eye", np.eye(n, dtype=dtype))

        def full(self, shape, fill_value, dtype=None):
            return self._record("full", np.full(shape, fill_value, dtype=dtype))

        def empty(self, shape, dtype=None):
            return self._record("empty", np.empty(shape, dtype=dtype))

        def arange(self, n, dtype=None):
            return self._record("arange", np.arange(n, dtype=dtype))

        def ones(self, shape, dtype=None):
            return self._record("ones", np.ones(shape, dtype=dtype))

    device = FakeDevice()
    ref = FakeCuPyRef(device)
    xp = FakeCuPyNamespace(device)

    backends.xp_zeros((2,), np.float64, xp, ref)
    backends.xp_eye(2, np.float64, xp, ref)
    backends.xp_full((2,), 3.0, np.float64, xp, ref)
    backends.xp_empty((2,), np.float64, xp, ref)
    backends.xp_arange(2, dtype=np.int64, xp=xp, ref_arr=ref)
    backends.xp_ones((2,), np.float64, xp, ref)

    assert xp.calls == ["zeros", "eye", "full", "empty", "arange", "ones"]
    assert device.entries == 6
    assert not device.active


def test_panel_linalg_identity_uses_reference_device_helper(monkeypatch):
    """Gram certification must source identity matrices from the design device."""
    calls = []
    original = _linalg.xp_eye

    def spy_eye(n, dtype, xp, ref_arr=None):
        calls.append(ref_arr)
        return original(n, dtype, xp, ref_arr)

    monkeypatch.setattr(_linalg, "xp_eye", spy_eye)
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    _linalg.panel_working_pseudoinverse(X, np)
    assert calls and calls[-1] is X

    X_batch = np.stack([X, X + np.array([0.0, 0.25])], axis=0)
    y_batch = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=np.float64,
    )
    _linalg.panel_lstsq_gram_certified_batched(X_batch, y_batch, np)
    assert calls[-1] is X_batch
