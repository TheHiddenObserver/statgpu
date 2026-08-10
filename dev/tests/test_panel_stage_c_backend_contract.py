"""Stage-C backend contracts that can be enforced without a physical GPU."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import _utils


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
