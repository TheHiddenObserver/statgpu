"""Regression test: cummin/cummax on non-contiguous CuPy arrays."""
import numpy as np
import pytest


def test_cummin_cummax_noncontiguous():
    """cp.flip() returns non-contiguous views; cummin/cummax must handle them."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not installed")
    from statgpu.backends._cupy import CuPyBackend
    backend = CuPyBackend()

    # Contiguous case (baseline)
    x = cp.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result_cont = backend.cummin(x)
    expected = np.array([3.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(cp.asnumpy(result_cont), expected)

    # Non-contiguous case (flip)
    x_flipped = cp.flip(x)  # [5, 1, 4, 1, 3] — non-contiguous view
    result_flip = backend.cummin(x_flipped)
    expected_flip = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(cp.asnumpy(result_flip), expected_flip)

    # cummax non-contiguous
    result_flip_max = backend.cummax(x_flipped)
    expected_flip_max = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(cp.asnumpy(result_flip_max), expected_flip_max)


def test_cummin_cummax_2d_noncontiguous():
    """Non-contiguous 2D arrays (e.g. from transpose or slicing)."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not installed")
    from statgpu.backends._cupy import CuPyBackend
    backend = CuPyBackend()

    x = cp.array([[3.0, 1.0, 4.0], [2.0, 5.0, 0.0]])
    # Transpose makes columns non-contiguous
    x_t = cp.ascontiguousarray(x.T)  # ensure cont for expected
    x_t_nc = x.T  # non-contiguous view

    # cummin along axis=0 (down columns of transposed = across rows of original)
    result = backend.cummin(x_t_nc, axis=0)
    expected = backend.cummin(x_t, axis=0)
    np.testing.assert_array_equal(cp.asnumpy(result), cp.asnumpy(expected))
