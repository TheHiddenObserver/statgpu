"""
NumPy / SciPy CPU backend.
"""

import numpy as np

from ._base import BackendBase


class NumpyBackend(BackendBase):
    """
    CPU backend powered by NumPy.

    This backend is always available and serves as the fallback when no GPU
    library is installed.
    """

    name = "numpy"

    @property
    def xp(self):
        return np

    def asarray(self, x, dtype=None):
        if hasattr(x, "get"):
            # CuPy arrays expose a .get() method that transfers the array from
            # GPU memory to a NumPy ndarray on the host.  We use duck-typing
            # here to avoid importing cupy when it may not be installed.
            x = x.get()
        elif hasattr(x, "cpu"):
            # PyTorch tensors expose a .cpu() method that moves the tensor to
            # CPU memory before converting to NumPy.  Duck-typing avoids a
            # mandatory torch import.
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype)

    def to_numpy(self, x) -> np.ndarray:
        return self.asarray(x)

    def is_available(self) -> bool:
        return True

    def lstsq(self, A, b, rcond=None):
        return np.linalg.lstsq(A, b, rcond=rcond)

    # ------------------------------------------------------------------
    # Helper methods for array operations (mirror CuPyBackend API)
    # ------------------------------------------------------------------

    def sum(self, x, axis=None, keepdims=False):
        """Sum over specified axis/axes."""
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        """Mean over specified axis/axes."""
        return np.mean(x, axis=axis, keepdims=keepdims)

    def sqrt(self, x):
        """Element-wise square root."""
        return np.sqrt(x)

    def abs(self, x):
        """Element-wise absolute value."""
        return np.abs(x)

    def max(self, x, axis=None, keepdims=False):
        """Maximum value along axis."""
        return np.max(x, axis=axis, keepdims=keepdims)

    def outer(self, a, b):
        """Outer product."""
        return np.outer(a.flatten(), b.flatten())

    def stack(self, arrays, axis=0):
        """Stack arrays along a new axis."""
        return np.stack(arrays, axis=axis)

    def zeros(self, shape, dtype=None):
        """Create array of zeros."""
        return np.zeros(shape, dtype=dtype)

    def array(self, val, dtype=None):
        """Create a scalar or array from a value."""
        return np.array(val, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        """Create range array."""
        if stop is None:
            result = np.arange(start, step=step)
        else:
            result = np.arange(start, stop, step=step)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def full(self, shape, fill_value, dtype=None):
        """Create array filled with a constant value."""
        return np.full(shape, fill_value, dtype=dtype)

    def atleast_1d(self, x):
        """Ensure array is at least 1D."""
        return np.atleast_1d(x)

    @property
    def newaxis(self):
        """Alias for None, used in indexing."""
        return np.newaxis

    @property
    def float64(self):
        """float64 dtype."""
        return np.float64

    @property
    def float32(self):
        """float32 dtype."""
        return np.float32

    @property
    def int64(self):
        """int64 dtype."""
        return np.int64

    @property
    def int32(self):
        """int32 dtype."""
        return np.int32

    def minimum(self, x, y):
        """Element-wise minimum of two arrays."""
        return np.minimum(x, y)

    def maximum(self, x, y):
        """Element-wise maximum of two arrays."""
        return np.maximum(x, y)

    def clip(self, x, min_val, max_val):
        """Clip values to [min_val, max_val]."""
        return np.clip(x, min_val, max_val)
