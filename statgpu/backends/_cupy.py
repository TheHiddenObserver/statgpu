"""
CuPy GPU backend.
"""

import numpy as np

from ._base import BackendBase


class CuPyBackend(BackendBase):
    """
    GPU backend powered by CuPy.

    Requires ``cupy`` (install via ``pip install statgpu[gpu11]`` for CUDA 11
    or ``pip install statgpu[gpu12]`` for CUDA 12).
    """

    name = "cupy"

    @property
    def xp(self):
        import cupy as cp  # deferred so import doesn't fail without cupy
        return cp

    def asarray(self, x, dtype=None):
        import cupy as cp
        if hasattr(x, "cpu"):
            # PyTorch tensors expose a .cpu() method that moves the tensor to
            # CPU memory before converting to NumPy.  Duck-typing avoids a
            # mandatory torch import.
            x = x.detach().cpu().numpy()
        return cp.asarray(x, dtype=dtype)

    def to_numpy(self, x) -> np.ndarray:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        # Fallback for numpy or other array-likes
        if hasattr(x, "get"):
            return x.get()
        return np.asarray(x)

    def is_available(self) -> bool:
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            return True
        except Exception:
            return False

    def lstsq(self, A, b, rcond=None):
        import cupy as cp
        # CuPy's lstsq signature matches NumPy's
        return cp.linalg.lstsq(A, b, rcond=rcond)

    def solve_triangular(self, A, b, lower=False, trans=False, unit_triangular=False):
        """
        Solve the triangular system Ax = b.

        Parameters
        ----------
        A : cupy.ndarray
            Triangular matrix (n, n).
        b : cupy.ndarray
            Right-hand side (n,) or (n, k).
        lower : bool, default=False
            Whether to use the lower triangle of A.
        trans : bool, default=False
            Whether to transpose A.
        unit_triangular : bool, default=False
            Whether to assume the diagonal of A is all ones.

        Returns
        -------
        x : cupy.ndarray
            Solution to the system.
        """
        import cupy as cp
        # Use cupyx.scipy.linalg.solve_triangular for proper triangular solve
        # This is much faster than generic solve for triangular systems
        try:
            from cupyx.scipy.linalg import solve_triangular
            return solve_triangular(A, b, lower=lower, trans=trans, unit_diagonal=unit_triangular)
        except ImportError:
            # Fallback to generic solve if cupyx.scipy not available
            return cp.linalg.solve(A, b)

    # ------------------------------------------------------------------
    # Helper methods for array operations
    # ------------------------------------------------------------------

    def sum(self, x, axis=None, keepdims=False):
        """Sum over specified axis/axes."""
        import cupy as cp
        return cp.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        """Mean over specified axis/axes."""
        import cupy as cp
        return cp.mean(x, axis=axis, keepdims=keepdims)

    def sqrt(self, x):
        """Element-wise square root."""
        import cupy as cp
        return cp.sqrt(x)

    def abs(self, x):
        """Element-wise absolute value."""
        import cupy as cp
        return cp.abs(x)

    def max(self, x, axis=None, keepdims=False):
        """Maximum value along axis."""
        import cupy as cp
        return cp.max(x, axis=axis, keepdims=keepdims)

    def outer(self, a, b):
        """Outer product."""
        import cupy as cp
        return cp.outer(a.flatten(), b.flatten())

    def stack(self, arrays, axis=0):
        """Stack arrays along a new axis."""
        import cupy as cp
        return cp.stack(arrays, axis=axis)

    def zeros(self, shape, dtype=None):
        """Create array of zeros."""
        import cupy as cp
        return cp.zeros(shape, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        """Create range array."""
        import cupy as cp
        if stop is None:
            result = cp.arange(start, step=step)
        else:
            result = cp.arange(start, stop, step=step)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def array(self, val, dtype=None):
        """Create a scalar or array from a value."""
        import cupy as cp
        return cp.array(val, dtype=dtype)

    def atleast_1d(self, x):
        """Ensure array is at least 1D."""
        import cupy as cp
        return cp.atleast_1d(x)

    @property
    def newaxis(self):
        """Alias for None, used in indexing."""
        import cupy as cp
        return cp.newaxis

    @property
    def float64(self):
        """float64 dtype."""
        import cupy as cp
        return cp.float64

    @property
    def float32(self):
        """float32 dtype."""
        import cupy as cp
        return cp.float32

    @property
    def int64(self):
        """int64 dtype."""
        import cupy as cp
        return cp.int64

    @property
    def int32(self):
        """int32 dtype."""
        import cupy as cp
        return cp.int32

    def clip(self, x, min_val, max_val):
        """Clip values to [min_val, max_val]."""
        import cupy as cp
        return cp.clip(x, min_val, max_val)

    def minimum(self, x, y):
        """Element-wise minimum of two arrays."""
        import cupy as cp
        return cp.minimum(x, y)

    def maximum(self, x, y):
        """Element-wise maximum of two arrays."""
        import cupy as cp
        return cp.maximum(x, y)

    def exp(self, x):
        """Element-wise exponential."""
        import cupy as cp
        return cp.exp(x)

    def log(self, x):
        """Element-wise natural logarithm."""
        import cupy as cp
        return cp.log(x)

    def copy(self, x):
        """Return a copy of x."""
        import cupy as cp
        return x.copy()

    def ones(self, shape, dtype=None):
        """Create array of ones."""
        import cupy as cp
        return cp.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        """Create array filled with a constant value."""
        import cupy as cp
        return cp.full(shape, fill_value, dtype=dtype)

    def diag(self, x, k=0):
        """Extract diagonal or create diagonal matrix."""
        import cupy as cp
        return cp.diag(x, k=k)

    def transpose(self, x, axes=None):
        """Transpose array."""
        import cupy as cp
        return cp.transpose(x, axes)

    def eye(self, n, m=None, dtype=None):
        """Create identity matrix."""
        import cupy as cp
        if m is None:
            m = n
        return cp.eye(n, m, dtype=dtype)

    def cummin(self, arr, axis=0):
        """Cumulative minimum along *axis* (CuPy fallback via CPU)."""
        import cupy as cp
        return cp.asarray(np.minimum.accumulate(cp.asnumpy(arr), axis=axis))

    def cummax(self, arr, axis=0):
        """Cumulative maximum along *axis* (CuPy fallback via CPU)."""
        import cupy as cp
        return cp.asarray(np.maximum.accumulate(cp.asnumpy(arr), axis=axis))
