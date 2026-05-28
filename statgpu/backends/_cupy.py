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
        """Cumulative minimum along *axis* (GPU-native for small arrays)."""
        import cupy as cp
        if arr.ndim == 1:
            return self._cumop_1d(arr, cp.minimum)
        # Multi-dim: transpose target axis to last, scan, transpose back
        if axis != arr.ndim - 1:
            axes = list(range(arr.ndim))
            axes[axis], axes[-1] = axes[-1], axes[axis]
            arr = cp.transpose(arr, axes)
            return cp.transpose(self._cumop_last_axis(arr, cp.minimum), axes)
        return self._cumop_last_axis(arr, cp.minimum)

    def cummax(self, arr, axis=0):
        """Cumulative maximum along *axis* (GPU-native for small arrays)."""
        import cupy as cp
        if arr.ndim == 1:
            return self._cumop_1d(arr, cp.maximum)
        if axis != arr.ndim - 1:
            axes = list(range(arr.ndim))
            axes[axis], axes[-1] = axes[-1], axes[axis]
            arr = cp.transpose(arr, axes)
            return cp.transpose(self._cumop_last_axis(arr, cp.maximum), axes)
        return self._cumop_last_axis(arr, cp.maximum)

    @staticmethod
    def _cumop_1d(arr, op):
        """1D cumulative op using sequential write."""
        import cupy as cp
        # Ensure contiguous memory for CUDA kernel
        arr = cp.ascontiguousarray(arr)
        n = len(arr)
        result = cp.empty_like(arr)
        result[0] = arr[0]
        if n > 1:
            _launch_cumop_1d(arr, result, n, op is cp.minimum)
        return result

    @staticmethod
    def _cumop_last_axis(arr, op):
        """Cumulative op along last axis for N-D arrays."""
        import cupy as cp
        shape = arr.shape
        K = shape[-1]
        flat = arr.reshape(-1, K)
        # Ensure contiguous memory for CUDA kernel
        flat = cp.ascontiguousarray(flat)
        N = flat.shape[0]
        result = cp.empty_like(flat)
        result[:, 0] = flat[:, 0]
        if K > 1:
            _launch_cumop_2d(flat, result, N, K, op is cp.minimum)
        return result.reshape(shape)


# ── Raw CUDA kernels for cumulative scan ──
_cummin_1d_src = r'''
extern "C" __global__
void cummin_1d(const double* __restrict__ x,
               double* __restrict__ out, int n) {
    double min_val = x[0];
    out[0] = min_val;
    for (int j = 1; j < n; j++) {
        if (x[j] < min_val) min_val = x[j];
        out[j] = min_val;
    }
}
'''

_cummax_1d_src = r'''
extern "C" __global__
void cummax_1d(const double* __restrict__ x,
               double* __restrict__ out, int n) {
    double max_val = x[0];
    out[0] = max_val;
    for (int j = 1; j < n; j++) {
        if (x[j] > max_val) max_val = x[j];
        out[j] = max_val;
    }
}
'''

_cummin_2d_src = r'''
extern "C" __global__
void cummin_2d(const double* __restrict__ x,
               double* __restrict__ out, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const double* row = x + tid * K;
    double* orow = out + tid * K;
    double min_val = row[0];
    orow[0] = min_val;
    for (int j = 1; j < K; j++) {
        if (row[j] < min_val) min_val = row[j];
        orow[j] = min_val;
    }
}
'''

_cummax_2d_src = r'''
extern "C" __global__
void cummax_2d(const double* __restrict__ x,
               double* __restrict__ out, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const double* row = x + tid * K;
    double* orow = out + tid * K;
    double max_val = row[0];
    orow[0] = max_val;
    for (int j = 1; j < K; j++) {
        if (row[j] > max_val) max_val = row[j];
        orow[j] = max_val;
    }
}
'''

# Compile once at import time (guarded: cupy may not be installed)
try:
    import cupy as _cp
    _cummin_1d_mod = _cp.RawModule(code=_cummin_1d_src)
    _cummax_1d_mod = _cp.RawModule(code=_cummax_1d_src)
    _cummin_2d_mod = _cp.RawModule(code=_cummin_2d_src)
    _cummax_2d_mod = _cp.RawModule(code=_cummax_2d_src)
    _cummin_1d_kernel = _cummin_1d_mod.get_function('cummin_1d')
    _cummax_1d_kernel = _cummax_1d_mod.get_function('cummax_1d')
    _cummin_2d_kernel = _cummin_2d_mod.get_function('cummin_2d')
    _cummax_2d_kernel = _cummax_2d_mod.get_function('cummax_2d')
    del _cp
except ImportError:
    _cummin_1d_kernel = None
    _cummax_1d_kernel = None
    _cummin_2d_kernel = None
    _cummax_2d_kernel = None


def _launch_cumop_1d(arr, result, n, is_min):
    import cupy as cp
    kernel = _cummin_1d_kernel if is_min else _cummax_1d_kernel
    kernel((1,), (1,), (arr, result, n))


def _launch_cumop_2d(arr, result, N, K, is_min):
    import cupy as cp
    kernel = _cummin_2d_kernel if is_min else _cummax_2d_kernel
    block = min(N, 256)
    grid = (N + block - 1) // block
    kernel((grid,), (block,), (arr, result, N, K))
