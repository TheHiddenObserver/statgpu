"""
CuPy GPU backend.
"""

import numpy as np

from statgpu.backends._base import BackendBase
from statgpu.backends._utils import _torch_to_cupy_dlpack


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
            arr = _torch_to_cupy_dlpack(x)
            if arr is not None:
                return arr.astype(dtype, copy=False) if dtype is not None else arr
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

    def matmul(self, a, b):
        """Matrix multiplication."""
        import cupy as cp
        return cp.matmul(a, b)

    def min(self, x, axis=None, keepdims=False):
        """Minimum value along axis."""
        import cupy as cp
        return cp.min(x, axis=axis, keepdims=keepdims)

    def expand_dims(self, x, axis):
        """Expand array dimensions."""
        import cupy as cp
        return cp.expand_dims(x, axis)

    def eigh(self, a):
        """Eigenvalue decomposition for symmetric/Hermitian matrices."""
        import cupy as cp
        return cp.linalg.eigh(a)

    def argmin(self, x, axis=None):
        """Indices of minimum values along axis."""
        import cupy as cp
        return cp.argmin(x, axis=axis)

    def argmax(self, x, axis=None):
        """Indices of maximum values along axis."""
        import cupy as cp
        return cp.argmax(x, axis=axis)

    def argsort(self, x, axis=-1):
        """Indices that would sort the array."""
        import cupy as cp
        return cp.argsort(x, axis=axis)

    def where(self, condition, x, y):
        """Element-wise conditional selection."""
        import cupy as cp
        return cp.where(condition, x, y)

    def flip(self, x, axis=None):
        """Reverse array order along axis."""
        import cupy as cp
        return cp.flip(x, axis=axis)

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
        if arr.size == 0 or arr.shape[axis] == 0:
            return arr.copy()
        if str(arr.dtype) not in _CUPY_CUMOP_DTYPES:
            return cp.minimum.accumulate(arr, axis=axis)
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
        if arr.size == 0 or arr.shape[axis] == 0:
            return arr.copy()
        if str(arr.dtype) not in _CUPY_CUMOP_DTYPES:
            return cp.maximum.accumulate(arr, axis=axis)
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
        # Ensure contiguous for CUDA kernel compatibility
        if not arr.flags.c_contiguous:
            arr = cp.ascontiguousarray(arr)
        n = len(arr)
        if n == 0:
            return cp.empty_like(arr)
        result = cp.empty_like(arr)
        result[0] = arr[0]
        if n > 1:
            _launch_cumop_1d(arr, result, n, op is cp.minimum)
        return result

    @staticmethod
    def _cumop_last_axis(arr, op):
        """Cumulative op along last axis for N-D arrays."""
        import cupy as cp
        # Ensure contiguous for CUDA kernel compatibility
        if not arr.flags.c_contiguous:
            arr = cp.ascontiguousarray(arr)
        shape = arr.shape
        K = shape[-1]
        if K == 0:
            return cp.empty_like(arr)
        flat = arr.reshape(-1, K)
        N = flat.shape[0]
        if N == 0:
            return cp.empty_like(arr)
        result = cp.empty_like(flat)
        result[:, 0] = flat[:, 0]
        if K > 1:
            _launch_cumop_2d(flat, result, N, K, op is cp.minimum)
        return result.reshape(shape)

    # ── Linear algebra ──

    def qr(self, a, mode='reduced'):
        """QR decomposition."""
        import cupy as cp
        return cp.linalg.qr(a, mode=mode)

    def svd(self, a, full_matrices=True):
        """Singular value decomposition."""
        import cupy as cp
        return cp.linalg.svd(a, full_matrices=full_matrices)

    def solve(self, A, b):
        """Solve the linear system Ax = b."""
        import cupy as cp
        return cp.linalg.solve(A, b)

    def norm(self, x, ord=None, axis=None):
        """Compute matrix or vector norm."""
        import cupy as cp
        return cp.linalg.norm(x, ord=ord, axis=axis)

    # ── Dtype properties ──

    @property
    def bool(self):
        """Boolean dtype."""
        import cupy as cp
        return cp.bool_

    @property
    def nan(self):
        """NaN value."""
        import cupy as cp
        return cp.nan

    @property
    def inf(self):
        """Infinity value."""
        import cupy as cp
        return cp.inf

    @property
    def pi(self):
        """Pi constant."""
        return 3.141592653589793

    # ── Array creation (like) ──

    def zeros_like(self, x, dtype=None):
        """Create zeros array with same shape as x."""
        import cupy as cp
        result = cp.zeros_like(x)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def ones_like(self, x, dtype=None):
        """Create ones array with same shape as x."""
        import cupy as cp
        result = cp.ones_like(x)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def full_like(self, x, fill_value, dtype=None):
        """Create filled array with same shape as x."""
        import cupy as cp
        result = cp.full_like(x, fill_value)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    # ── Element-wise math ──

    def isnan(self, x):
        """Element-wise NaN check."""
        import cupy as cp
        return cp.isnan(x)

    def isinf(self, x):
        """Element-wise Inf check."""
        import cupy as cp
        return cp.isinf(x)

    def nan_to_num(self, x, nan=0.0, posinf=None, neginf=None):
        """Replace NaN and Inf values."""
        import cupy as cp
        return cp.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    def square(self, x):
        """Element-wise square."""
        import cupy as cp
        return cp.square(x)

    def log1p(self, x):
        """Element-wise log(1 + x)."""
        import cupy as cp
        return cp.log1p(x)

    def sign(self, x):
        """Element-wise sign."""
        import cupy as cp
        return cp.sign(x)

    # ── Reduction / logic ──

    def count_nonzero(self, x):
        """Count non-zero elements."""
        import cupy as cp
        return cp.count_nonzero(x)

    def any(self, x, axis=None):
        """Check if any element is true."""
        import cupy as cp
        return cp.any(x, axis=axis)

    def all(self, x, axis=None):
        """Check if all elements are true."""
        import cupy as cp
        return cp.all(x, axis=axis)

    def unique(self, x, return_counts=False):
        """Return unique elements."""
        import cupy as cp
        if return_counts:
            return cp.unique(x, return_counts=True)
        return cp.unique(x)

    def sort(self, x, axis=-1):
        """Sort array along axis."""
        import cupy as cp
        return cp.sort(x, axis=axis)

    # ── Array manipulation ──

    def reshape(self, x, shape):
        """Reshape array."""
        import cupy as cp
        return cp.reshape(x, shape)

    def flatten(self, x):
        """Flatten array."""
        return x.flatten()

    def squeeze(self, x, axis=None):
        """Remove singleton dimensions."""
        import cupy as cp
        return cp.squeeze(x, axis=axis)

    def astype(self, x, dtype):
        """Cast array to dtype."""
        return x.astype(dtype)

    def cat(self, arrays, axis=0):
        """Concatenate arrays along an axis."""
        import cupy as cp
        return cp.concatenate(arrays, axis=axis)

    def concatenate(self, arrays, axis=0):
        """Concatenate arrays along an axis."""
        import cupy as cp
        return cp.concatenate(arrays, axis=axis)

    def einsum(self, equation, *operands):
        """Einstein summation."""
        import cupy as cp
        return cp.einsum(equation, *operands)

    def tensordot(self, a, b, axes=2):
        """Tensor dot product."""
        import cupy as cp
        return cp.tensordot(a, b, axes=axes)

    def meshgrid(self, *arrays, indexing='xy'):
        """Create coordinate matrices from coordinate vectors."""
        import cupy as cp
        return cp.meshgrid(*arrays, indexing=indexing)

    def item(self, x):
        """Extract scalar value from single-element array."""
        return x.item()

    # ── Memory management ──

    def empty_cache(self):
        """Free GPU memory pool."""
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# ── Raw CUDA kernels for cumulative scan ──
_cumop_1d_template = r'''
extern "C" __global__
void {name}(const {dtype}* __restrict__ x,
            {dtype}* __restrict__ out, int n) {{
    {dtype} cur = x[0];
    out[0] = cur;
    for (int j = 1; j < n; j++) {{
        if ({cmp}) cur = x[j];
        out[j] = cur;
    }}
}}
'''

_cumop_2d_template = r'''
extern "C" __global__
void {name}(const {dtype}* __restrict__ x,
            {dtype}* __restrict__ out, int N, int K) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const {dtype}* row = x + tid * K;
    {dtype}* orow = out + tid * K;
    {dtype} cur = row[0];
    orow[0] = cur;
    for (int j = 1; j < K; j++) {{
        if ({cmp}) cur = row[j];
        orow[j] = cur;
    }}
}}
'''
_CUPY_CUMOP_DTYPES = {
    "float64": "double",
    "float32": "float",
    "int64": "long long",
    "int32": "int",
}
_cumop_kernels = {}


def _get_cumop_kernels(dtype):
    dtype = str(dtype)
    if dtype not in _CUPY_CUMOP_DTYPES:
        raise TypeError(f"Unsupported dtype for CuPy cumop kernels: {dtype}")
    if dtype in _cumop_kernels:
        return _cumop_kernels[dtype]
    import cupy as cp
    ctype = _CUPY_CUMOP_DTYPES[dtype]

    kmin1_mod = cp.RawModule(code=_cumop_1d_template.format(name="cummin_1d", dtype=ctype, cmp="x[j] < cur"))
    kmax1_mod = cp.RawModule(code=_cumop_1d_template.format(name="cummax_1d", dtype=ctype, cmp="x[j] > cur"))
    kmin2_mod = cp.RawModule(code=_cumop_2d_template.format(name="cummin_2d", dtype=ctype, cmp="row[j] < cur"))
    kmax2_mod = cp.RawModule(code=_cumop_2d_template.format(name="cummax_2d", dtype=ctype, cmp="row[j] > cur"))

    kernels = (
        kmin1_mod.get_function('cummin_1d'),
        kmax1_mod.get_function('cummax_1d'),
        kmin2_mod.get_function('cummin_2d'),
        kmax2_mod.get_function('cummax_2d'),
    )
    _cumop_kernels[dtype] = kernels
    return kernels


def _cumop_kernels_available(dtype=None):
    """Check if CuPy cumop kernels can be compiled (lazy, caches on first call)."""
    try:
        _get_cumop_kernels(dtype or "float64")
        return True
    except Exception:
        return False


def _launch_cumop_1d(arr, result, n, is_min):
    if arr is None or result is None:
        raise RuntimeError(
            "CuPy cumop kernels failed to compile or unavailable. "
            "Cannot run cummin/cummax on this device."
        )
    kmin1, kmax1, _, _ = _get_cumop_kernels(arr.dtype)
    kernel = kmin1 if is_min else kmax1
    kernel((1,), (1,), (arr, result, n))


def _launch_cumop_2d(arr, result, N, K, is_min):
    if arr is None or result is None:
        raise RuntimeError(
            "CuPy cumop kernels failed to compile or unavailable. "
            "Cannot run cummin/cummax on this device."
        )
    _, _, kmin2, kmax2 = _get_cumop_kernels(arr.dtype)
    kernel = kmin2 if is_min else kmax2
    block = min(N, 256)
    grid = (N + block - 1) // block
    kernel((grid,), (block,), (arr, result, N, K))
