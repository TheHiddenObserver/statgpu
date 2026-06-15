"""
NumPy / SciPy CPU backend.
"""

import numpy as np

from statgpu.backends._base import BackendBase


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

    def transpose(self, x, axes=None):
        """Transpose array."""
        return np.transpose(x, axes)

    def clip(self, x, min_val, max_val):
        """Clip values to [min_val, max_val]."""
        return np.clip(x, min_val, max_val)

    # ------------------------------------------------------------------
    # Additional methods matching TorchBackend API
    # ------------------------------------------------------------------

    def matmul(self, a, b):
        """Matrix multiplication."""
        return np.matmul(a, b)

    def svd(self, a, full_matrices=True):
        """Singular value decomposition."""
        return np.linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a):
        """Eigenvalue decomposition for symmetric/Hermitian matrices."""
        return np.linalg.eigh(a)

    def qr(self, a, mode='reduced'):
        """QR decomposition."""
        return np.linalg.qr(a, mode=mode)

    def solve(self, a, b):
        """Solve linear system Ax = b."""
        return np.linalg.solve(a, b)

    def solve_triangular(self, a, b, lower=True):
        """Solve triangular system."""
        from scipy.linalg import solve_triangular
        return solve_triangular(a, b, lower=lower)

    def expand_dims(self, x, axis):
        """Expand array dimensions."""
        return np.expand_dims(x, axis=axis)

    def eye(self, n, dtype=None):
        """Create identity matrix."""
        return np.eye(n, dtype=dtype)

    def argmin(self, x, axis=None):
        """Return indices of minimum values."""
        return np.argmin(x, axis=axis)

    def argmax(self, x, axis=None):
        """Return indices of maximum values."""
        return np.argmax(x, axis=axis)

    def argsort(self, x, axis=-1):
        """Return indices that would sort the array."""
        return np.argsort(x, axis=axis)

    def diag(self, x):
        """Extract diagonal or construct diagonal matrix."""
        return np.diag(x)

    def log(self, x):
        """Element-wise natural logarithm."""
        return np.log(x)

    def log1p(self, x):
        """Element-wise log(1 + x)."""
        return np.log1p(x)

    def exp(self, x):
        """Element-wise exponential."""
        return np.exp(x)

    def square(self, x):
        """Element-wise square."""
        return np.square(x)

    def sign(self, x):
        """Element-wise sign."""
        return np.sign(x)

    def item(self, x):
        """Extract scalar value from single-element array."""
        return x.item()

    def min(self, x, axis=None, keepdims=False):
        """Minimum value along axis."""
        return np.min(x, axis=axis, keepdims=keepdims)

    def norm(self, x, ord=None, axis=None):
        """Compute matrix or vector norm."""
        return np.linalg.norm(x, ord=ord, axis=axis)

    def ones(self, shape, dtype=None):
        """Create array of ones."""
        return np.ones(shape, dtype=dtype)

    def ones_like(self, x):
        """Create array of ones with same shape."""
        return np.ones_like(x)

    def zeros_like(self, x):
        """Create array of zeros with same shape."""
        return np.zeros_like(x)

    def full_like(self, x, fill_value):
        """Create array filled with value, same shape."""
        return np.full_like(x, fill_value)

    def where(self, condition, x, y):
        """Element-wise conditional."""
        return np.where(condition, x, y)

    def any(self, x, axis=None):
        """Test whether any element is True."""
        return np.any(x, axis=axis)

    def all(self, x, axis=None):
        """Test whether all elements are True."""
        return np.all(x, axis=axis)

    def unique(self, x):
        """Find unique elements."""
        return np.unique(x)

    def sort(self, x, axis=-1):
        """Sort array along axis."""
        return np.sort(x, axis=axis)

    def isnan(self, x):
        """Element-wise NaN check."""
        return np.isnan(x)

    def isinf(self, x):
        """Element-wise Inf check."""
        return np.isinf(x)

    def nan_to_num(self, x):
        """Replace NaN/Inf with numbers."""
        return np.nan_to_num(x)

    def count_nonzero(self, x):
        """Count non-zero elements."""
        return np.count_nonzero(x)

    def einsum(self, subscripts, *operands):
        """Einstein summation."""
        return np.einsum(subscripts, *operands)

    def tensordot(self, a, b, axes=2):
        """Tensor dot product."""
        return np.tensordot(a, b, axes=axes)

    def meshgrid(self, *xi):
        """Create N-D coordinate arrays."""
        return np.meshgrid(*xi)

    def squeeze(self, x, axis=None):
        """Remove single-dimensional entries."""
        return np.squeeze(x, axis=axis)

    def flatten(self, x):
        """Return flattened array."""
        return x.flatten()

    def empty_cache(self):
        """No-op for CPU backend."""
        pass

    @property
    def inf(self):
        """Infinity value."""
        return np.inf

    @property
    def nan(self):
        """NaN value."""
        return np.nan

    @property
    def pi(self):
        """Pi constant."""
        return np.pi

    @property
    def bool(self):
        """Boolean dtype."""
        return np.bool_
