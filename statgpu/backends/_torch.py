"""
PyTorch GPU/CPU backend.

PyTorch tensors do *not* mirror the NumPy array API 1:1 (e.g. ``torch.linalg``
vs ``numpy.linalg``, different dtypes, etc.).  The ``xp`` property therefore
returns the ``torch`` module itself; callers that need NumPy-compatible ops
should use the helper methods on this class instead of ``xp.<op>`` directly.

Note on API compatibility
-------------------------
Model code should use ``backend.xp`` for basic operations like:
- ``backend.xp.sum``, ``backend.xp.matmul``, ``backend.xp.sqrt``, etc.
- ``backend.xp.linalg.solve``, ``backend.xp.linalg.cholesky``, etc.

For operations with API differences (e.g. ``axis`` vs ``dim``), use helper
methods on this backend class.
"""

import numpy as np

from statgpu.backends._base import BackendBase
from statgpu.backends._utils import (
    _cupy_to_torch_dlpack,
    _move_torch_tensor,
    _numpy_to_torch_tensor,
)

# Default CUDA device string used when moving tensors to GPU.
_DEFAULT_TORCH_DEVICE = "cuda"


class TorchBackend(BackendBase):
    """
    GPU (or CPU) backend powered by PyTorch.

    Requires ``torch`` (install via ``pip install statgpu[torch]``).

    Parameters
    ----------
    device : str, default='cuda'
        Torch device string, e.g. ``'cuda'``, ``'cuda:0'``, or ``'cpu'``.

    Examples
    --------
    >>> from statgpu.backends import TorchBackend
    >>> backend = TorchBackend(device='cuda')
    >>> xp = backend.xp  # torch module
    >>> arr = backend.asarray([1, 2, 3])
    >>> backend.to_numpy(arr)
    array([1, 2, 3])
    """

    name = "torch"

    def __init__(self, device: str = _DEFAULT_TORCH_DEVICE):
        self._device = device
        self._initialized = False

    def _ensure_initialized(self):
        """Perform one-time CUDA warmup to avoid lazy kernel init penalty."""
        if self._initialized:
            return
        if self._device != 'cpu':
            import torch
            if torch.cuda.is_available():
                # Warmup: small matmul to trigger CUDA kernel initialization
                _ = torch.randn(32, 32, device=self._device) @ torch.randn(32, 32, device=self._device)
                torch.cuda.synchronize()
        self._initialized = True

    @property
    def xp(self):
        import torch  # deferred import
        return torch

    def asarray(self, x, dtype=None):
        """
        Convert x to Torch tensor on the configured device.

        Parameters
        ----------
        x : array-like
            Input data (list, numpy.ndarray, cupy.ndarray, or torch.Tensor).
        dtype : torch.dtype, optional
            Desired data type (e.g., torch.float64).

        Returns
        -------
        torch.Tensor
        """
        import torch
        self._ensure_initialized()  # Warmup on first use to avoid lazy CUDA init
        if isinstance(x, torch.Tensor):
            t = _move_torch_tensor(x, device=self._device, dtype=dtype)
        elif hasattr(x, "get"):
            # CuPy arrays expose a .get() method that transfers the array from
            # GPU memory to a NumPy ndarray on the host.  Duck-typing avoids a
            # mandatory cupy import here.
            t = _cupy_to_torch_dlpack(x, device=self._device)
            if t is None:
                t = _numpy_to_torch_tensor(
                    x.get(),
                    device=self._device,
                    dtype=dtype,
                    pin_memory=self._device.startswith("cuda"),
                )
            elif dtype is not None:
                t = _move_torch_tensor(t, dtype=dtype)
        else:
            # Use torch.from_numpy for numpy arrays, then ensure contiguous memory
            t = _numpy_to_torch_tensor(
                x,
                device=self._device,
                dtype=dtype,
                pin_memory=self._device.startswith("cuda"),
            )
        # Ensure result is contiguous for optimal performance
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def to_numpy(self, x) -> np.ndarray:
        """
        Convert Torch tensor to NumPy array.

        Parameters
        ----------
        x : torch.Tensor or array-like
            A native tensor produced by this backend (or any array-like).

        Returns
        -------
        numpy.ndarray
        """
        import torch
        if isinstance(x, torch.Tensor):
            # Move to CPU first, then convert to numpy
            return x.detach().cpu().numpy()
        if hasattr(x, "get"):
            # CuPy arrays expose a .get() method that transfers the array from
            # GPU memory to a NumPy ndarray on the host.
            return x.get()
        return np.asarray(x)

    def is_available(self) -> bool:
        """Return True if PyTorch can be used in the current environment."""
        try:
            import torch
            # Allow CPU-based torch backend as well.
            if self._device.startswith("cuda"):
                return torch.cuda.is_available()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Override helpers to use torch.linalg
    # ------------------------------------------------------------------

    def solve(self, A, b):
        """Solve the linear system Ax = b using torch.linalg.solve."""
        import torch
        return torch.linalg.solve(A, b)

    def lstsq(self, A, b, rcond=None):
        """
        Return the least-squares solution to Ax ≈ b.

        torch.linalg.lstsq returns a named tuple; we unpack it for
        compatibility with numpy's lstsq interface.

        Returns
        -------
        solution : torch.Tensor
        residuals : torch.Tensor
        rank : int
        singular_values : torch.Tensor
        """
        import torch
        result = torch.linalg.lstsq(A, b)
        return result.solution, result.residuals, result.rank, result.singular_values

    def solve_triangular(self, A, b, lower=False, trans=False, unit_triangular=False):
        """
        Solve the triangular system Ax = b.

        Parameters
        ----------
        A : torch.Tensor
            Triangular matrix (n, n).
        b : torch.Tensor
            Right-hand side (n,) or (n, k).
        lower : bool, default=False
            Whether to use the lower triangle of A.
        trans : bool, default=False
            Whether to transpose A.
        unit_triangular : bool, default=False
            Whether to assume the diagonal of A is all ones.

        Returns
        -------
        x : torch.Tensor
            Solution to the system.
        """
        import torch
        if isinstance(trans, str):
            trans_flag = trans.upper() in ("T", "C")
        else:
            trans_flag = bool(trans)
        if trans_flag:
            A = A.transpose(-2, -1)
            lower = not lower
        return torch.linalg.solve_triangular(
            A,
            b,
            upper=not lower,
            unitriangular=bool(unit_triangular),
        )

    # ------------------------------------------------------------------
    # Additional Torch-native helpers for common operations
    # ------------------------------------------------------------------

    def sum(self, x, axis=None, keepdims=False):
        """
        Sum over specified axis/axes.

        Note: Torch uses 'dim' instead of 'axis'.
        """
        import torch
        if axis is None:
            return torch.sum(x)
        if isinstance(axis, int):
            return torch.sum(x, dim=axis, keepdim=keepdims)
        # Multiple axes: sum iteratively
        for ax in sorted(axis, reverse=True):
            x = torch.sum(x, dim=ax, keepdim=keepdims)
        return x

    def mean(self, x, axis=None, keepdims=False):
        """Mean over specified axis/axes."""
        import torch
        if axis is None:
            return torch.mean(x)
        if isinstance(axis, int):
            return torch.mean(x, dim=axis, keepdim=keepdims)
        # For multiple axes, compute manually
        if isinstance(axis, (list, tuple)):
            n_elem = 1
            for ax in axis:
                n_elem *= x.shape[ax]
            return self.sum(x, axis=axis, keepdims=keepdims) / n_elem
        return torch.mean(x, dim=axis, keepdim=keepdims)

    def sqrt(self, x):
        """Element-wise square root."""
        import torch
        return torch.sqrt(x)

    def abs(self, x):
        """Element-wise absolute value."""
        import torch
        return torch.abs(x)

    def max(self, x, axis=None, keepdims=False):
        """Maximum value along axis."""
        import torch
        if axis is None:
            return torch.max(x)
        if isinstance(axis, int):
            result = torch.max(x, dim=axis, keepdim=keepdims)
            return result.values if hasattr(result, 'values') else result[0]
        # Multiple axes: reduce iteratively
        for ax in sorted(axis, reverse=True):
            result = torch.max(x, dim=ax, keepdim=keepdims)
            x = result.values if hasattr(result, 'values') else result[0]
        return x

    def square(self, x):
        """Element-wise square."""
        import torch
        return torch.square(x)

    def exp(self, x):
        """Element-wise exponential."""
        import torch
        return torch.exp(x)

    def log(self, x):
        """Element-wise natural logarithm."""
        import torch
        return torch.log(x)

    def log1p(self, x):
        """Element-wise log(1 + x)."""
        import torch
        return torch.log1p(x)

    def maximum(self, x, y):
        """Element-wise maximum of two arrays."""
        import torch
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)

    def minimum(self, x, y):
        """Element-wise minimum of two arrays."""
        import torch
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)
        return torch.minimum(x, y)

    def clip(self, x, min_val, max_val):
        """Clip values to [min_val, max_val]."""
        import torch
        return torch.clamp(x, min_val, max_val)

    def where(self, cond, x, y):
        """Element-wise selection based on condition."""
        import torch
        return torch.where(cond, x, y)

    def stack(self, arrays, axis=0):
        """Stack arrays along a new axis."""
        import torch
        return torch.stack(arrays, dim=axis)

    def cat(self, arrays, axis=0):
        """Concatenate arrays along an axis."""
        import torch
        return torch.cat(arrays, dim=axis)

    def diag(self, x, k=0):
        """Extract diagonal or create diagonal matrix."""
        import torch
        return torch.diag(x, diagonal=k)

    def einsum(self, equation, *operands):
        """Einstein summation."""
        import torch
        return torch.einsum(equation, *operands)

    def transpose(self, x, axes=None):
        """Transpose array."""
        import torch
        if axes is None:
            return x.T
        return x.permute(axes)

    def arange(self, start, stop=None, step=1, dtype=None):
        """Create range array."""
        import torch
        if stop is None:
            result = torch.arange(0, start, step, device=self._device)
        else:
            result = torch.arange(start, stop, step, device=self._device)
        if dtype is not None:
            result = result.to(dtype)
        return result

    def zeros(self, shape, dtype=None):
        """Create array of zeros."""
        import torch
        return torch.zeros(shape, device=self._device, dtype=dtype if dtype is not None else torch.float64)

    def ones(self, shape, dtype=None):
        """Create array of ones."""
        import torch
        return torch.ones(shape, device=self._device, dtype=dtype if dtype is not None else torch.float64)

    def eye(self, n, m=None, dtype=None):
        """Create identity matrix."""
        import torch
        if m is None:
            m = n
        return torch.eye(n, m, device=self._device, dtype=dtype if dtype is not None else torch.float64)

    def full(self, shape, fill_value, dtype=None):
        """Create array filled with a constant value."""
        import torch
        if isinstance(shape, int):
            shape = (shape,)
        return torch.full(shape, fill_value, device=self._device, dtype=dtype if dtype is not None else torch.float64)

    def array(self, val, dtype=None):
        """Create a scalar or array from a value."""
        import torch
        return torch.tensor(val, device=self._device, dtype=dtype if dtype is not None else torch.float64)

    def isnan(self, x):
        """Element-wise isnan check."""
        import torch
        return torch.isnan(x)

    def isinf(self, x):
        """Element-wise isinf check."""
        import torch
        return torch.isinf(x)

    def nan_to_num(self, x, nan=0.0, posinf=None, neginf=None):
        """Replace NaN and Inf values."""
        import torch
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    def matmul(self, a, b):
        """Matrix multiplication."""
        import torch
        return torch.matmul(a, b)

    def min(self, x, axis=None, keepdims=False):
        """Minimum value along axis."""
        import torch
        if axis is None:
            return torch.min(x)
        result = torch.min(x, dim=axis)
        if keepdims:
            return result.values.unsqueeze(axis)
        return result.values

    def expand_dims(self, x, axis):
        """Expand array dimensions."""
        import torch
        return torch.unsqueeze(x, axis)

    def eigh(self, a):
        """Eigenvalue decomposition for symmetric/Hermitian matrices."""
        import torch
        return torch.linalg.eigh(a)

    def qr(self, a, mode='reduced'):
        """QR decomposition."""
        import torch
        Q, R = torch.linalg.qr(a)
        if mode == 'r':
            return R
        return Q, R

    def svd(self, a, full_matrices=True):
        """Singular value decomposition."""
        import torch
        return torch.linalg.svd(a, full_matrices=full_matrices)

    def solve(self, A, b):
        """Solve the linear system Ax = b."""
        import torch
        return torch.linalg.solve(A, b)

    def argmin(self, x, axis=None):
        """Indices of minimum values along axis."""
        import torch
        return torch.argmin(x, dim=axis)

    def argmax(self, x, axis=None):
        """Indices of maximum values along axis."""
        import torch
        return torch.argmax(x, dim=axis)

    def argsort(self, x, axis=-1):
        """Indices that would sort the array."""
        import torch
        return torch.argsort(x, dim=axis)

    def flip(self, x, axis=None):
        """Reverse array order along axis."""
        import torch
        if axis is None:
            return torch.flip(x, list(range(x.ndim)))
        if isinstance(axis, int):
            axis = [axis]
        return torch.flip(x, axis)

    def logsumexp(self, arr, axis=None):
        """Log-sum-exp along axis (torch-compatible)."""
        import torch
        if axis is None:
            m = torch.max(arr)
        else:
            m = torch.max(arr, dim=axis, keepdim=True).values
        # squeeze m to match arr shape after reduction
        if axis is not None:
            m_squeezed = torch.squeeze(m, dim=axis)
        else:
            m_squeezed = m
        return m_squeezed + torch.log(torch.sum(torch.exp(arr - m), dim=axis))

    def tensordot(self, a, b, axes=2):
        """Tensor dot product."""
        import torch
        return torch.tensordot(a, b, dims=axes)

    def outer(self, a, b):
        """Outer product."""
        import torch
        return torch.outer(a.flatten(), b.flatten())

    def newaxis(self):
        """Alias for None, used in indexing."""
        return None

    def meshgrid(self, *arrays, indexing='xy'):
        """Create coordinate matrices from coordinate vectors."""
        import torch
        return torch.meshgrid(*arrays, indexing=indexing)

    def argmax(self, x, axis=None):
        """Return index of maximum value."""
        import torch
        if axis is None:
            return torch.argmax(x)
        return torch.argmax(x, dim=axis)

    def argmin(self, x, axis=None):
        """Return index of minimum value."""
        import torch
        if axis is None:
            return torch.argmin(x)
        return torch.argmin(x, dim=axis)

    def sort(self, x, axis=-1):
        """Sort array along axis."""
        import torch
        return torch.sort(x, dim=axis).values

    def argsort(self, x, axis=-1):
        """Return indices that would sort array."""
        import torch
        return torch.argsort(x, dim=axis)

    def unique(self, x, return_counts=False):
        """Return unique elements."""
        import torch
        if return_counts:
            return torch.unique(x, return_counts=return_counts)
        return torch.unique(x)

    def any(self, x, axis=None):
        """Check if any element is true."""
        import torch
        if axis is None:
            return torch.any(x)
        return torch.any(x, dim=axis)

    def all(self, x, axis=None):
        """Check if all elements are true."""
        import torch
        if axis is None:
            return torch.all(x)
        return torch.all(x, dim=axis)

    def zeros_like(self, x, dtype=None):
        """Create zeros array with same shape as x."""
        import torch
        result = torch.zeros_like(x)
        if dtype is not None:
            result = result.to(dtype)
        return result

    def ones_like(self, x, dtype=None):
        """Create ones array with same shape as x."""
        import torch
        result = torch.ones_like(x)
        if dtype is not None:
            result = result.to(dtype)
        return result

    def full_like(self, x, fill_value, dtype=None):
        """Create filled array with same shape as x."""
        import torch
        result = torch.full_like(x, fill_value)
        if dtype is not None:
            result = result.to(dtype)
        return result

    def copy(self, x):
        """Return a copy of x."""
        import torch
        return x.clone()

    def reshape(self, x, shape):
        """Reshape array."""
        import torch
        return x.reshape(shape)

    def flatten(self, x):
        """Flatten array."""
        import torch
        return x.flatten()

    def squeeze(self, x, axis=None):
        """Remove singleton dimensions."""
        import torch
        if axis is None:
            return x.squeeze()
        return x.squeeze(axis)

    def expand_dims(self, x, axis):
        """Add singleton dimension."""
        import torch
        return x.unsqueeze(axis)

    def atleast_1d(self, x):
        """Ensure array is at least 1D."""
        import torch
        x = torch.as_tensor(x)
        if x.ndim == 0:
            return x.reshape(1)
        return x

    def astype(self, x, dtype):
        """Cast array to dtype."""
        import torch
        return x.to(dtype)

    def concatenate(self, arrays, axis=0):
        """Concatenate *arrays* along *axis* (torch.cat)."""
        import torch
        return torch.cat(arrays, dim=axis)

    def take_along_axis(self, arr, indices, axis):
        """Gather elements along *axis* (torch.take_along_dim)."""
        import torch
        return torch.take_along_dim(arr, indices, dim=axis)

    def cummin(self, arr, axis=0):
        """Cumulative minimum along *axis* (torch.cummin)."""
        import torch
        vals, _ = torch.cummin(arr, dim=axis)
        return vals

    def cummax(self, arr, axis=0):
        """Cumulative maximum along *axis* (torch.cummax)."""
        import torch
        vals, _ = torch.cummax(arr, dim=axis)
        return vals

    def flip(self, arr, axis=0):
        """Reverse the order of elements along *axis* (torch.flip)."""
        import torch
        return torch.flip(arr, dims=[axis])

    @property
    def float64(self):
        """float64 dtype."""
        import torch
        return torch.float64

    @property
    def float32(self):
        """float32 dtype."""
        import torch
        return torch.float32

    @property
    def int64(self):
        """int64 dtype."""
        import torch
        return torch.int64

    @property
    def int32(self):
        """int32 dtype."""
        import torch
        return torch.int32

    @property
    def bool(self):
        """bool dtype."""
        import torch
        return torch.bool

    @property
    def nan(self):
        """NaN value."""
        import torch
        return torch.tensor(float('nan'), dtype=torch.float64, device=self._device)

    @property
    def inf(self):
        """Infinity value."""
        import torch
        return torch.tensor(float('inf'), dtype=torch.float64, device=self._device)

    @property
    def pi(self):
        """Pi constant."""
        import torch
        return torch.tensor(3.141592653589793, dtype=torch.float64, device=self._device)

    def empty_cache(self):
        """Clear GPU cache (Torch-specific)."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def count_nonzero(self, x):
        """Count non-zero elements."""
        import torch
        return torch.count_nonzero(x)

    def sign(self, x):
        """Element-wise sign."""
        import torch
        return torch.sign(x)
