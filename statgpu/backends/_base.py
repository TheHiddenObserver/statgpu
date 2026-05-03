"""
Abstract base class for compute backends.

A backend wraps an array library (NumPy, CuPy, or PyTorch) and exposes a
uniform interface so that model implementations can stay array-library agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Array-type detection helpers (deferred imports to avoid hard deps)
# ---------------------------------------------------------------------------

def _is_cupy_array(x: Any) -> bool:
    """Return True if *x* is a CuPy ndarray."""
    try:
        import cupy as cp
        return isinstance(x, cp.ndarray)
    except Exception:
        return False


def _is_torch_array(x: Any) -> bool:
    """Return True if *x* is a PyTorch Tensor."""
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _resolve_backend(backend: str, *arrays) -> str:
    """Resolve the named *backend* string to one of ``'numpy'``, ``'cupy'``,
    ``'torch'``.

    When *backend* is ``'auto'``, inspect *arrays* and return the
    matching backend name based on the first recognised array type.
    Falls back to ``'numpy'`` when no array matches.
    """
    backend_name = str(backend).strip().lower()
    if backend_name not in ("auto", "numpy", "cupy", "torch"):
        raise ValueError("backend must be one of: 'auto', 'numpy', 'cupy', 'torch'")
    if backend_name != "auto":
        return backend_name

    for arr in arrays:
        if arr is not None:
            if _is_torch_array(arr):
                return "torch"
            if _is_cupy_array(arr):
                return "cupy"
    return "numpy"


class BackendBase(ABC):
    """
    Abstract base for compute backends.

    Subclasses wrap a specific array library and expose:

    * ``xp``        – the underlying array module (numpy / cupy / torch).
    * ``asarray``   – convert arbitrary inputs to the backend's native array.
    * ``to_numpy``  – convert the backend's arrays back to ``numpy.ndarray``.
    * ``is_available`` – runtime check for the library being usable.

    The ``xp`` object follows the NumPy array API so that operations such as
    ``xp.linalg.solve``, ``xp.sum``, ``xp.exp`` etc. work without
    library-specific branches in the calling code.
    """

    #: Short name used in repr and config ('numpy', 'cupy', 'torch').
    name: str = ""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Return the array module (numpy / cupy / torch)."""

    @abstractmethod
    def asarray(self, x, dtype=None) -> Any:
        """
        Convert *x* to this backend's native array type.

        Parameters
        ----------
        x : array-like, numpy.ndarray, cupy.ndarray, or torch.Tensor
            Input data.
        dtype : dtype-like, optional
            Desired data type.

        Returns
        -------
        array
            Native array on the backend's device.
        """

    @abstractmethod
    def to_numpy(self, x) -> np.ndarray:
        """
        Convert *x* to a ``numpy.ndarray``.

        Parameters
        ----------
        x : array-like
            A native array produced by this backend (or any array-like).

        Returns
        -------
        numpy.ndarray
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used in the current environment."""

    # ------------------------------------------------------------------
    # Convenience helpers (non-abstract, built on top of xp)
    # ------------------------------------------------------------------

    def solve(self, A, b):
        """Solve the linear system *Ax = b*."""
        return self.xp.linalg.solve(A, b)

    def lstsq(self, A, b, rcond=None):
        """Return the least-squares solution to *Ax ≈ b*."""
        return self.xp.linalg.lstsq(A, b, rcond=rcond)

    def astype(self, arr, dtype):
        """Cast *arr* to *dtype* (backend-agnostic .astype / .to)."""
        return arr.astype(dtype)

    def concatenate(self, arrays, axis=0):
        """Concatenate *arrays* along *axis* (.concatenate / .cat)."""
        return self.xp.concatenate(arrays, axis=axis)

    def matmul(self, a, b):
        """Matrix multiplication."""
        return self.xp.matmul(a, b)

    def svd(self, a, full_matrices=True):
        """Singular value decomposition."""
        return self.xp.linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a):
        """Eigenvalue decomposition for symmetric/Hermitian matrices."""
        return self.xp.linalg.eigh(a)

    def qr(self, a):
        """QR decomposition."""
        return self.xp.linalg.qr(a)

    def diag(self, x, k=0):
        """Extract diagonal or create diagonal matrix."""
        return self.xp.diag(x, k=k)

    def argmin(self, x, axis=None):
        """Return indices of minimum values."""
        return self.xp.argmin(x, axis=axis)

    def argmax(self, x, axis=None):
        """Return indices of maximum values."""
        return self.xp.argmax(x, axis=axis)

    def min(self, x, axis=None, keepdims=False):
        """Minimum value along an axis."""
        return self.xp.min(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        """Maximum value along an axis."""
        return self.xp.max(x, axis=axis, keepdims=keepdims)

    def exp(self, x):
        """Element-wise exponential."""
        return self.xp.exp(x)

    def log(self, x):
        """Element-wise natural logarithm."""
        return self.xp.log(x)

    def where(self, cond, x, y):
        """Element-wise selection based on condition."""
        return self.xp.where(cond, x, y)

    def bincount(self, x, weights=None, minlength=0):
        """Count occurrences of non-negative integer labels."""
        return self.xp.bincount(x, weights=weights, minlength=minlength)

    def logsumexp(self, x, axis=None, keepdims=False):
        """Stable log(sum(exp(x))) reduction."""
        max_x = self.max(x, axis=axis, keepdims=True)
        shifted = x - max_x
        out = self.log(self.sum(self.exp(shifted), axis=axis, keepdims=True)) + max_x
        if keepdims:
            return out
        if axis is None:
            return self.reshape(out, ())
        return self.xp.squeeze(out, axis=axis)

    def norm(self, x, axis=None, keepdims=False):
        """Euclidean norm."""
        return self.xp.sqrt(self.sum(x * x, axis=axis, keepdims=keepdims))

    def item(self, x):
        """Convert a scalar array/tensor to a Python scalar."""
        if hasattr(x, "detach"):
            return x.detach().cpu().item()
        if hasattr(x, "get"):
            return x.get().item()
        if hasattr(x, "item"):
            return x.item()
        return x

    def zeros_like(self, x, dtype=None):
        """Create zeros with the same shape as x."""
        out = self.xp.zeros_like(x)
        return out.astype(dtype) if dtype is not None else out

    def ones_like(self, x, dtype=None):
        """Create ones with the same shape as x."""
        out = self.xp.ones_like(x)
        return out.astype(dtype) if dtype is not None else out

    def argsort(self, x, axis=-1):
        """Return indices that would sort an array."""
        return self.xp.argsort(x, axis=axis)

    def sort(self, x, axis=-1):
        """Sort an array."""
        return self.xp.sort(x, axis=axis)

    def reshape(self, x, shape):
        """Reshape an array."""
        return self.xp.reshape(x, shape)

    def expand_dims(self, x, axis):
        """Add a singleton dimension."""
        return self.xp.expand_dims(x, axis)

    def copy(self, x):
        """Return a copy of an array."""
        return x.copy()

    def take_along_axis(self, arr, indices, axis):
        """Gather elements along *axis* (.take_along_axis / .take_along_dim)."""
        if self.name == "torch":
            return self.xp.take_along_dim(arr, indices, dim=axis)
        return self.xp.take_along_axis(arr, indices, axis=axis)

    def cummin(self, arr, axis=0):
        """Cumulative minimum along *axis*."""
        return self.xp.minimum.accumulate(arr, axis=axis)

    def cummax(self, arr, axis=0):
        """Cumulative maximum along *axis*."""
        return self.xp.maximum.accumulate(arr, axis=axis)

    def flip(self, arr, axis=0):
        """Reverse the order of elements along *axis*."""
        if self.name == "torch":
            return self.xp.flip(arr, dims=(axis,))
        return self.xp.flip(arr, axis=axis)

    def __repr__(self) -> str:
        available = "available" if self.is_available() else "unavailable"
        return f"{self.__class__.__name__}(name={self.name!r}, {available})"
