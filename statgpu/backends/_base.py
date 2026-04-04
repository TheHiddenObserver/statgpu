"""
Abstract base class for compute backends.

A backend wraps an array library (NumPy, CuPy, or PyTorch) and exposes a
uniform interface so that model implementations can stay array-library agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


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

    def __repr__(self) -> str:
        available = "available" if self.is_available() else "unavailable"
        return f"{self.__class__.__name__}(name={self.name!r}, {available})"
