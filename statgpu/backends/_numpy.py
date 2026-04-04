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
