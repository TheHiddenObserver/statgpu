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
