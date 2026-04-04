"""
PyTorch GPU/CPU backend.

PyTorch tensors do *not* mirror the NumPy array API 1:1 (e.g. ``torch.linalg``
vs ``numpy.linalg``, different dtypes, etc.).  The ``xp`` property therefore
returns the ``torch`` module itself; callers that need NumPy-compatible ops
should use the helper methods on this class instead of ``xp.<op>`` directly.
"""

import numpy as np

from ._base import BackendBase

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
    """

    name = "torch"

    def __init__(self, device: str = _DEFAULT_TORCH_DEVICE):
        self._device = device

    @property
    def xp(self):
        import torch  # deferred import
        return torch

    def asarray(self, x, dtype=None):
        import torch
        if isinstance(x, torch.Tensor):
            t = x.to(self._device)
        elif hasattr(x, "get"):
            # CuPy arrays expose a .get() method that transfers the array from
            # GPU memory to a NumPy ndarray on the host.  Duck-typing avoids a
            # mandatory cupy import here.
            t = torch.from_numpy(x.get()).to(self._device)
        else:
            t = torch.from_numpy(np.asarray(x)).to(self._device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    def to_numpy(self, x) -> np.ndarray:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if hasattr(x, "get"):
            # CuPy arrays expose a .get() method that transfers the array from
            # GPU memory to a NumPy ndarray on the host.
            return x.get()
        return np.asarray(x)

    def is_available(self) -> bool:
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
        import torch
        return torch.linalg.solve(A, b)

    def lstsq(self, A, b, rcond=None):
        import torch
        # torch.linalg.lstsq returns a named tuple; return (solution, ...) for
        # compatibility with numpy's lstsq interface.
        result = torch.linalg.lstsq(A, b)
        return result.solution, result.residuals, result.rank, result.singular_values
