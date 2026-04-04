"""
statgpu.backends – pluggable compute backends for array operations.

Supported backends
------------------
* **NumpyBackend** – CPU, always available.
* **CuPyBackend**  – CUDA GPU via CuPy (install ``statgpu[gpu11]`` or
  ``statgpu[gpu12]``).
* **TorchBackend** – CUDA GPU (or CPU) via PyTorch (install
  ``statgpu[torch]``).

Quick start
-----------
>>> from statgpu.backends import get_backend
>>> backend = get_backend()        # auto-detects best available backend
>>> xp = backend.xp                # array module (numpy / cupy / torch)
>>> arr = backend.asarray([1, 2, 3])
>>> backend.to_numpy(arr)
array([1, 2, 3])

Use ``get_backend(backend='cupy')`` or ``get_backend(backend='torch')`` to
force a specific library.
"""

from ._base import BackendBase
from ._numpy import NumpyBackend
from ._cupy import CuPyBackend
from ._torch import TorchBackend
from ._factory import get_backend

__all__ = [
    "BackendBase",
    "NumpyBackend",
    "CuPyBackend",
    "TorchBackend",
    "get_backend",
]
