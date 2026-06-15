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

from ._base import BackendBase, _is_cupy_array, _is_torch_array, _resolve_backend
from ._numpy import NumpyBackend
from ._cupy import CuPyBackend
from ._torch import TorchBackend
from ._factory import get_backend
from ._utils import (
    _get_xp,
    _to_numpy,
    _to_float_scalar,
    _get_torch_device_str,
    _cupy_to_torch_dlpack,
    _torch_to_cupy_dlpack,
    _numpy_to_torch_tensor,
    _move_torch_tensor,
    _torch_dev,
    _LINALG_ERRORS,
    xp_zeros,
    xp_eye,
    xp_full,
    xp_astype,
    xp_asarray,
    xp_empty,
    xp_arange,
    xp_ones,
    xp_maximum,
    xp_copy,
    xp_cholesky_solve,
)

__all__ = [
    "BackendBase",
    "NumpyBackend",
    "CuPyBackend",
    "TorchBackend",
    "get_backend",
    "_is_cupy_array",
    "_is_torch_array",
    "_resolve_backend",
    "_get_xp",
    "_to_numpy",
    "_to_float_scalar",
    "_get_torch_device_str",
    "_cupy_to_torch_dlpack",
    "_torch_to_cupy_dlpack",
    "_numpy_to_torch_tensor",
    "_move_torch_tensor",
    "_torch_dev",
    "xp_zeros",
    "xp_eye",
    "xp_full",
    "xp_astype",
    "xp_asarray",
    "xp_empty",
    "xp_arange",
    "xp_ones",
    "xp_maximum",
    "xp_copy",
    "xp_cholesky_solve",
]
