"""General-purpose backend utility functions.

These helpers are used across statgpu submodules to avoid duplicating
array-library detection, module resolution, and scalar conversion logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_xp(backend_name: str):
    """Return the array module (numpy / cupy / torch) for *backend_name*.

    Parameters
    ----------
    backend_name : str
        One of ``'numpy'``, ``'cupy'``, or ``'torch'``.

    Returns
    -------
    module
        The array module (``numpy``, ``cupy``, or ``torch``).

    Raises
    ------
    ValueError
        If *backend_name* is not recognised.
    ImportError
        If the requested library is not installed.
    """
    if backend_name == "numpy":
        return np
    if backend_name == "cupy":
        try:
            import cupy as cp

            return cp
        except ImportError as exc:
            raise ImportError(
                "backend='cupy' requires CuPy, but CuPy is not installed"
            ) from exc
    if backend_name == "torch":
        try:
            import torch

            return torch
        except ImportError as exc:
            raise ImportError(
                "backend='torch' requires PyTorch, but PyTorch is not installed"
            ) from exc
    raise ValueError(f"Unsupported backend: {backend_name}")


def _to_numpy(x):
    """Convert *x* to a ``numpy.ndarray``.

    Handles CuPy arrays (``.get()``) and PyTorch tensors (``.cpu().numpy()``).
    """
    if hasattr(x, "get"):
        return x.get()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else x.cpu().numpy()
    return np.asarray(x)


def _to_float_scalar(x: Any) -> float:
    """Extract a Python ``float`` from a backend array scalar."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _get_torch_device_str() -> str:
    """Return ``'cuda'`` if PyTorch CUDA is available, else ``'cpu'``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
    except Exception as e:
        import warnings
        warnings.warn(f"torch.cuda.is_available() failed, falling back to CPU: {e}")
        return "cpu"


# ---------------------------------------------------------------------------
# Device-aware array creation helpers
# ---------------------------------------------------------------------------

def _torch_dev(arr):
    """Extract device from a torch tensor, or ``None`` for non-torch arrays."""
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.device
    except (ImportError, AttributeError):
        pass
    return None


def xp_zeros(shape, dtype, xp, ref_arr=None):
    """Device-aware ``xp.zeros``.  *ref_arr* provides the target device."""
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        return xp.zeros(shape, dtype=dtype, device=dev)
    return xp.zeros(shape, dtype=dtype)


def xp_eye(n, dtype, xp, ref_arr=None):
    """Device-aware ``xp.eye``.  *ref_arr* provides the target device."""
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        return xp.eye(n, dtype=dtype, device=dev)
    return xp.eye(n, dtype=dtype)


def xp_full(shape, fill_value, dtype, xp, ref_arr=None):
    """Device-aware ``xp.full`` with int→tuple normalisation."""
    if isinstance(shape, int):
        shape = (shape,)
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        return xp.full(shape, fill_value, dtype=dtype, device=dev)
    return xp.full(shape, fill_value, dtype=dtype)


def xp_astype(arr, dtype, xp):
    """Backend-safe type cast (``.to()`` for torch, ``.astype()`` otherwise)."""
    if _torch_dev(arr) is not None:
        return arr.to(dtype)
    return arr.astype(dtype)


def xp_asarray(data, dtype=None, xp=None, ref_arr=None):
    """Device-aware ``xp.asarray``.  *ref_arr* provides the target device."""
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        kwargs = {'device': dev}
        if dtype is not None:
            kwargs['dtype'] = dtype
        return xp.asarray(data, **kwargs)
    if dtype is not None:
        return xp.asarray(data, dtype=dtype)
    return xp.asarray(data)


def xp_empty(shape, dtype, xp, ref_arr=None):
    """Device-aware ``xp.empty``.  *ref_arr* provides the target device."""
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        return xp.empty(shape, dtype=dtype, device=dev)
    return xp.empty(shape, dtype=dtype)


def xp_arange(n, dtype=None, xp=None, ref_arr=None):
    """Device-aware ``xp.arange``.  *ref_arr* provides the target device."""
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        kwargs = {'device': dev}
        if dtype is not None:
            kwargs['dtype'] = dtype
        return xp.arange(n, **kwargs)
    if dtype is not None:
        return xp.arange(n, dtype=dtype)
    return xp.arange(n)


def xp_ones(shape, dtype, xp, ref_arr=None):
    """Device-aware ``xp.ones``.  *ref_arr* provides the target device."""
    dev = _torch_dev(ref_arr) if ref_arr is not None else None
    if dev is not None:
        return xp.ones(shape, dtype=dtype, device=dev)
    return xp.ones(shape, dtype=dtype)


def xp_maximum(arr, value, xp=None):
    """Element-wise maximum that works for both numpy/cupy and torch.

    Torch's ``maximum()`` requires both args to be tensors; numpy/cupy accept
    scalars.  This helper wraps *value* as needed.
    """
    if _torch_dev(arr) is not None:
        import torch
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=arr.dtype, device=arr.device)
        return torch.maximum(arr, value)
    return xp.maximum(arr, value) if xp is not None else np.maximum(arr, value)


def xp_copy(arr):
    """Backend-safe copy (``.clone()`` for torch, ``.copy()`` otherwise)."""
    if _torch_dev(arr) is not None:
        return arr.clone()
    return arr.copy()


def xp_cholesky_solve(A, b, xp):
    """Solve ``A @ x = b`` via Cholesky decomposition.

    Works across numpy, cupy, and torch backends.  Handles the torch-specific
    argument difference for ``solve_triangular`` (``upper=False`` vs ``lower=True``).
    """
    L = xp.linalg.cholesky(A)
    if _torch_dev(L) is not None:
        tmp = xp.linalg.solve_triangular(L, b, upper=False)
        return xp.linalg.solve_triangular(L.T, tmp, upper=True)
    tmp = xp.linalg.solve_triangular(L, b, lower=True)
    return xp.linalg.solve_triangular(L.T, tmp, lower=False)
