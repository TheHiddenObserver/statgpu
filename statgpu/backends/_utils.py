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
        return x.cpu().numpy()
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
    except Exception:
        return "cpu"
