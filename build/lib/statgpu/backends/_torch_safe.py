"""Safe torch import wrapper for Torch 2.8+ compatibility.

Torch 2.8.0+ may raise RuntimeError('Only a single TORCH_LIBRARY can be
used to register the namespace prims') when imported in environments where
torch has already been loaded (e.g., Jupyter kernels, other processes).

This module provides a safe import that catches this error and marks torch
as unavailable. All torch imports in statgpu should go through this module
via: from statgpu.backends._torch_safe import get_torch
"""

_torch = None
_torch_available = None  # None = not checked, True/False = checked


def get_torch():
    """Return the torch module, or None if not available.

    Catches RuntimeError from TORCH_LIBRARY registration conflicts
    that occur on Torch 2.8+ in environments with pre-existing torch state.
    """
    global _torch, _torch_available

    if _torch_available is True:
        return _torch
    if _torch_available is False:
        return None

    try:
        import torch
        _torch = torch
        _torch_available = True
        return _torch
    except (ImportError, RuntimeError) as e:
        # RuntimeError: TORCH_LIBRARY conflict on Torch 2.8+
        # ImportError: torch not installed
        _torch = None
        _torch_available = False
        return None


def torch_available():
    """Check if torch is available without importing it."""
    global _torch_available
    if _torch_available is None:
        get_torch()
    return _torch_available
