"""
Backend utilities for GLM loss functions.

Provides wrapper functions that dispatch to numpy/cupy/torch
based on the input array type, so GLM loss functions can use
a single code path for all backends.
"""


def _xp(arr):
    """Get the array module (numpy/cupy/torch) from array type."""
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        import cupy
        return cupy
    if mod.startswith("torch"):
        import torch
        return torch
    import numpy
    return numpy


def _clip(arr, lo, hi):
    """Clip array values."""
    xp = _xp(arr)
    if xp.__name__ == "torch":
        result = arr.clone()
        if lo is not None:
            result = xp.clamp(result, min=lo)
        if hi is not None:
            result = xp.clamp(result, max=hi)
        return result
    return xp.clip(arr, lo, hi)


def _exp(arr):
    """Element-wise exponential."""
    xp = _xp(arr)
    return xp.exp(arr)


def _log(arr):
    """Element-wise natural log."""
    xp = _xp(arr)
    return xp.log(arr)


def _log1p(arr):
    """Element-wise log(1+x)."""
    xp = _xp(arr)
    return xp.log1p(arr)


def _sigmoid(arr):
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    xp = _xp(arr)
    z = _clip(arr, -500, 500)
    return 1.0 / (1.0 + xp.exp(-z))


def _sum(arr):
    """Sum of all elements."""
    xp = _xp(arr)
    return xp.sum(arr)


def _eigvalsh(arr):
    """Eigenvalues of a symmetric matrix (sorted ascending)."""
    xp = _xp(arr)
    return xp.linalg.eigvalsh(arr)
