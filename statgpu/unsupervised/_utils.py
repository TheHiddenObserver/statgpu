"""Shared utilities for unsupervised estimators."""

from __future__ import annotations

from scipy import sparse


def check_2d_array(X, name: str = "X") -> None:
    """Validate that *X* is a non-empty 2D array-like object."""
    if getattr(X, "ndim", None) != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one sample and one feature")


def reject_sparse(X, estimator_name: str) -> None:
    """Raise a consistent error for unsupported sparse inputs."""
    if sparse.issparse(X):
        raise NotImplementedError(f"sparse input is not supported in {estimator_name} v1")


def scalar_to_float(x) -> float:
    """Convert a NumPy/CuPy/Torch scalar to Python float."""
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    if hasattr(x, "get"):
        return float(x.get())
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def scalar_to_int(x) -> int:
    """Convert a NumPy/CuPy/Torch scalar to Python int."""
    if hasattr(x, "detach"):
        return int(x.detach().cpu().item())
    if hasattr(x, "get"):
        return int(x.get())
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)
