"""Backend-native validation helpers for public numerical inputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _raise_nonfinite(name: str) -> None:
    raise ValueError(f"{name} must contain only finite values; found NaN or Inf")


def check_finite(value: Any, *, name: str = "array") -> Any:
    """Reject NaN/Inf without transferring complete GPU arrays to CPU.

    Numeric NumPy, CuPy, Torch, pandas, scalar, and nested sequence
    inputs are checked. Non-numeric labels are intentionally ignored.
    Homogeneous Python sequences are converted once and checked with a
    vectorized reduction; only genuinely ragged sequences are traversed by
    top-level component. Only the final boolean reduction is synchronized for
    GPU arrays. The original object is returned unchanged.
    """
    if value is None:
        return value

    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        real = float(np.real(value))
        imag = float(np.imag(value))
        if not math.isfinite(real) or not math.isfinite(imag):
            _raise_nonfinite(name)
        return value
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return value

    module = type(value).__module__
    if module.startswith("torch"):
        import torch

        tensor = value
        if getattr(tensor, "is_sparse", False):
            tensor = tensor.coalesce().values()
        if not bool(torch.isfinite(tensor).all().item()):
            _raise_nonfinite(name)
        return value

    if module.startswith("cupy"):
        import cupy as cp

        if not bool(cp.isfinite(value).all().item()):
            _raise_nonfinite(name)
        return value

    if module.startswith("pandas"):
        if hasattr(value, "select_dtypes"):
            numeric = value.select_dtypes(include=["number", "bool"])
            if getattr(numeric, "shape", (0, 0))[1] == 0:
                return value
            array = numeric.to_numpy()
        else:
            array = value.to_numpy()
        if array.dtype.kind in "biufc" and not np.isfinite(array).all():
            _raise_nonfinite(name)
        return value

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_finite(item, name=f"{name}[{index}]")
        return value

    if array.dtype.kind in "biufc":
        if not np.isfinite(array).all():
            _raise_nonfinite(name)
        return value

    if array.dtype.kind == "O" and isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_finite(item, name=f"{name}[{index}]")
    return value
