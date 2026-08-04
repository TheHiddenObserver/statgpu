"""Backend-native validation helpers for public numerical inputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _raise_nonfinite(name: str) -> None:
    raise ValueError(
        f"{name} must contain only finite values; found NaN or infinite values"
    )


def check_finite(value: Any, *, name: str = "array") -> Any:
    """Reject NaN/Inf without transferring complete GPU arrays to CPU.

    NumPy, CuPy, Torch, scipy/cupyx sparse values, pandas numerical data,
    scalars, and nested/object sequences are checked. GPU arrays perform the
    reduction on device and synchronize only the final scalar boolean. The
    original object is returned unchanged.
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

    if module.startswith("scipy.sparse") or module.startswith("cupyx.scipy.sparse"):
        check_finite(value.data, name=name)
        return value

    if module.startswith("torch"):
        import torch

        tensor = value
        if getattr(tensor, "layout", torch.strided) != torch.strided:
            tensor = tensor.values()
        if not bool(torch.isfinite(tensor).all().item()):
            _raise_nonfinite(name)
        return value

    if module.startswith("cupy"):
        import cupy as cp

        if not bool(cp.isfinite(value).all().item()):
            _raise_nonfinite(name)
        return value

    if module.startswith("pandas"):
        import pandas as pd

        missing = pd.isna(value)
        if hasattr(missing, "to_numpy"):
            missing = missing.to_numpy()
        if bool(np.asarray(missing).any()):
            _raise_nonfinite(name)
        try:
            array = value.to_numpy()
        except Exception:
            return value
        if array.dtype.kind in "biufc":
            if not np.isfinite(array).all():
                _raise_nonfinite(name)
            return value
        if array.dtype.kind == "O":
            for index, item in np.ndenumerate(array):
                check_finite(item, name=f"{name}{index}")
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

    if array.dtype.kind == "O":
        for index, item in np.ndenumerate(array):
            if item is value:
                continue
            check_finite(item, name=f"{name}{index}")
    return value
