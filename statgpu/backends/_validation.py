"""Backend-native validation helpers for public numerical inputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _tag_finite_backend(
    exc: BaseException,
    backend: str | None,
    *,
    device: str | None = None,
) -> None:
    """Attach private accelerator provenance without changing public exceptions."""
    if backend is None:
        return
    try:
        exc._statgpu_finite_backend = str(backend).lower()
        if device is not None:
            exc._statgpu_finite_device = str(device).lower()
    except Exception:
        # Exception subclasses are normally mutable, but provenance is best-effort
        # and must never mask the original validation failure.
        pass


def _raise_nonfinite(
    name: str,
    *,
    backend: str | None = None,
    device: str | None = None,
) -> None:
    exc = ValueError(
        f"{name} must contain only finite values; found NaN or infinite values"
    )
    # Private provenance for fit reset/cleanup contracts. Keep the public
    # exception type and message unchanged while recording which accelerator
    # allocator produced finite-check temporaries before the fit transaction.
    _tag_finite_backend(exc, backend, device=device)
    raise exc


def _torch_cuda_device_label(device: Any) -> str | None:
    """Return a concrete cuda:N label when a Torch device identifies one."""
    if str(getattr(device, "type", "")) != "cuda":
        return None
    text = str(device)
    if text.startswith("cuda:"):
        return text
    index = getattr(device, "index", None)
    if index is not None:
        return f"cuda:{int(index)}"
    return None


def _cupy_cuda_device_label(value: Any) -> str | None:
    """Return a concrete cuda:N label for a CuPy value when available."""
    device_id = getattr(getattr(value, "device", None), "id", None)
    if device_id is None:
        return None
    try:
        return f"cuda:{int(device_id)}"
    except (TypeError, ValueError):
        return None


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
        tensor_device = getattr(tensor, "device", None)
        device_type = str(getattr(tensor_device, "type", ""))
        backend = "torch" if device_type == "cuda" else None
        device = _torch_cuda_device_label(tensor_device) if backend else None
        try:
            if getattr(tensor, "layout", torch.strided) != torch.strided:
                tensor = tensor.values()
            finite = bool(torch.isfinite(tensor).all().item())
        except Exception as exc:
            _tag_finite_backend(exc, backend, device=device)
            raise
        if not finite:
            _raise_nonfinite(name, backend=backend, device=device)
        return value

    if module.startswith("cupy"):
        import cupy as cp

        device = _cupy_cuda_device_label(value)
        try:
            finite = bool(cp.isfinite(value).all().item())
        except Exception as exc:
            _tag_finite_backend(exc, "cupy", device=device)
            raise
        if not finite:
            _raise_nonfinite(name, backend="cupy", device=device)
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
