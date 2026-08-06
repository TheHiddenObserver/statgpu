"""Strict numeric-grid validation shared by cross-validation frontends."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_numpy


def coerce_real_numeric_grid(values, *, name: str) -> np.ndarray:
    """Return a one-dimensional float64 grid without lossy coercion.

    Boolean values, numeric text, bytes, complex values, non-scalar object
    elements, and values that cannot be represented as finite/infinite float64
    scalars are rejected before NumPy can silently reinterpret them.
    Finiteness and sign constraints remain the responsibility of the caller.
    """
    if isinstance(values, (list, tuple)):
        raw = np.asarray(values, dtype=object)
    else:
        try:
            raw = np.asarray(_to_numpy(values))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain real numeric values") from exc

    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")

    kind = raw.dtype.kind
    if kind == "b":
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    if kind == "c":
        raise ValueError(f"{name} must contain real numeric values")
    if kind in {"S", "U"}:
        raise ValueError(
            f"{name} must contain real numeric values, not strings or bytes"
        )

    if kind == "O":
        grid = np.empty(raw.size, dtype=np.float64)
        for index, value in enumerate(raw.tolist()):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"{name} must contain real numeric values, not booleans"
                )
            if isinstance(value, (str, bytes, np.str_, np.bytes_)):
                raise ValueError(
                    f"{name} must contain real numeric values, not strings or bytes"
                )
            value_array = np.asarray(value)
            if value_array.ndim != 0:
                raise ValueError(f"{name} must contain scalar real numeric values")
            if value_array.dtype.kind == "c" or np.iscomplexobj(value):
                raise ValueError(f"{name} must contain real numeric values")
            if value_array.dtype.kind in {"b", "S", "U"}:
                label = (
                    "booleans"
                    if value_array.dtype.kind == "b"
                    else "strings or bytes"
                )
                raise ValueError(
                    f"{name} must contain real numeric values, not {label}"
                )
            try:
                grid[index] = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must contain real numeric values"
                ) from exc
        return grid

    if kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(raw, dtype=np.float64)


__all__ = ["coerce_real_numeric_grid"]
