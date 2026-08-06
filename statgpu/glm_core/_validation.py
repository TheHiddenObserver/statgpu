"""Backend-native validation for scalar GLM design and weight inputs."""

from __future__ import annotations

import numpy as np


def _as_native_array(value, *, name):
    """Return an array while preserving existing Torch/CuPy device residency."""
    module = type(value).__module__
    if module.startswith("pandas"):
        value = value.to_numpy()
        module = type(value).__module__
    if module.startswith("torch"):
        import torch

        return value if torch.is_tensor(value) else torch.as_tensor(value)
    if module.startswith("cupy"):
        import cupy as cp

        return value if isinstance(value, cp.ndarray) else cp.asarray(value)
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array-like.") from exc


def _require_real_finite(values, *, name):
    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        if torch.is_complex(values):
            raise ValueError(f"{name} must contain real numeric values.")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError(f"{name} must contain finite values.")
        return
    if module.startswith("cupy"):
        import cupy as cp

        if getattr(values.dtype, "kind", "") not in "biuf":
            raise ValueError(f"{name} must contain real numeric values.")
        if not bool(cp.all(cp.isfinite(values)).item()):
            raise ValueError(f"{name} must contain finite values.")
        return

    if getattr(values.dtype, "kind", "") not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain finite values.")


def _safe_weight_sum(values) -> float:
    """Accumulate analytic weights in float64 to avoid integer wraparound."""
    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        total = torch.sum(values.to(dtype=torch.float64))
        return float(total.item())
    if module.startswith("cupy"):
        import cupy as cp

        total = cp.sum(values, dtype=cp.float64)
        return float(total.item())
    return float(np.sum(np.asarray(values), dtype=np.float64))


def validate_glm_design_matrix(X, *, name="X"):
    """Validate a dense scalar-GLM design matrix and return its native array."""
    values = _as_native_array(X, name=name)
    if int(values.ndim) != 2:
        raise ValueError(f"{name} must be a two-dimensional design matrix.")
    if int(values.shape[0]) == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    _require_real_finite(values, name=name)
    return values


def validate_binary_response(y, n_samples=None, *, context="LogisticRegression"):
    """Validate a strict 0/1 response while preserving GPU residency."""
    values = _as_native_array(y, name="binary y")
    if int(values.ndim) == 2 and int(values.shape[1]) == 1:
        values = values.reshape(-1)
    elif int(values.ndim) != 1:
        raise ValueError(f"{context} requires one-dimensional binary y")
    if int(values.shape[0]) == 0:
        raise ValueError(f"{context} requires at least one binary response")
    if n_samples is not None and int(values.shape[0]) != int(n_samples):
        raise ValueError("Response length must match the number of X rows.")
    _require_real_finite(values, name="binary y")

    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        valid = torch.all((values == 0) | (values == 1))
    elif module.startswith("cupy"):
        import cupy as cp

        valid = cp.all((values == 0) | (values == 1))
    else:
        valid = np.all((values == 0) | (values == 1))
    if not bool(valid.item() if hasattr(valid, "item") else valid):
        raise ValueError(f"{context} requires binary y with values 0 or 1")
    return values


def validate_glm_sample_weight(sample_weight, n_samples, *, name="sample_weight"):
    """Validate analytic weights and normalize integral inputs to float64."""
    values = _as_native_array(sample_weight, name=name)
    if int(values.ndim) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if int(values.shape[0]) != int(n_samples):
        raise ValueError(f"{name} must have length n_samples")
    _require_real_finite(values, name=name)

    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        if bool(torch.any(values < 0).item()):
            raise ValueError(f"{name} must be non-negative")
    elif module.startswith("cupy"):
        import cupy as cp

        if bool(cp.any(values < 0).item()):
            raise ValueError(f"{name} must be non-negative")
    else:
        if np.any(values < 0):
            raise ValueError(f"{name} must be non-negative")
    total = _safe_weight_sum(values)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a finite positive sum")

    # Returning integer weights would reintroduce wraparound in downstream
    # objective normalizers that call ``sum()`` directly. Preserve device
    # residency while promoting integral/bool weights once at validation.
    kind = getattr(values.dtype, "kind", "")
    if module.startswith("torch"):
        import torch

        if not torch.is_floating_point(values):
            values = values.to(dtype=torch.float64)
    elif kind in "biu":
        if module.startswith("cupy"):
            import cupy as cp

            values = values.astype(cp.float64, copy=False)
        else:
            values = values.astype(np.float64, copy=False)
    return values
