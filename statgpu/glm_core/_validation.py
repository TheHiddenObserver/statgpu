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


def validate_glm_design_matrix(X, *, name="X"):
    """Validate a dense scalar-GLM design matrix and return its native array."""
    values = _as_native_array(X, name=name)
    if int(values.ndim) != 2:
        raise ValueError(f"{name} must be a two-dimensional design matrix.")
    if int(values.shape[0]) == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    _require_real_finite(values, name=name)
    return values


def validate_glm_sample_weight(sample_weight, n_samples, *, name="sample_weight"):
    """Validate analytic sample weights without copying GPU arrays to NumPy."""
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
        total = float(torch.sum(values).item())
    elif module.startswith("cupy"):
        import cupy as cp

        if bool(cp.any(values < 0).item()):
            raise ValueError(f"{name} must be non-negative")
        total = float(cp.sum(values).item())
    else:
        if np.any(values < 0):
            raise ValueError(f"{name} must be non-negative")
        total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a finite positive sum")
    return values
