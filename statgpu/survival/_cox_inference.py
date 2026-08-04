"""Backend-preserving linear algebra for Cox coefficient inference.

The helpers in this module are intentionally stateless. Canonical and legacy
Cox implementations share the same rank and positive-definiteness contract
without coupling the public estimator to historical reference kernels.
"""

from __future__ import annotations

import numpy as np

from statgpu.inference._covariance import (
    classify_covariance_spectrum as _classify_covariance_spectrum,
    joint_wald_from_covariance as _joint_wald_from_covariance,
    standard_errors_from_covariance as _standard_errors_from_covariance,
)


_SINGULAR_INFORMATION_MESSAGE = (
    "Cox observed information is singular or not positive definite; "
    "coefficient inference is not identifiable"
)
_ROBUST_COVARIANCE_TYPES = {"hc0", "hc1", "cluster"}


def _validate_robust_inference_units(cov_type, n_units, n_features):
    """Validate independent-unit counts and return the finite-unit factor."""
    cov_type = str(cov_type).lower()
    n_units = int(n_units)
    n_features = int(n_features)
    if cov_type not in _ROBUST_COVARIANCE_TYPES:
        raise ValueError("cov_type must be 'hc0', 'hc1', or 'cluster'")
    if n_units < 2:
        raise RuntimeError(
            f"{cov_type} covariance requires at least two independent units"
        )
    if cov_type == "hc1":
        if n_units <= n_features:
            raise RuntimeError(
                "HC1 covariance requires n_units > n_features"
            )
        return n_units / (n_units - n_features)
    return 1.0


def _information_eigenvalue_tolerance(max_eigenvalue, n_features):
    """Return a scale-aware rank threshold for an information matrix."""
    return max(
        np.finfo(np.float64).tiny,
        float(max_eigenvalue) * max(int(n_features), 1) * 1e-12,
    )


def _invert_information_numpy(information):
    """Validate and invert a NumPy Cox observed-information matrix."""
    information = np.asarray(information, dtype=np.float64)
    information = 0.5 * (information + information.T)
    eigvals = np.linalg.eigvalsh(information)
    tolerance = _information_eigenvalue_tolerance(
        float(np.max(eigvals)), information.shape[0]
    )
    if not np.all(np.isfinite(eigvals)) or float(np.min(eigvals)) <= tolerance:
        raise RuntimeError(_SINGULAR_INFORMATION_MESSAGE)
    return np.linalg.solve(information, np.eye(information.shape[0]))


def _invert_information_cupy(information):
    """Validate and invert a CuPy Cox observed-information matrix."""
    import cupy as cp

    information = 0.5 * (information + information.T)
    eigvals = cp.linalg.eigvalsh(information)
    tolerance = _information_eigenvalue_tolerance(
        float(cp.max(eigvals).item()), information.shape[0]
    )
    if bool(cp.any(~cp.isfinite(eigvals)).item()) or float(
        cp.min(eigvals).item()
    ) <= tolerance:
        raise RuntimeError(_SINGULAR_INFORMATION_MESSAGE)
    return cp.linalg.solve(
        information, cp.eye(information.shape[0], dtype=information.dtype)
    )


def _invert_information_torch(information):
    """Validate and invert a Torch Cox observed-information matrix."""
    import torch

    information = 0.5 * (information + information.transpose(0, 1))
    eigvals = torch.linalg.eigvalsh(information)
    tolerance = _information_eigenvalue_tolerance(
        float(torch.max(eigvals).item()), information.shape[0]
    )
    if bool(torch.any(~torch.isfinite(eigvals)).item()) or float(
        torch.min(eigvals).item()
    ) <= tolerance:
        raise RuntimeError(_SINGULAR_INFORMATION_MESSAGE)
    identity = torch.eye(
        information.shape[0],
        dtype=information.dtype,
        device=information.device,
    )
    return torch.linalg.solve(information, identity)


__all__ = [
    "_classify_covariance_spectrum",
    "_information_eigenvalue_tolerance",
    "_invert_information_numpy",
    "_invert_information_cupy",
    "_invert_information_torch",
    "_joint_wald_from_covariance",
    "_standard_errors_from_covariance",
    "_validate_robust_inference_units",
]
