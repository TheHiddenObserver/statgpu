"""Shared public-boundary controls used directly by CoxPH and CoxPHCV.

The module name is retained for source compatibility with the PR #80 history,
but public classes no longer install or replace methods at import time.
"""

from __future__ import annotations

import numpy as np

from statgpu._config import Device


_NATIVE_ARRAY_MODULES = ("cupy", "torch")
_TIE_METHODS = ("breslow", "efron", "exact")
_COVARIANCE_TYPES = ("nonrobust", "hc0", "hc1", "cluster")
_INFERENCE_MODES = ("strict", "approx")


class _PreencodedCoxLabels:
    """Internal backend-native group codes with host display labels."""

    __slots__ = ("codes", "labels")

    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = np.asarray(labels).copy()


def _is_native_backend_array(value) -> bool:
    """Return whether slicing ``value`` preserves a CuPy/Torch backend."""
    return type(value).__module__.startswith(_NATIVE_ARRAY_MODULES)


def _normalize_boolean_control(value, name: str) -> bool:
    """Normalize an actual boolean or integer 0/1 without truthy strings."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or integer 0/1")


def _normalize_device_control(value) -> Device:
    """Normalize a public device value without silently selecting CPU."""
    try:
        return value if isinstance(value, Device) else Device(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "device must be one of: 'auto', 'cpu', 'cuda', or 'torch'"
        ) from exc


def _normalize_choice_control(value, choices, name: str) -> str:
    """Lowercase and validate a finite string-like choice control."""
    normalized = str(value).lower()
    if normalized not in choices:
        if name == "ties":
            raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
        if name == "cov_type":
            raise ValueError(
                "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'"
            )
        raise ValueError("inference_mode must be strict or approx")
    return normalized


def _normalize_mutable_fit_controls(estimator) -> None:
    """Revalidate CoxPH controls that may have changed through ``set_params``."""
    estimator._validate_optimization_controls()
    estimator.tol = float(estimator.tol)
    estimator.penalty = float(estimator.penalty)
    estimator.ties = _normalize_choice_control(
        estimator.ties, _TIE_METHODS, "ties"
    )
    estimator.cov_type = _normalize_choice_control(
        estimator.cov_type, _COVARIANCE_TYPES, "cov_type"
    )
    estimator.inference_mode = _normalize_choice_control(
        estimator.inference_mode, _INFERENCE_MODES, "inference_mode"
    )
    estimator.device = _normalize_device_control(estimator.device)
    estimator.compute_inference = _normalize_boolean_control(
        estimator.compute_inference, "compute_inference"
    )
    estimator.compute_cindex = _normalize_boolean_control(
        estimator.compute_cindex, "compute_cindex"
    )
    estimator.gpu_memory_cleanup = _normalize_boolean_control(
        estimator.gpu_memory_cleanup, "gpu_memory_cleanup"
    )


def _normalize_mutable_cv_controls(estimator) -> None:
    """Validate CoxPHCV controls before any fold fitting is attempted."""
    estimator.ties = _normalize_choice_control(
        estimator.ties, _TIE_METHODS, "ties"
    )
    estimator.cov_type = _normalize_choice_control(
        estimator.cov_type, _COVARIANCE_TYPES, "cov_type"
    )
    estimator.inference_mode = _normalize_choice_control(
        estimator.inference_mode, _INFERENCE_MODES, "inference_mode"
    )
    estimator.device = _normalize_device_control(estimator.device)
    estimator.compute_inference = _normalize_boolean_control(
        estimator.compute_inference, "compute_inference"
    )
    estimator.gpu_memory_cleanup = _normalize_boolean_control(
        estimator.gpu_memory_cleanup, "gpu_memory_cleanup"
    )
    if (
        estimator.ties == "exact"
        and estimator.compute_inference
        and estimator.cov_type != "nonrobust"
    ):
        raise NotImplementedError(
            "robust covariance is not yet defined for ties='exact'; "
            "use cov_type='nonrobust' or compute_inference=False"
        )


__all__ = [
    "_is_native_backend_array",
    "_PreencodedCoxLabels",
    "_normalize_boolean_control",
    "_normalize_mutable_fit_controls",
    "_normalize_mutable_cv_controls",
]
