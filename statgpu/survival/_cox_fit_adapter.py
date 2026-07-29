"""Shared public-boundary controls used directly by CoxPH and CoxPHCV.

The module name is retained for source compatibility with the PR #80 history,
but public classes no longer install or replace methods at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from statgpu._config import Device


_NATIVE_ARRAY_MODULES = ("cupy", "torch")
_TIE_METHODS = ("breslow", "efron", "exact")
_COVARIANCE_TYPES = ("nonrobust", "hc0", "hc1", "cluster")
_INFERENCE_MODES = ("strict", "approx")


@dataclass(frozen=True)
class _CoxFitControls:
    """Validated controls used by one CoxPH fit without mutating parameters."""

    ties: str
    tol: float
    max_iter: int
    device: Device
    compute_inference: bool
    compute_cindex: bool
    cov_type: str
    gpu_memory_cleanup: bool
    penalty: float
    inference_mode: str


@dataclass(frozen=True)
class _CoxCVFitControls:
    """Validated controls used by one CoxPHCV fit."""

    ties: str
    device: Device
    compute_inference: bool
    cov_type: str
    inference_mode: str
    gpu_memory_cleanup: bool


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


def _normalize_mutable_fit_controls(estimator) -> _CoxFitControls:
    """Return validated CoxPH controls without rewriting public parameters."""
    estimator._validate_optimization_controls()
    return _CoxFitControls(
        ties=_normalize_choice_control(estimator.ties, _TIE_METHODS, "ties"),
        tol=float(estimator.tol),
        max_iter=int(estimator.max_iter),
        device=_normalize_device_control(estimator.device),
        compute_inference=_normalize_boolean_control(
            estimator.compute_inference, "compute_inference"
        ),
        compute_cindex=_normalize_boolean_control(
            estimator.compute_cindex, "compute_cindex"
        ),
        cov_type=_normalize_choice_control(
            estimator.cov_type, _COVARIANCE_TYPES, "cov_type"
        ),
        gpu_memory_cleanup=_normalize_boolean_control(
            estimator.gpu_memory_cleanup, "gpu_memory_cleanup"
        ),
        penalty=float(estimator.penalty),
        inference_mode=_normalize_choice_control(
            estimator.inference_mode, _INFERENCE_MODES, "inference_mode"
        ),
    )


def _normalize_mutable_cv_controls(estimator) -> _CoxCVFitControls:
    """Return validated CoxPHCV controls without rewriting parameters."""
    controls = _CoxCVFitControls(
        ties=_normalize_choice_control(estimator.ties, _TIE_METHODS, "ties"),
        device=_normalize_device_control(estimator.device),
        compute_inference=_normalize_boolean_control(
            estimator.compute_inference, "compute_inference"
        ),
        cov_type=_normalize_choice_control(
            estimator.cov_type, _COVARIANCE_TYPES, "cov_type"
        ),
        inference_mode=_normalize_choice_control(
            estimator.inference_mode, _INFERENCE_MODES, "inference_mode"
        ),
        gpu_memory_cleanup=_normalize_boolean_control(
            estimator.gpu_memory_cleanup, "gpu_memory_cleanup"
        ),
    )
    if (
        controls.ties == "exact"
        and controls.compute_inference
        and controls.cov_type != "nonrobust"
    ):
        raise NotImplementedError(
            "robust covariance is not yet defined for ties='exact'; "
            "use cov_type='nonrobust' or compute_inference=False"
        )
    return controls


__all__ = [
    "_is_native_backend_array",
    "_CoxCVFitControls",
    "_CoxFitControls",
    "_PreencodedCoxLabels",
    "_normalize_boolean_control",
    "_normalize_mutable_fit_controls",
    "_normalize_mutable_cv_controls",
]
