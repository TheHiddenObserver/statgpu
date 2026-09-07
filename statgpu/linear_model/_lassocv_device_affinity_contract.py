"""Concrete-device and final-refit affinity for LassoCV.

LassoCV resolves a backend before entering its dedicated CV selector.  When a
native CuPy design lives on a non-current CUDA device, BaseEstimator._to_cupy
preserves that design array while NumPy response/weight inputs are allocated on
the current device.  Direct Gaussian/penalized fits already realign those
operands to the design's concrete device.  Apply the same rule to LassoCV so
AUTO-native CuPy input cannot create cross-device CV arithmetic.

The inverse boundary matters as well: an explicit/resolved CPU request owns the
execution backend and must convert heterogeneous GPU-resident X/y/weights to
NumPy before the dedicated CV selector runs.  This mirrors direct penalized-fit
semantics instead of asking the CV helper to reinterpret the input container.

Finally, once LassoCV has resolved a concrete CPU/CuPy/Torch device for CV, the
selected-alpha full-data Lasso refit must use that same device.  Temporarily pin
the estimator's private runtime device after CV preparation and restore the
caller's public AUTO/explicit request transactionally around ``fit``.  This
matches the shared ``cv_refit_device`` contract used by the other maintained CV
estimators and prevents a second AUTO decision during final refit.

Maintenance tests also exercise synthetic backend doubles that intentionally do
not expose CuPy's concrete ``.device.id`` contract.  Those are not physical
CuPy arrays and must remain transparent to the production-only CUDA affinity
layer.
"""

from __future__ import annotations

import functools

from statgpu._config import Device
from statgpu.backends import _is_cupy_array
from statgpu.backends._utils import _cupy_asarray_on_device
from statgpu.linear_model.cv._lasso_cv import LassoCV


_AFFINITY_MARKER = "__statgpu_pr138_lassocv_cupy_affinity__"
_FIT_AFFINITY_MARKER = "__statgpu_pr138_lassocv_refit_affinity__"
_ORIGINAL_PREPARE = LassoCV._prepare_cv_inputs_for_resolved_device
_ORIGINAL_FIT = LassoCV.fit


def _align_cupy_cv_inputs(X_cv, y_cv, sample_weight_cv):
    # Synthetic maintenance backends deliberately do not implement CuPy's
    # concrete-device protocol. Keep those test doubles transparent while
    # requiring exact ordinal affinity for real CuPy arrays.
    if not _is_cupy_array(X_cv):
        return X_cv, y_cv, sample_weight_cv

    device_id = getattr(getattr(X_cv, "device", None), "id", None)
    if device_id is None:
        raise RuntimeError(
            "LassoCV CuPy preparation did not preserve a concrete design device."
        )
    device_id = int(device_id)
    y_aligned = _cupy_asarray_on_device(y_cv, device_id)
    weight_aligned = (
        None
        if sample_weight_cv is None
        else _cupy_asarray_on_device(sample_weight_cv, device_id)
    )
    return X_cv, y_aligned, weight_aligned


def _convert_cpu_cv_inputs(self, X_cv, y_cv, sample_weight_cv):
    """Convert all dedicated-CV operands to NumPy for an authoritative CPU target."""
    X_cpu = self._to_array(X_cv, Device.CPU, backend="numpy")
    y_cpu = self._to_array(y_cv, Device.CPU, backend="numpy")
    weight_cpu = (
        None
        if sample_weight_cv is None
        else self._to_array(sample_weight_cv, Device.CPU, backend="numpy")
    )
    return X_cpu, y_cpu, weight_cpu


@functools.wraps(_ORIGINAL_PREPARE)
def _prepare_cv_inputs_for_resolved_device(
    self,
    X,
    y,
    sample_weight,
    device_name: str,
):
    X_cv, y_cv, sample_weight_cv = _ORIGINAL_PREPARE(
        self,
        X,
        y,
        sample_weight,
        device_name,
    )
    resolved = str(device_name).strip().lower()
    try:
        resolved_device = Device(resolved)
    except ValueError as exc:
        raise ValueError(
            f"LassoCV resolved an unsupported execution device {device_name!r}."
        ) from exc

    # Pin the private runtime device from this point through final refit.  The
    # outer transactional fit wrapper below restores the caller-owned request.
    self._device = resolved_device

    if resolved_device is Device.CPU:
        return _convert_cpu_cv_inputs(self, X_cv, y_cv, sample_weight_cv)
    if resolved_device is Device.CUDA:
        return _align_cupy_cv_inputs(X_cv, y_cv, sample_weight_cv)
    return X_cv, y_cv, sample_weight_cv


@functools.wraps(_ORIGINAL_FIT)
def _fit_with_resolved_refit_affinity(self, *args, **kwargs):
    requested_device = self._device
    try:
        return _ORIGINAL_FIT(self, *args, **kwargs)
    finally:
        self._device = requested_device


def install_lassocv_device_affinity_contract():
    current_prepare = LassoCV._prepare_cv_inputs_for_resolved_device
    if not getattr(current_prepare, _AFFINITY_MARKER, False):
        setattr(_prepare_cv_inputs_for_resolved_device, _AFFINITY_MARKER, True)
        LassoCV._prepare_cv_inputs_for_resolved_device = (
            _prepare_cv_inputs_for_resolved_device
        )

    current_fit = LassoCV.fit
    if not getattr(current_fit, _FIT_AFFINITY_MARKER, False):
        setattr(_fit_with_resolved_refit_affinity, _FIT_AFFINITY_MARKER, True)
        LassoCV.fit = _fit_with_resolved_refit_affinity


__all__ = [
    "install_lassocv_device_affinity_contract",
    "_align_cupy_cv_inputs",
    "_convert_cpu_cv_inputs",
]
