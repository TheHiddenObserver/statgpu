"""Concrete-device affinity for LassoCV CuPy preparation.

LassoCV resolves a backend before entering its dedicated CV selector.  When a
native CuPy design lives on a non-current CUDA device, BaseEstimator._to_cupy
preserves that design array while NumPy response/weight inputs are allocated on
the current device.  Direct Gaussian/penalized fits already realign those
operands to the design's concrete device.  Apply the same rule to LassoCV so
AUTO-native CuPy input cannot create cross-device CV arithmetic.

Maintenance tests also exercise synthetic backend doubles that intentionally do
not expose CuPy's concrete ``.device.id`` contract.  Those are not physical
CuPy arrays and must remain transparent to this production-only affinity layer.
"""

from __future__ import annotations

import functools

from statgpu.backends import _is_cupy_array
from statgpu.backends._utils import _cupy_asarray_on_device
from statgpu.linear_model.cv._lasso_cv import LassoCV


_AFFINITY_MARKER = "__statgpu_pr138_lassocv_cupy_affinity__"
_ORIGINAL_PREPARE = LassoCV._prepare_cv_inputs_for_resolved_device


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
    if str(device_name).strip().lower() == "cuda":
        return _align_cupy_cv_inputs(X_cv, y_cv, sample_weight_cv)
    return X_cv, y_cv, sample_weight_cv


def install_lassocv_device_affinity_contract():
    current = LassoCV._prepare_cv_inputs_for_resolved_device
    if getattr(current, _AFFINITY_MARKER, False):
        return
    setattr(_prepare_cv_inputs_for_resolved_device, _AFFINITY_MARKER, True)
    LassoCV._prepare_cv_inputs_for_resolved_device = _prepare_cv_inputs_for_resolved_device


__all__ = [
    "install_lassocv_device_affinity_contract",
    "_align_cupy_cv_inputs",
]
