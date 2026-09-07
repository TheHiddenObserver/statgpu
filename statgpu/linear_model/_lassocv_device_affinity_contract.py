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

Finally, once LassoCV AUTO routing has resolved a concrete CPU/CuPy/Torch target,
the selected-alpha full-data Lasso refit must use that same target.  Reuse the
existing #137 fit-device transaction instead of adding another ``fit`` wrapper:
native CuPy/Torch-CUDA input keeps its input-native priority when global AUTO is
active, while plain input resolves through the normal statgpu global-device
policy.  The existing transaction then pins the concrete device for both CV and
final refit and restores the caller-owned ``device='auto'`` state in ``finally``.
This preserves the warning stack discipline established by #135.

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
from statgpu.linear_model import _penalized_inference_api_contract as _api_contract
from statgpu.linear_model.cv._lasso_cv import LassoCV


_AFFINITY_MARKER = "__statgpu_pr138_lassocv_cupy_affinity__"
_AUTO_REFIT_MARKER = "__statgpu_pr138_lassocv_auto_refit_device__"
_ORIGINAL_PREPARE = LassoCV._prepare_cv_inputs_for_resolved_device
_ORIGINAL_INPUT_NATIVE_DEVICE = _api_contract._input_native_device


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


def _input_native_or_resolved_lassocv_device(self, X):
    """Return one concrete AUTO target for the whole LassoCV fit transaction."""
    target = _ORIGINAL_INPUT_NATIVE_DEVICE(self, X)
    if target is not None:
        return target
    if isinstance(self, LassoCV) and getattr(self, "_device", None) is Device.AUTO:
        return self._get_compute_device()
    return None


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

    if resolved_device is Device.CPU:
        return _convert_cpu_cv_inputs(self, X_cv, y_cv, sample_weight_cv)
    if resolved_device is Device.CUDA:
        return _align_cupy_cv_inputs(X_cv, y_cv, sample_weight_cv)
    return X_cv, y_cv, sample_weight_cv


def install_lassocv_device_affinity_contract():
    current_prepare = LassoCV._prepare_cv_inputs_for_resolved_device
    if not getattr(current_prepare, _AFFINITY_MARKER, False):
        setattr(_prepare_cv_inputs_for_resolved_device, _AFFINITY_MARKER, True)
        LassoCV._prepare_cv_inputs_for_resolved_device = (
            _prepare_cv_inputs_for_resolved_device
        )

    current_resolver = _api_contract._input_native_device
    if not getattr(current_resolver, _AUTO_REFIT_MARKER, False):
        setattr(
            _input_native_or_resolved_lassocv_device,
            _AUTO_REFIT_MARKER,
            True,
        )
        _api_contract._input_native_device = _input_native_or_resolved_lassocv_device


__all__ = [
    "install_lassocv_device_affinity_contract",
    "_align_cupy_cv_inputs",
    "_convert_cpu_cv_inputs",
    "_input_native_or_resolved_lassocv_device",
]
