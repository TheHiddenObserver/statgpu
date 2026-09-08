"""Concrete-device and final-refit affinity for LassoCV.

LassoCV resolves a backend before entering its dedicated CV selector. When a
native CuPy or Torch-CUDA design lives on a non-current CUDA device, the design
can remain on that concrete device while NumPy response/weight inputs are
allocated on the current device. Direct Gaussian/penalized fits already realign
those operands to the design's concrete device. Apply the same rule to LassoCV
so AUTO-native GPU input cannot create cross-device CV arithmetic.

Input alignment alone is not sufficient. The dedicated selector and its path
solvers also allocate fold indices, work arrays, and solver buffers using generic
CuPy/Torch creation APIs. Those APIs follow the current CUDA device when no
reference array is supplied. Bind the entire selector transaction to the design's
concrete device so a design on, for example, ``cuda:3`` cannot later acquire
``cuda:0`` indices or FISTA buffers merely because device 0 was current.

Finite-CV-evidence selection policy belongs to ``LassoCV.fit`` itself, not this
shared selector wrapper. Other private consumers of ``_select_lasso_alpha_cv``
(such as knockoff feature-selection helpers) therefore retain their existing
selection semantics. This module changes only execution-device ownership. Torch
CPU tensors are deliberately transparent here: only Torch CUDA designs need a
CUDA device context.

The inverse boundary matters as well: an explicit/resolved CPU request owns the
execution backend and must convert heterogeneous GPU-resident X/y/weights to
NumPy before the dedicated CV selector runs. This mirrors direct penalized-fit
semantics instead of asking the CV helper to reinterpret the input container.

Finally, once LassoCV AUTO routing has resolved a concrete CPU/CuPy/Torch target,
the selected-alpha full-data Lasso refit must use that same target. Reuse the
existing #137 fit-device transaction: native CuPy/Torch-CUDA input keeps its
input-native priority when global AUTO is active, while plain input resolves
through the normal statgpu global-device policy. The existing transaction pins
the concrete device for both CV and final refit and restores the caller-owned
``device='auto'`` state in ``finally``.

Maintenance tests also exercise synthetic backend doubles that intentionally do
not expose real CuPy/Torch concrete-device protocols. Those doubles remain
transparent to the production-only GPU affinity layer.
"""

from __future__ import annotations

import functools

from statgpu._config import Device
from statgpu.backends import _is_cupy_array, _is_torch_array
from statgpu.backends._utils import _cupy_asarray_on_device, _move_torch_tensor
from statgpu.linear_model import _penalized_inference_api_contract as _api_contract
from statgpu.linear_model.cv._lasso_cv import LassoCV
from statgpu.linear_model.wrappers import _lasso as _lasso_impl


_AFFINITY_MARKER = "__statgpu_pr138_lassocv_device_affinity__"
_AUTO_REFIT_MARKER = "__statgpu_pr138_lassocv_auto_refit_device__"
_SELECTOR_AFFINITY_MARKER = "__statgpu_pr138_lassocv_selector_device_affinity__"
_ORIGINAL_PREPARE = LassoCV._prepare_cv_inputs_for_resolved_device
_ORIGINAL_INPUT_NATIVE_DEVICE = _api_contract._input_native_device
_ORIGINAL_SELECT = _lasso_impl._select_lasso_alpha_cv


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


def _align_torch_cv_inputs(X_cv, y_cv, sample_weight_cv):
    """Keep response/weights on the concrete Torch CUDA device that owns X."""
    if not _is_torch_array(X_cv):
        return X_cv, y_cv, sample_weight_cv

    device = getattr(X_cv, "device", None)
    device_name = str(device or "")
    if not device_name.startswith("cuda:") and device_name != "cuda":
        raise RuntimeError(
            "LassoCV Torch preparation did not preserve a concrete CUDA design device."
        )
    y_aligned = _move_torch_tensor(y_cv, device=device_name)
    weight_aligned = (
        None
        if sample_weight_cv is None
        else _move_torch_tensor(sample_weight_cv, device=device_name)
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
    if resolved_device is Device.TORCH:
        return _align_torch_cv_inputs(X_cv, y_cv, sample_weight_cv)
    return X_cv, y_cv, sample_weight_cv


@functools.wraps(_ORIGINAL_SELECT)
def _select_lasso_alpha_cv_on_design_device(X, y, *args, **kwargs):
    """Run GPU selector allocations on the concrete device that owns X."""
    if _is_cupy_array(X):
        import cupy as cp

        device_id = getattr(getattr(X, "device", None), "id", None)
        if device_id is None:
            raise RuntimeError(
                "LassoCV CuPy selector requires a concrete design device."
            )
        with cp.cuda.Device(int(device_id)):
            return _ORIGINAL_SELECT(X, y, *args, **kwargs)

    if _is_torch_array(X):
        device = getattr(X, "device", None)
        device_name = str(device or "")
        if not device_name.startswith("cuda"):
            return _ORIGINAL_SELECT(X, y, *args, **kwargs)

        import torch

        with torch.cuda.device(device):
            return _ORIGINAL_SELECT(X, y, *args, **kwargs)

    return _ORIGINAL_SELECT(X, y, *args, **kwargs)


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

    current_selector = _lasso_impl._select_lasso_alpha_cv
    if not getattr(current_selector, _SELECTOR_AFFINITY_MARKER, False):
        setattr(_select_lasso_alpha_cv_on_design_device, _SELECTOR_AFFINITY_MARKER, True)
        _lasso_impl._select_lasso_alpha_cv = _select_lasso_alpha_cv_on_design_device


__all__ = [
    "install_lassocv_device_affinity_contract",
    "_align_cupy_cv_inputs",
    "_align_torch_cv_inputs",
    "_convert_cpu_cv_inputs",
    "_input_native_or_resolved_lassocv_device",
    "_select_lasso_alpha_cv_on_design_device",
]
