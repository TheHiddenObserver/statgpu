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

Within the public ``LassoCV.fit`` transaction, genuine multi-alpha selection must
also use complete finite CV evidence. A candidate that failed numerically on one
or more folds must not become eligible merely because its remaining finite folds
have a small mean. Only candidates with finite MSE on every actually executed
fold are eligible; if none remain, selection fails closed. Applying the same rule
to weighted and unweighted LassoCV calls preserves the public identity that
positive constant analytic weights are exactly the unweighted statistical
problem. Degenerate no-selection paths (single alpha, too few rows, or fewer
than two executed folds) preserve their historical refit semantics. The evidence
check derives the fold count from the returned MSE matrix instead of re-reading
``cv_splits`` so one-shot iterables/generators remain valid public inputs.

The finite-evidence policy is deliberately scoped to the already-existing
``LassoCV.fit`` device wrapper. During that fit, input preparation arms a
context-local one-shot marker; the immediately following selector consumes and
clears it even on failure. Direct calls to the shared selector, including knockoff
feature-selection utilities, therefore retain their existing selection semantics.
This avoids adding another transparent ``fit`` frame, so the established
caller-facing deprecation-warning stacklevel remains unchanged.

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

from contextvars import ContextVar
import functools

import numpy as np

from statgpu._config import Device
from statgpu.backends import _is_cupy_array, _is_torch_array
from statgpu.backends._utils import _cupy_asarray_on_device, _move_torch_tensor
from statgpu.linear_model import _penalized_inference_api_contract as _api_contract
from statgpu.linear_model.cv._lasso_cv import LassoCV
from statgpu.linear_model.wrappers import _lasso as _lasso_impl


_AFFINITY_MARKER = "__statgpu_pr138_lassocv_device_affinity__"
_AUTO_REFIT_MARKER = "__statgpu_pr138_lassocv_auto_refit_device__"
_SELECTOR_AFFINITY_MARKER = "__statgpu_pr138_lassocv_selector_device_affinity__"
_LASSOCV_SELECTION_SCOPE = ContextVar(
    "statgpu_pr138_lassocv_selection_scope",
    default=False,
)
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


def _arm_lassocv_selection_scope(self) -> None:
    """Arm one selector call only when preparation runs inside the existing fit wrapper."""
    depth = int(getattr(self, "_statgpu_post_selection_fit_device_depth", 0))
    if depth > 0:
        _LASSOCV_SELECTION_SCOPE.set(True)


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
        prepared = _convert_cpu_cv_inputs(self, X_cv, y_cv, sample_weight_cv)
    elif resolved_device is Device.CUDA:
        prepared = _align_cupy_cv_inputs(X_cv, y_cv, sample_weight_cv)
    elif resolved_device is Device.TORCH:
        prepared = _align_torch_cv_inputs(X_cv, y_cv, sample_weight_cv)
    else:
        prepared = (X_cv, y_cv, sample_weight_cv)

    _arm_lassocv_selection_scope(self)
    return prepared


def _validate_weighted_cv_selection_evidence(X, result, *, kwargs):
    """Return details whose selected alpha is supported by every executed fold."""
    if not isinstance(result, dict):
        return result

    alphas = np.asarray(result.get("alphas", ()), dtype=np.float64).reshape(-1)
    mse_path = np.asarray(result.get("mse_path", ()), dtype=np.float64)
    n_samples = int(getattr(X, "shape", (0,))[0])

    if mse_path.ndim != 2 or mse_path.shape[0] != alphas.size:
        raise RuntimeError(
            "LassoCV returned an inconsistent validation-MSE layout"
        )
    n_folds_evaluated = int(mse_path.shape[1])

    if n_samples < 4 or alphas.size <= 1 or n_folds_evaluated < 2:
        return result

    complete = np.all(np.isfinite(mse_path), axis=1)
    if not np.any(complete):
        raise FloatingPointError(
            "LassoCV produced no candidate with finite validation MSE on every "
            "fold; refusing to select alpha"
        )

    mean_mse = np.full(alphas.size, np.nan, dtype=np.float64)
    mean_mse[complete] = np.mean(mse_path[complete], axis=1)
    eligible_indices = np.flatnonzero(complete)
    best_local = int(np.argmin(mean_mse[complete]))
    best_index = int(eligible_indices[best_local])

    checked = dict(result)
    checked["alpha"] = float(alphas[best_index])
    checked["mean_mse"] = mean_mse
    return checked


def _call_selector_with_evidence(X, y, args, kwargs):
    """Run the selector and validate multi-alpha finite evidence."""
    requested_details = bool(kwargs.get("return_details", False))
    call_kwargs = kwargs
    if not requested_details:
        call_kwargs = dict(kwargs)
        call_kwargs["return_details"] = True

    result = _ORIGINAL_SELECT(X, y, *args, **call_kwargs)
    result = _validate_weighted_cv_selection_evidence(X, result, kwargs=call_kwargs)
    if not requested_details:
        return float(result["alpha"])
    return result


def _selector_call_in_scope(X, y, args, kwargs):
    active = bool(_LASSOCV_SELECTION_SCOPE.get())
    if not active:
        return _ORIGINAL_SELECT(X, y, *args, **kwargs)

    # Preparation arms exactly the next selector call in this execution context.
    # Clear first so nested/private selector calls inside the implementation do
    # not accidentally inherit the LassoCV-only evidence policy.
    _LASSOCV_SELECTION_SCOPE.set(False)
    return _call_selector_with_evidence(X, y, args, kwargs)


@functools.wraps(_ORIGINAL_SELECT)
def _select_lasso_alpha_cv_on_design_device(X, y, *args, **kwargs):
    """Bind selector allocation to X's device; validate evidence only for LassoCV."""
    if _is_cupy_array(X):
        import cupy as cp

        device_id = getattr(getattr(X, "device", None), "id", None)
        if device_id is None:
            raise RuntimeError(
                "LassoCV CuPy selector requires a concrete design device."
            )
        with cp.cuda.Device(int(device_id)):
            return _selector_call_in_scope(X, y, args, kwargs)

    if _is_torch_array(X):
        import torch

        device = getattr(X, "device", None)
        device_name = str(device or "")
        if not device_name.startswith("cuda"):
            raise RuntimeError(
                "LassoCV Torch selector requires a concrete CUDA design device."
            )
        with torch.cuda.device(device):
            return _selector_call_in_scope(X, y, args, kwargs)

    return _selector_call_in_scope(X, y, args, kwargs)


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
        setattr(
            _select_lasso_alpha_cv_on_design_device,
            _SELECTOR_AFFINITY_MARKER,
            True,
        )
        _lasso_impl._select_lasso_alpha_cv = _select_lasso_alpha_cv_on_design_device


__all__ = [
    "install_lassocv_device_affinity_contract",
    "_align_cupy_cv_inputs",
    "_align_torch_cv_inputs",
    "_convert_cpu_cv_inputs",
    "_input_native_or_resolved_lassocv_device",
    "_select_lasso_alpha_cv_on_design_device",
    "_validate_weighted_cv_selection_evidence",
    "_LASSOCV_SELECTION_SCOPE",
    "_arm_lassocv_selection_scope",
]
