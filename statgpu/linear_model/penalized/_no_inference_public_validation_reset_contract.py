"""Cleanup and fail-closed reset for PGLM public finite validation."""

from __future__ import annotations

import functools
import sys

from ._base import PenalizedGeneralizedLinearModel
from ._no_inference_cleanup_contract import _invalidate_failed_no_inference_fit


_DIRECT_CUDA_CLEANUP = "_statgpu_finite_direct_cleanup_cuda"
_DIRECT_TORCH_CLEANUP = "_statgpu_finite_direct_cleanup_torch"


def _cupy_device_id(device_label: str) -> int | None:
    text = str(device_label or "").lower()
    if not text.startswith("cuda:"):
        return None
    try:
        return int(text.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def _best_effort_cleanup(cleanup) -> None:
    try:
        cleanup()
    except Exception:
        # Cleanup is advisory and must never replace the validation exception.
        return None


def _direct_cleanup(estimator, attr_name, fallback_name):
    cleanup = getattr(estimator, attr_name, None)
    if callable(cleanup):
        return cleanup
    return getattr(estimator, fallback_name)


def _cleanup_public_finite_validation(estimator, *, use_captured: bool) -> None:
    exc = sys.exc_info()[1]
    backend = str(getattr(exc, "_statgpu_finite_backend", None) or "").lower().strip()
    device = str(getattr(exc, "_statgpu_finite_device", None) or "").lower().strip()

    if backend == "torch":
        attr_name, fallback_name = _DIRECT_TORCH_CLEANUP, "_cleanup_torch_memory"
    elif backend == "cupy":
        attr_name, fallback_name = _DIRECT_CUDA_CLEANUP, "_cleanup_cuda_memory"
    else:
        return

    # Generic fits can use the methods captured before the inner transactions
    # wrapped them. Typed fits may fail before that capture; use their methods.
    cleanup = (
        _direct_cleanup(estimator, attr_name, fallback_name)
        if use_captured
        else getattr(estimator, fallback_name)
    )
    device_id = _cupy_device_id(device) if backend == "cupy" else None
    if device_id is None:
        _best_effort_cleanup(cleanup)
        return

    try:
        import cupy as cp
    except ImportError:
        _best_effort_cleanup(cleanup)
        return

    try:
        context = cp.cuda.Device(device_id)
        context.__enter__()
    except Exception:
        # The recorded device can itself be unavailable during error recovery.
        _best_effort_cleanup(cleanup)
        return

    try:
        _best_effort_cleanup(cleanup)
    finally:
        try:
            context.__exit__(None, None, None)
        except Exception:
            pass


def _cleanup_failed_public_finite_validation(estimator) -> None:
    """Clean finite-check temporaries on their device, not the last fit's."""
    _cleanup_public_finite_validation(estimator, use_captured=True)


def _install_no_inference_public_validation_reset_contract() -> None:
    """Invalidate stale fitted state when outer finite validation rejects a refit."""
    current_reset = getattr(PenalizedGeneralizedLinearModel, "_reset_fit_state", None)
    if getattr(current_reset, "_statgpu_no_inference_fit_reset", False):
        return

    def _reset_fit_state(self):
        # BaseEstimator calls this hook from inside the finite-validation
        # exception handler, before the public fit transaction is entered on
        # typed subclasses. Use the private exception provenance set by
        # check_finite() so only the allocator that created validation
        # temporaries is considered for cleanup. The configured cleanup method
        # remains a no-op when gpu_memory_cleanup=False.
        _cleanup_failed_public_finite_validation(self)
        if not bool(getattr(self, "compute_inference", False)):
            _invalidate_failed_no_inference_fit(self)
            return None
        if callable(current_reset):
            return current_reset(self)
        return None

    if callable(current_reset):
        functools.update_wrapper(_reset_fit_state, current_reset)
    _reset_fit_state._statgpu_no_inference_fit_reset = True
    _reset_fit_state._statgpu_original = current_reset
    PenalizedGeneralizedLinearModel._reset_fit_state = _reset_fit_state


_install_no_inference_public_validation_reset_contract()
