"""Fail-closed public finite-validation reset for estimation-only PGLM fits."""

from __future__ import annotations

import functools
import sys

from ._base import PenalizedGeneralizedLinearModel
from ._no_inference_cleanup_contract import _invalidate_failed_no_inference_fit


def _cleanup_failed_public_finite_validation(estimator) -> None:
    """Release accelerator temporaries created by the outer finite-input guard."""
    exc = sys.exc_info()[1]
    backend = str(
        getattr(exc, "_statgpu_finite_backend", "") or ""
    ).lower()
    if backend == "cupy":
        estimator._cleanup_cuda_memory()
    elif backend == "torch":
        estimator._cleanup_torch_memory()


def _install_no_inference_public_validation_reset_contract() -> None:
    """Invalidate stale fitted state when outer finite validation rejects a refit."""
    current_reset = getattr(PenalizedGeneralizedLinearModel, "_reset_fit_state", None)
    if getattr(current_reset, "_statgpu_no_inference_fit_reset", False):
        return

    def _reset_fit_state(self):
        # BaseEstimator calls this hook from inside the finite-validation
        # exception handler, before the public fit transaction is entered on
        # typed subclasses.  Use the private exception provenance set by
        # check_finite() so only the allocator that created validation
        # temporaries is considered for cleanup.  The configured cleanup method
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
