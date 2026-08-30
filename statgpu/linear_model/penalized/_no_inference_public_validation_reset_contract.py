"""Fail-closed public finite-validation reset for estimation-only PGLM fits."""

from __future__ import annotations

import functools

from ._base import PenalizedGeneralizedLinearModel
from ._no_inference_cleanup_contract import _invalidate_failed_no_inference_fit


def _install_no_inference_public_validation_reset_contract() -> None:
    """Invalidate stale fitted state when outer finite validation rejects a refit."""
    current_reset = getattr(PenalizedGeneralizedLinearModel, "_reset_fit_state", None)
    if getattr(current_reset, "_statgpu_no_inference_fit_reset", False):
        return

    def _reset_fit_state(self):
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
