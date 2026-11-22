"""Direct-device cleanup contract for outer PGLM finite validation.

The public finite-input guard executes inside the fit wrapper stack.  Gaussian
and estimation-only transactions temporarily replace the estimator cleanup
methods with tracked delegates that consult fitted-device provenance.  During a
refit rejected by the outer finite guard, that provenance can still describe a
previous fit.  Capture the direct cleanup callables before those inner wrappers
install their delegates, then use the finite-check exception's concrete device
when cleanup is required.
"""

from __future__ import annotations

import functools

from . import _no_inference_public_validation_reset_contract as _public_reset_contract
from ._base import PenalizedGeneralizedLinearModel
from ._no_inference_public_validation_reset_contract import (
    _DIRECT_CUDA_CLEANUP,
    _DIRECT_TORCH_CLEANUP,
    _best_effort_cleanup as _best_effort_cleanup,
    _direct_cleanup as _direct_cleanup,
)


_MISSING = object()


def _restore_instance_attr(instance, name, previous) -> None:
    if previous is _MISSING:
        instance.__dict__.pop(name, None)
    else:
        instance.__dict__[name] = previous


def _cleanup_failed_public_finite_validation_direct(estimator) -> None:
    """Compatibility entry point for the direct-device cleanup."""
    _public_reset_contract._cleanup_public_finite_validation(
        estimator, use_captured=True
    )


def _install_direct_public_validation_cleanup_contract() -> None:
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, "_statgpu_direct_finite_cleanup_contract", False):
        return

    @functools.wraps(current)
    def _fit_with_direct_finite_cleanup(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        saved_cuda = self.__dict__.get(_DIRECT_CUDA_CLEANUP, _MISSING)
        saved_torch = self.__dict__.get(_DIRECT_TORCH_CLEANUP, _MISSING)

        # Capture cleanup before Gaussian/no-inference inner transactions replace
        # these methods with delegates that read mutable fitted-device state.
        self.__dict__[_DIRECT_CUDA_CLEANUP] = self._cleanup_cuda_memory
        self.__dict__[_DIRECT_TORCH_CLEANUP] = self._cleanup_torch_memory
        try:
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        finally:
            _restore_instance_attr(self, _DIRECT_CUDA_CLEANUP, saved_cuda)
            _restore_instance_attr(self, _DIRECT_TORCH_CLEANUP, saved_torch)

    _fit_with_direct_finite_cleanup._statgpu_direct_finite_cleanup_contract = True
    _fit_with_direct_finite_cleanup._statgpu_original = current
    PenalizedGeneralizedLinearModel.fit = _fit_with_direct_finite_cleanup


_install_direct_public_validation_cleanup_contract()
