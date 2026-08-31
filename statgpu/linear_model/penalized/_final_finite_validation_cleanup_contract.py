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
import sys

from . import _no_inference_public_validation_reset_contract as _public_reset_contract
from ._base import PenalizedGeneralizedLinearModel


_MISSING = object()
_DIRECT_CUDA_CLEANUP = "_statgpu_finite_direct_cleanup_cuda"
_DIRECT_TORCH_CLEANUP = "_statgpu_finite_direct_cleanup_torch"


def _restore_instance_attr(instance, name, previous) -> None:
    if previous is _MISSING:
        instance.__dict__.pop(name, None)
    else:
        instance.__dict__[name] = previous


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


def _cleanup_failed_public_finite_validation_direct(estimator) -> None:
    """Clean validation temporaries without following stale fit provenance."""
    exc = sys.exc_info()[1]
    backend = str(
        getattr(exc, "_statgpu_finite_backend", "") or ""
    ).lower().strip()
    device = str(
        getattr(exc, "_statgpu_finite_device", "") or ""
    ).lower().strip()

    if backend == "torch":
        cleanup = _direct_cleanup(
            estimator,
            _DIRECT_TORCH_CLEANUP,
            "_cleanup_torch_memory",
        )
        _best_effort_cleanup(cleanup)
        return

    if backend != "cupy":
        return

    cleanup = _direct_cleanup(
        estimator,
        _DIRECT_CUDA_CLEANUP,
        "_cleanup_cuda_memory",
    )
    if not device.startswith("cuda:"):
        _best_effort_cleanup(cleanup)
        return

    try:
        device_id = int(device.split(":", 1)[1])
    except (TypeError, ValueError):
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
        # If the recorded device itself cannot be entered, preserve the public
        # validation exception and retain the established best-effort fallback.
        _best_effort_cleanup(cleanup)
        return

    try:
        _best_effort_cleanup(cleanup)
    finally:
        try:
            context.__exit__(None, None, None)
        except Exception:
            pass


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

    # The installed reset hook resolves this module-global function dynamically,
    # so replace only its cleanup implementation; reset/invalidation ownership is
    # otherwise unchanged.
    _public_reset_contract._cleanup_failed_public_finite_validation = (
        _cleanup_failed_public_finite_validation_direct
    )


_install_direct_public_validation_cleanup_contract()
