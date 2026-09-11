"""Fit-transaction binding for the penalized-GLM inference resolver.

The public constructor parameter remains the user's request (for example
``auto``), while GPU sparse inference needs the concrete dispatch method during
``_fit_gpu`` / ``_fit_torch``. Bind the resolved method immediately after the
pre-fit inference validator and restore the public request after the complete
fit transaction.

A failed refit must also stop advertising any prior successful fit. This matters
because inference compatibility is validated before the core fit clears fitted
state; a newly unsupported method/loss/penalty row can therefore fail before the
usual reset point.
"""

from __future__ import annotations

import functools

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


_MARKER = "_statgpu_penalized_glm_inference_fit_transaction"


def _install_validator_binding():
    current = PenalizedGeneralizedLinearModel._validate_inference_request
    if getattr(current, _MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self):
        value = current(self)
        contract = getattr(self, "_statgpu_pending_inference_contract", None)
        if contract is not None:
            self.inference_method = contract["dispatch"]
            self._inference_method = contract["dispatch"]
        return value

    setattr(wrapped, _MARKER, True)
    PenalizedGeneralizedLinearModel._validate_inference_request = wrapped


def _invalidate_failed_refit(self):
    """Clear result-bearing state after any failed inference-enabled refit."""
    try:
        self._clear_inference_state()
    except Exception:
        # Failure invalidation is best effort but must never replace the
        # original statistical/validation exception.
        pass
    for name, value in (
        ("coef_", None),
        ("intercept_", None),
        ("n_iter_", 0),
        ("_selected_solver", None),
        ("_selected_backend_name", None),
        ("_selected_backend_device", None),
        ("_native_fit_coef", None),
        ("_native_fit_intercept", None),
        ("_inference_precomputed", False),
        ("_precomputed_gaussian_state", None),
    ):
        try:
            setattr(self, name, value)
        except Exception:
            pass
    self._fitted = False


def _install_fit_restore():
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, _MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        public_request = getattr(self, "inference_method", None)
        internal_request = getattr(self, "_inference_method", None)
        try:
            return current(self, *args, **kwargs)
        except Exception:
            if bool(getattr(self, "compute_inference", False)):
                _invalidate_failed_refit(self)
            raise
        finally:
            if public_request is not None:
                self.inference_method = public_request
            if internal_request is not None:
                self._inference_method = internal_request
            else:
                self.__dict__.pop("_inference_method", None)

    setattr(wrapped, _MARKER, True)
    PenalizedGeneralizedLinearModel.fit = wrapped


def install_penalized_glm_inference_fit_transaction():
    _install_validator_binding()
    _install_fit_restore()


__all__ = ["install_penalized_glm_inference_fit_transaction"]
