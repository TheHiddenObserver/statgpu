"""Fit-transaction binding for the penalized-GLM inference resolver.

The public constructor parameter remains the user's request (for example
``auto``), while GPU sparse inference needs the concrete dispatch method during
``_fit_gpu`` / ``_fit_torch``.  Bind the resolved method immediately after the
pre-fit inference validator and restore the public request after the complete
fit transaction.

Kept separate from the method resolver so the transaction can be reviewed and
removed independently if the core fit later grows an explicit resolved-method
slot.
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
