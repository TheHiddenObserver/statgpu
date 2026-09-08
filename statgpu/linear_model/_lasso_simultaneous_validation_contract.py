"""Public validation parity for Lasso simultaneous inference controls.

The modern Lasso wrapper validates simultaneous method/inference compatibility,
but historically omitted the alpha and bootstrap-count guards present in the
legacy implementation.  PR #138 now has centered simultaneous NumPy/CuPy/Torch
numerical paths, so those public controls must fail identically before backend
dispatch rather than depending on backend-specific helper behavior.
"""

from __future__ import annotations

import functools
import math

from statgpu.linear_model.wrappers._lasso import Lasso


_CONSTRUCTOR_MARKER = "__statgpu_pr138_lasso_simultaneous_validation__"
_ORIGINAL_INIT = Lasso.__init__


@functools.wraps(_ORIGINAL_INIT)
def _init_with_simultaneous_validation(self, *args, **kwargs):
    result = _ORIGINAL_INIT(self, *args, **kwargs)
    if not bool(getattr(self, "enable_simultaneous_inference", False)):
        return result

    alpha = float(getattr(self, "simultaneous_alpha", 0.05))
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("simultaneous_alpha must be in (0, 1).")

    n_bootstrap = int(getattr(self, "simultaneous_n_bootstrap", 1000))
    if n_bootstrap <= 0:
        raise ValueError("simultaneous_n_bootstrap must be a positive integer.")
    return result


def install_lasso_simultaneous_validation_contract() -> None:
    current = Lasso.__init__
    if getattr(current, _CONSTRUCTOR_MARKER, False):
        return
    setattr(_init_with_simultaneous_validation, _CONSTRUCTOR_MARKER, True)
    Lasso.__init__ = _init_with_simultaneous_validation


__all__ = ["install_lasso_simultaneous_validation_contract"]
