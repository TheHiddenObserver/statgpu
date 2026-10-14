"""Public compatibility boundaries for penalized Cox estimators and CV."""

from __future__ import annotations

from functools import wraps

from statgpu.cross_validation._grid_validation import coerce_real_numeric_grid

from . import _penalized_cox_cv as _cv_module
from ._penalized_cox import PenalizedCoxPHModel


_ORIGINAL_VALIDATE_ALPHA_GRID = _cv_module._validate_alpha_grid


@wraps(_ORIGINAL_VALIDATE_ALPHA_GRID)
def _validate_alpha_grid_strict(alpha_grid, penalty_name):
    """Reject lossy scalar coercion before Cox CV sign validation."""
    grid = coerce_real_numeric_grid(alpha_grid, name="alpha_grid")
    return _ORIGINAL_VALIDATE_ALPHA_GRID(grid, penalty_name)


_cv_module._validate_alpha_grid = _validate_alpha_grid_strict

# The historical class body placed a support-name constant before its long
# string literal, so Python did not recognize that literal as ``__doc__``.
# Restore public introspection without changing constructor or fitted behavior.
if not PenalizedCoxPHModel.__doc__:
    PenalizedCoxPHModel.__doc__ = """Penalized Cox proportional hazards model.

    The estimator minimizes -ell(coef) / n plus a validated L1, L2/Ridge,
    ElasticNet, SCAD, MCP, or null penalty. Here ell is the summed
    right-censored Cox partial log likelihood and n counts all training rows,
    including censored observations, rather than just events. With the built-in
    string penalty='l2', the penalty is alpha * ||coef||**2 / 2. On the same
    data and tie method, its objective matches CoxPH(penalty=n * alpha / 2).
    This direct-fit conversion does not equate the two CV searches, whose
    training sizes and held-out score normalizations differ. The
    Cox partial likelihood has no identifiable intercept, so
    ``fit_intercept=True`` is rejected. Breslow and Efron ties are supported on
    NumPy, CuPy, and Torch CUDA backends.

    Penalized Cox inference is currently estimation-only:
    ``compute_inference=True`` raises ``NotImplementedError``. Use
    :class:`statgpu.survival.CoxPH` for unpenalized or fixed-L2 Cox inference,
    aligning its penalty normalization with the desired objective.
    """


__all__ = ["_validate_alpha_grid_strict"]
