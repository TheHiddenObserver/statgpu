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

    The estimator minimizes the negative right-censored Cox partial likelihood
    plus a validated L1, L2/Ridge, ElasticNet, SCAD, MCP, or null penalty. The
    Cox partial likelihood has no identifiable intercept, so
    ``fit_intercept=True`` is rejected. Breslow and Efron ties are supported on
    NumPy, CuPy, and Torch CUDA backends.

    Penalized Cox inference is currently estimation-only:
    ``compute_inference=True`` raises ``NotImplementedError``. Use
    :class:`statgpu.survival.CoxPH` for unpenalized Cox inference.
    """


__all__ = ["_validate_alpha_grid_strict"]
