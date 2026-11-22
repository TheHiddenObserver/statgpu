"""Historical imports for canonical penalized Cox validation and introspection.

Validation now belongs to ``_penalized_cox_cv`` and the public class owns its
actual docstring. Importing or reloading this compatibility module never mutates
either owner.
"""

from __future__ import annotations

from statgpu.cross_validation._grid_validation import coerce_real_numeric_grid

from . import _penalized_cox_cv as _cv_module
from ._penalized_cox import PenalizedCoxPHModel
from ._penalized_cox_cv import (
    _validate_alpha_grid as _validate_alpha_grid_strict,
    _validate_alpha_grid_unchecked as _ORIGINAL_VALIDATE_ALPHA_GRID,
)


__all__ = ["_validate_alpha_grid_strict"]
