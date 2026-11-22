"""Compatibility path for the former adaptive-group Lipschitz wrapper.

The shared solver utility now owns the unchanged nonsmooth-curvature rule.
Retain the old module-level callable for imports and serialized global paths,
without rebinding helpers already imported by FISTA consumers.
"""

from ._utils import (
    _smooth_penalty_lipschitz as _smooth_penalty_lipschitz_with_adaptive_group,
)
