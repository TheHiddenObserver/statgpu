"""Shared penalty category constants.

Single source of truth for penalty name sets used across solver and model layers.
Adding a new penalty type only requires updating this file.
"""

from __future__ import annotations

# Smooth penalties (differentiable, no proximal needed)
SMOOTH_PENALTIES = frozenset({"none", "null", "l2"})

# Non-smooth but convex penalties (need proximal operator)
NONSMOOTH_CONVEX = frozenset({
    "l1", "elasticnet", "en", "adaptive_l1", "adaptive_lasso",
    "group_lasso", "gl",
})

# Non-convex penalties (need LLA or specialized solver)
NONCONVEX = frozenset({
    "scad", "mcp", "group_mcp", "gmcp", "group_scad", "gscad",
})

# All non-smooth penalties (convex + non-convex)
NONSMOOTH = NONSMOOTH_CONVEX | NONCONVEX

# All sparse penalties (L1-type, produce sparse solutions)
SPARSE = frozenset({
    "l1", "elasticnet", "en", "adaptive_l1", "adaptive_lasso",
    "scad", "mcp",
})

# Group penalties
GROUP = frozenset({
    "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
})

# Penalties that disable BB step (use standard FISTA instead)
# Same as GROUP: BB step doesn't work well with group structure
BB_DISABLED = GROUP
