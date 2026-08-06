"""Install the missing nonsmooth Lipschitz contract for Adaptive Group Lasso.

The generic FISTA utility treats purely nonsmooth penalties as contributing no
smooth Lipschitz curvature. ``AdaptiveGroupLassoPenalty`` is object-only and was
historically absent from the hand-written name list, causing ``alpha`` to be
added as fictitious smooth curvature and shrinking every FISTA step. Install
this narrow compatibility boundary before solver modules import the helper.
"""

from __future__ import annotations

from . import _utils


_current = _utils._smooth_penalty_lipschitz


if not getattr(_current, "_statgpu_adaptive_group_contract", False):

    def _smooth_penalty_lipschitz_with_adaptive_group(penalty):
        name = str(getattr(penalty, "name", "none")).lower().strip()
        if name == "adaptive_group_lasso":
            return 0.0
        return _current(penalty)

    _smooth_penalty_lipschitz_with_adaptive_group._statgpu_adaptive_group_contract = True
    _smooth_penalty_lipschitz_with_adaptive_group._statgpu_original = _current
    _utils._smooth_penalty_lipschitz = (
        _smooth_penalty_lipschitz_with_adaptive_group
    )
