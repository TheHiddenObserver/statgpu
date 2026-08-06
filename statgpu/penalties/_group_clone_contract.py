"""Constructor-only library clone contract for public group penalties.

``Penalty.clone()`` historically calls descriptive ``get_params()`` and cannot
reconstruct group penalties because that representation contains ``n_groups``
rather than constructor ``groups``. sklearn clone already requests
``deep=False``; the library clone is aligned with the same constructor-only
contract here.
"""

from __future__ import annotations

from ._group_lasso_layout import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
)
from ._group_nonconvex_layout import GroupMCPPenalty, GroupSCADPenalty


def _clone_from_constructor_params(self):
    return type(self)(**self.get_params(deep=False))


def _install():
    for cls in (
        GroupLassoPenalty,
        AdaptiveGroupLassoPenalty,
        GroupMCPPenalty,
        GroupSCADPenalty,
    ):
        cls.clone = _clone_from_constructor_params


_install()
