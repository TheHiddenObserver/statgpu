"""Canonical public Group Lasso penalty boundary.

Nested public group specifications may list members within a group in any
order. Group penalties are invariant to these within-group permutations, but
the GPU block-coordinate solver has a fast path for equal-size contiguous
layouts. Canonicalizing each explicit group's member order keeps all downstream
layout metadata and fast-path checks truthful without changing the statistical
objective or the user-visible order of groups.
"""

from __future__ import annotations

import numpy as np

from . import _group_lasso as _group_lasso_impl


_BaseGroupLassoPenalty = _group_lasso_impl.GroupLassoPenalty


def _canonicalize_nested_groups(groups):
    """Sort indices within explicit groups while preserving group order."""
    if not isinstance(groups, (list, tuple)) or not groups:
        return groups
    first = groups[0]
    if not isinstance(first, (list, tuple, np.ndarray)):
        return groups
    return [np.sort(np.asarray(group, dtype=int)) for group in groups]


class GroupLassoPenalty(_BaseGroupLassoPenalty):
    """Group Lasso with canonical within-group index ordering."""

    def _init_groups(self, groups):
        super()._init_groups(_canonicalize_nested_groups(groups))


# Preserve the historical import/pickle path and ensure a direct import from
# ``statgpu.penalties._group_lasso`` observes the same public class after the
# package has initialized.
GroupLassoPenalty.__module__ = _group_lasso_impl.__name__
_group_lasso_impl.GroupLassoPenalty = GroupLassoPenalty
