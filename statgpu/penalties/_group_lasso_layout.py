"""Canonical public Group Lasso penalty boundary.

Explicit nested group specifications may list members within a group in any
order. Group penalties are invariant to these within-group permutations, but
optimized solver paths rely on truthful contiguous-layout metadata. This module
keeps the historical public import and pickle path while ensuring that new
objects, legacy serialized state, and sklearn reconstruction all rebuild layout
metadata from canonical, sorted group members.
"""

from __future__ import annotations

import numpy as np

from . import _group_lasso as _group_lasso_impl


_BaseGroupLassoPenalty = _group_lasso_impl.GroupLassoPenalty
_BaseAdaptiveGroupLassoPenalty = _group_lasso_impl.AdaptiveGroupLassoPenalty


def _canonicalize_nested_groups(groups):
    """Sort indices within explicit groups while preserving group order."""
    if not isinstance(groups, (list, tuple)) or not groups:
        return groups
    first = groups[0]
    if not isinstance(first, (list, tuple, np.ndarray)):
        return groups
    return [np.sort(np.asarray(group, dtype=int)) for group in groups]


class GroupLassoPenalty(_BaseGroupLassoPenalty):
    """Group Lasso with canonical layout and clone-safe constructor state.

    ``groups`` is retained as the constructor parameter object so sklearn
    versions that reconstruct from ``get_params(deep=False)`` can satisfy their
    identity check. Internal group arrays are independently canonicalized.

    ``__setstate__`` intentionally rebuilds all derived layout metadata. This
    migrates objects serialized by versions that preserved unsorted nested
    groups or stale contiguity flags.
    """

    def __init__(self, alpha: float = 1.0, groups=None):
        self.groups = groups
        super().__init__(alpha=alpha, groups=groups)

    def _init_groups(self, groups):
        # Preserve the exact constructor object for sklearn <=1.2 clone while
        # using a canonical copy for numerical metadata and solver routing.
        self.groups = groups
        super()._init_groups(_canonicalize_nested_groups(groups))

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError("GroupLassoPenalty pickle state must be a dict")
        self.__dict__.update(state)
        groups = state.get("groups", state.get("_group_indices"))
        self.groups = groups
        if groups is not None:
            # Re-parse rather than trusting serialized derived fields such as
            # _is_contiguous, _flat_indices, padded indices, or device caches.
            self._init_groups(groups)

    def get_params(self, deep: bool = True) -> dict:
        """Return descriptive state or constructor-only clone parameters."""
        if not deep:
            return {"alpha": self.alpha, "groups": self.groups}
        # Preserve the historical descriptive serialization contract.
        return _BaseGroupLassoPenalty.get_params(self)


class AdaptiveGroupLassoPenalty(
    _BaseAdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
):
    """Adaptive Group Lasso preserving the public Group Lasso hierarchy."""

    def get_params(self, deep: bool = True) -> dict:
        """Return descriptive state or constructor-only clone parameters."""
        if not deep:
            return {
                "groups": self.groups,
                "alpha": self.alpha,
                "weights": self._group_weights,
            }
        return _BaseAdaptiveGroupLassoPenalty.get_params(self)


# Preserve historical import/pickle paths and ensure direct imports from
# ``statgpu.penalties._group_lasso`` resolve to the same public classes after
# package initialization. Rebinding both classes keeps
# ``issubclass(AdaptiveGroupLassoPenalty, GroupLassoPenalty)`` true.
GroupLassoPenalty.__module__ = _group_lasso_impl.__name__
AdaptiveGroupLassoPenalty.__module__ = _group_lasso_impl.__name__
_group_lasso_impl.GroupLassoPenalty = GroupLassoPenalty
_group_lasso_impl.AdaptiveGroupLassoPenalty = AdaptiveGroupLassoPenalty
