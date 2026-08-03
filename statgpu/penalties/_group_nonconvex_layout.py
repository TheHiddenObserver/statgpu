"""Canonical public layout boundary for Group MCP and Group SCAD.

The original vectorized LLA implementations return repeated group derivatives
in grouped order for equal-size groups. That is correct only when the public
feature order is contiguous by group. For interleaved groups the returned
per-coordinate weights must be scattered through ``_flat_indices`` before the
LLA factory indexes them by the original feature indices.

This module also provides immutable constructor snapshots, sklearn-compatible
shallow parameters, and legacy pickle migration matching the Group Lasso public
boundary.
"""

from __future__ import annotations

from . import _group_mcp as _group_mcp_impl
from . import _group_scad as _group_scad_impl
from ._group_lasso_layout import (
    _canonicalize_nested_groups,
    _normalize_groups_parameter,
)


_BaseGroupMCPPenalty = _group_mcp_impl.GroupMCPPenalty
_BaseGroupSCADPenalty = _group_scad_impl.GroupSCADPenalty


class _CanonicalGroupNonconvexLayout:
    """Shared canonical groups, clone, pickle, and LLA scatter behavior."""

    def _init_groups(self, groups):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super()._init_groups(_canonicalize_nested_groups(normalized_groups))

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"{type(self).__name__} pickle state must be a dict")
        self.__dict__.update(state)
        groups = state.get("groups", state.get("_group_indices"))
        self.groups = _normalize_groups_parameter(groups)
        if groups is not None:
            self._init_groups(groups)

    def _scatter_equal_noncontiguous_lla_weights(self, coef, grouped_weights):
        if not self._all_equal_size or self._is_contiguous:
            return grouped_weights
        xp = _group_mcp_impl._get_xp(coef)
        scattered = xp.zeros_like(grouped_weights)
        if xp.__name__ == "numpy":
            flat_indices = self._flat_indices
        else:
            flat_indices = self._get_flat_indices(xp, coef)
        scattered[flat_indices] = grouped_weights
        return scattered

    def lla_weights(self, coef):
        grouped_weights = super().lla_weights(coef)
        return self._scatter_equal_noncontiguous_lla_weights(
            coef, grouped_weights
        )


class GroupMCPPenalty(_CanonicalGroupNonconvexLayout, _BaseGroupMCPPenalty):
    """Group MCP with canonical non-contiguous LLA coordinate semantics."""

    def __init__(self, alpha: float = 1.0, gamma: float = 3.0, groups=None):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super().__init__(alpha=alpha, gamma=gamma, groups=normalized_groups)

    def get_params(self, deep: bool = True) -> dict:
        if not deep:
            return {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "groups": self.groups,
            }
        return _BaseGroupMCPPenalty.get_params(self)


class GroupSCADPenalty(_CanonicalGroupNonconvexLayout, _BaseGroupSCADPenalty):
    """Group SCAD with canonical non-contiguous LLA coordinate semantics."""

    def __init__(self, alpha: float = 1.0, a: float = 3.7, groups=None):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super().__init__(alpha=alpha, a=a, groups=normalized_groups)

    def get_params(self, deep: bool = True) -> dict:
        if not deep:
            return {
                "alpha": self.alpha,
                "a": self.a,
                "groups": self.groups,
            }
        return _BaseGroupSCADPenalty.get_params(self)


# Preserve historical public import and pickle globals.
GroupMCPPenalty.__module__ = _group_mcp_impl.__name__
GroupSCADPenalty.__module__ = _group_scad_impl.__name__
_group_mcp_impl.GroupMCPPenalty = GroupMCPPenalty
_group_scad_impl.GroupSCADPenalty = GroupSCADPenalty
