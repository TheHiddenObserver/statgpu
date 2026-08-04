"""Canonical public layout boundary for Group MCP and Group SCAD.

The original vectorized LLA implementations return repeated group derivatives
in grouped order for equal-size groups. That is correct only when the public
feature order is contiguous by group. For interleaved groups the returned
per-coordinate weights must be scattered through ``_flat_indices`` before the
LLA factory indexes them by the original feature indices.

This module also provides strict group/hyperparameter validation, immutable
constructor snapshots, sklearn-compatible shallow parameters, design-width
coverage, and legacy pickle migration matching the Group Lasso public boundary.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from . import _group_mcp as _group_mcp_impl
from . import _group_scad as _group_scad_impl
from ._group_lasso_layout import (
    _canonicalize_nested_groups,
    _normalize_groups_parameter,
    _sync_groups_snapshot_after_base_init,
    _validate_group_feature_coverage,
)


_BaseGroupMCPPenalty = _group_mcp_impl.GroupMCPPenalty
_BaseGroupSCADPenalty = _group_scad_impl.GroupSCADPenalty


def _finite_scalar(value, *, name):
    """Return a finite real scalar without accepting coercible strings/bools."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite numeric scalar")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


class _CanonicalGroupNonconvexLayout:
    """Shared canonical groups, clone, pickle, coverage, and LLA scatter."""

    def _init_groups(self, groups):
        normalized_groups = _normalize_groups_parameter(groups)
        self.groups = normalized_groups
        super()._init_groups(_canonicalize_nested_groups(normalized_groups))
        _sync_groups_snapshot_after_base_init(self, normalized_groups)

    def validate_n_features(self, n_features):
        return _validate_group_feature_coverage(self, n_features)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"{type(self).__name__} pickle state must be a dict")
        self.__dict__.update(state)
        self._validate_hyperparameters()
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
        alpha_value = _finite_scalar(alpha, name="alpha")
        gamma_value = _finite_scalar(gamma, name="gamma")
        if alpha_value <= 0.0:
            raise ValueError("alpha must be positive for Group MCP")
        if gamma_value <= 1.0:
            raise ValueError("gamma must be greater than 1 for Group MCP")
        self.groups = normalized_groups
        super().__init__(
            alpha=alpha_value,
            gamma=gamma_value,
            groups=normalized_groups,
        )

    def _validate_hyperparameters(self):
        self.alpha = _finite_scalar(self.alpha, name="alpha")
        self.gamma = _finite_scalar(self.gamma, name="gamma")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive for Group MCP")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1 for Group MCP")

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
        alpha_value = _finite_scalar(alpha, name="alpha")
        a_value = _finite_scalar(a, name="a")
        if alpha_value <= 0.0:
            raise ValueError("alpha must be positive for Group SCAD")
        if a_value <= 2.0:
            raise ValueError("a must be greater than 2 for Group SCAD")
        self.groups = normalized_groups
        super().__init__(
            alpha=alpha_value,
            a=a_value,
            groups=normalized_groups,
        )

    def _validate_hyperparameters(self):
        self.alpha = _finite_scalar(self.alpha, name="alpha")
        self.a = _finite_scalar(self.a, name="a")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive for Group SCAD")
        if self.a <= 2.0:
            raise ValueError("a must be greater than 2 for Group SCAD")

    def get_params(self, deep: bool = True) -> dict:
        if not deep:
            return {
                "alpha": self.alpha,
                "a": self.a,
                "groups": self.groups,
            }
        return _BaseGroupSCADPenalty.get_params(self)


GroupMCPPenalty.__module__ = _group_mcp_impl.__name__
GroupSCADPenalty.__module__ = _group_scad_impl.__name__
_group_mcp_impl.GroupMCPPenalty = GroupMCPPenalty
_group_scad_impl.GroupSCADPenalty = GroupSCADPenalty
