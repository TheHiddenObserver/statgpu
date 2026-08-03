"""Strict coefficient-dimension contracts for public group penalties.

Estimator intercepts are handled explicitly by ``SelectivePenalty``. Public
penalty methods therefore require an exact one-dimensional feature vector. The
only exception is an internal LLA surrogate created by the Group MCP/SCAD
contract: it receives one trailing unpenalized intercept from the fused solver
and opts in through a private capability flag. User-created penalties never get
that flag, so direct solver calls cannot silently leave trailing coordinates
unpenalized.
"""

from __future__ import annotations

from ._group_lasso_layout import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
)
from ._group_nonconvex_layout import GroupMCPPenalty, GroupSCADPenalty


def _validate_dimension(penalty, coef, operation):
    shape = getattr(coef, "shape", None)
    ndim = getattr(coef, "ndim", None)
    if shape is None or ndim != 1:
        raise ValueError(
            f"{type(penalty).__name__}.{operation} requires a "
            "one-dimensional coefficient vector"
        )
    feature_map = getattr(penalty, "_group_feat_idx", None)
    if feature_map is None:
        raise ValueError("groups must be set before numerical penalty use")
    expected = int(feature_map.shape[0])
    actual = int(shape[0])
    allow_internal_intercept = bool(
        getattr(penalty, "_allow_trailing_unpenalized_intercept", False)
    )
    valid = actual == expected or (
        allow_internal_intercept and actual == expected + 1
    )
    if not valid:
        suffix = (
            " or one internal trailing intercept"
            if allow_internal_intercept
            else ""
        )
        raise ValueError(
            f"{type(penalty).__name__}.{operation} expected {expected} "
            f"feature coefficients from groups{suffix}, got {actual}"
        )


def _wrap_method(cls, name):
    current = getattr(cls, name, None)
    if current is None or getattr(current, "_statgpu_group_dimension", False):
        return

    def wrapped(self, coef, *args, **kwargs):
        _validate_dimension(self, coef, name)
        return current(self, coef, *args, **kwargs)

    wrapped.__name__ = getattr(current, "__name__", name)
    wrapped.__doc__ = getattr(current, "__doc__", None)
    wrapped._statgpu_group_dimension = True
    wrapped._statgpu_original = current
    setattr(cls, name, wrapped)


def _install():
    for cls in (
        GroupLassoPenalty,
        AdaptiveGroupLassoPenalty,
        GroupMCPPenalty,
        GroupSCADPenalty,
    ):
        for method_name in ("value", "gradient", "proximal", "lla_weights"):
            _wrap_method(cls, method_name)


_install()
