"""Group-penalty contract wrapper for the fused FISTA-LLA path.

The base fused solver expects an optional factory mapping the current penalty's
per-coordinate LLA derivatives to an inner convex penalty. Group MCP/SCAD
provide the derivative with respect to each group norm, repeated on the group's
original feature coordinates. The matching convex surrogate is

    sum_g D_g ||beta_g||_2.

``AdaptiveGroupLassoPenalty(alpha=1, weights=D_g/sqrt(p_g))`` represents this
surrogate exactly. The historical estimator caller instead took an L2 norm of
the repeated derivatives and used the target regularization strength again,
producing ``alpha_target * p_g * D_g``. Direct public solver calls without a
factory fell back to coordinate-wise Adaptive L1. Both paths optimize the wrong
surrogate and are normalized here.
"""

from __future__ import annotations

import numpy as np

from statgpu.penalties import AdaptiveGroupLassoPenalty
from ._fista_lla import fista_lla_path as _base_fista_lla_path


_GROUP_NONCONVEX_NAMES = frozenset(
    {"group_mcp", "gmcp", "group_scad", "gscad"}
)


def _group_surrogate_factory(scad_penalty):
    groups = getattr(scad_penalty, "_group_indices", None)
    if groups is None:
        raise ValueError("group penalty must define group indices for LLA")
    group_indices = [np.asarray(group, dtype=np.int64) for group in groups]
    group_sizes = np.asarray([len(group) for group in group_indices], dtype=float)
    if np.any(group_sizes <= 0):
        raise ValueError("group penalty contains an empty group")

    inner_penalty = AdaptiveGroupLassoPenalty(
        groups=group_indices,
        alpha=1.0,
        weights=np.ones(len(group_indices), dtype=float),
    )

    def factory(per_coordinate_derivatives):
        values = np.asarray(per_coordinate_derivatives, dtype=np.float64).ravel()
        group_weights = np.empty(len(group_indices), dtype=np.float64)
        for group_id, (indices, size) in enumerate(
            zip(group_indices, group_sizes)
        ):
            if indices.size == 0 or int(indices.max()) >= values.size:
                raise ValueError("LLA derivative vector is shorter than group indices")
            derivatives = values[indices]
            if derivatives.size != int(size):
                raise ValueError("LLA derivative vector is shorter than group indices")
            if not np.all(np.isfinite(derivatives)):
                raise FloatingPointError("group LLA derivatives must be finite")
            reference = float(derivatives[0])
            if not np.allclose(
                derivatives,
                reference,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "group LLA derivatives must be constant within each group"
                )
            if reference < -1e-12:
                raise ValueError("group LLA derivatives must be non-negative")
            group_weights[group_id] = max(reference, 0.0) / np.sqrt(size)
        inner_penalty.set_weights(group_weights)
        return inner_penalty

    return factory


def fista_lla_path(
    loss,
    scad_penalty,
    X,
    y,
    alpha_path,
    max_lla_per_step=6,
    lla_tol=1e-6,
    max_iter=1000,
    tol=1e-4,
    fit_intercept=True,
    sample_weight=None,
    lla_penalty_factory=None,
    init_coef=None,
    init_intercept=None,
    return_path=False,
):
    """Run the fused LLA path with exact Group MCP/SCAD surrogate scaling."""
    penalty_name = str(getattr(scad_penalty, "name", "")).lower()
    if penalty_name in _GROUP_NONCONVEX_NAMES:
        # Group-norm penalties require a group-norm convex surrogate whether
        # the caller supplied the historical factory or called this exported
        # solver directly without one.
        lla_penalty_factory = _group_surrogate_factory(scad_penalty)

    return _base_fista_lla_path(
        loss,
        scad_penalty,
        X,
        y,
        alpha_path,
        max_lla_per_step=max_lla_per_step,
        lla_tol=lla_tol,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        sample_weight=sample_weight,
        lla_penalty_factory=lla_penalty_factory,
        init_coef=init_coef,
        init_intercept=init_intercept,
        return_path=return_path,
    )
