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

The generic proximal-Newton inner loop is intentionally disabled for group
nonconvex LLA. Its Armijo condition is based on a smooth Newton direction plus a
post-hoc group proximal map; on valid Huber Group MCP/SCAD problems it can reject
all trial steps, restore the old iterate, and return without a failure status.
The group-aware fixed-step FISTA path uses the loss Lipschitz/step-scale contract
and the exact weighted Group Lasso proximal operator, so convergence is
observable through actual proximal updates rather than a silently stalled
Newton step.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from statgpu.penalties import AdaptiveGroupLassoPenalty
from ._fista_lla import fista_lla_path as _base_fista_lla_path


_GROUP_NONCONVEX_NAMES = frozenset(
    {"group_mcp", "gmcp", "group_scad", "gscad"}
)
_LLA_NONCONVEX_NAMES = frozenset({"scad", "mcp"}) | _GROUP_NONCONVEX_NAMES


class _GroupFISTALossProxy:
    """Delegate a loss while disabling the generic proximal-Newton branch."""

    has_hessian = False

    def __init__(self, loss):
        self._loss = loss

    def __getattr__(self, name):
        return getattr(self._loss, name)


class _QuantileWeightedStepScaleProxy:
    """Retain analytic weights across Quantile FISTA-LLA step refreshes.

    The fused engine supplies ``sample_weight`` on its initial step-scale call
    but omits it on periodic refreshes. Quantile's step scale follows the
    normalized weighted objective, so every public Quantile FISTA-LLA call must
    reuse the same backend-native weights. This proxy is Quantile-only; all
    other losses retain their previously validated behavior.
    """

    def __init__(self, loss):
        self._loss = loss
        self._sample_weight = None

    def __getattr__(self, name):
        return getattr(self._loss, name)

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        if sample_weight is not None:
            self._sample_weight = sample_weight
        effective_weight = (
            sample_weight if sample_weight is not None else self._sample_weight
        )
        return self._loss.lipschitz(
            X,
            coef,
            y=y,
            sample_weight=effective_weight,
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
    # The fused LLA solver appends one unpenalized intercept coordinate when
    # fit_intercept=True. Public group penalties remain exact-dimensional; only
    # this private surrogate opts into that one-coordinate extension.
    inner_penalty._allow_trailing_unpenalized_intercept = True

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
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be boolean")
    fit_intercept = bool(fit_intercept)

    if isinstance(max_lla_per_step, (bool, np.bool_)) or not isinstance(
        max_lla_per_step, Integral
    ):
        raise ValueError("max_lla_per_step must be a positive integer")
    max_lla_per_step = int(max_lla_per_step)
    if max_lla_per_step < 1:
        raise ValueError("max_lla_per_step must be a positive integer")

    for name, value in (("tol", tol), ("lla_tol", lla_tol)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite positive number")
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
        if name == "tol":
            tol = value
        else:
            lla_tol = value

    path_ndim = getattr(alpha_path, "ndim", 1)
    if path_ndim != 1:
        raise ValueError("alpha_path must be a non-empty one-dimensional sequence")
    try:
        path_len = len(alpha_path)
    except TypeError as exc:
        raise ValueError("alpha_path must be a non-empty one-dimensional sequence") from exc
    if path_len < 1:
        raise ValueError("alpha_path must be a non-empty one-dimensional sequence")
    path_values = []
    for value in alpha_path:
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise ValueError("alpha_path must contain finite positive numbers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "alpha_path must contain finite positive numbers"
            ) from exc
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("alpha_path must contain finite positive numbers")
        path_values.append(numeric)

    if any(
        path_values[i + 1] > path_values[i]
        for i in range(len(path_values) - 1)
    ):
        raise ValueError(
            "alpha_path must be non-increasing from continuation start to target"
        )

    if isinstance(max_iter, (list, tuple)):
        if len(max_iter) != path_len:
            raise ValueError(
                "max_iter sequence must have one positive integer per alpha_path step"
            )
        normalized_max_iter = []
        for value in max_iter:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(
                    "max_iter sequence must contain only positive integers"
                )
            value = int(value)
            if value < 1:
                raise ValueError(
                    "max_iter sequence must contain only positive integers"
                )
            normalized_max_iter.append(value)
        max_iter = normalized_max_iter
    else:
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral):
            raise ValueError("max_iter must be a positive integer or sequence")
        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be a positive integer or sequence")

    penalty_name = str(getattr(scad_penalty, "name", "") or "").lower().strip()
    if penalty_name not in _LLA_NONCONVEX_NAMES:
        raise ValueError(
            "fista_lla_path requires SCAD/MCP or Group SCAD/MCP penalty"
        )

    # Quantile's weighted step scale must remain objective-consistent for both
    # scalar and group penalties, including direct public low-level calls.
    if str(getattr(loss, "name", "")).lower() == "quantile":
        loss = _QuantileWeightedStepScaleProxy(loss)

    if penalty_name in _GROUP_NONCONVEX_NAMES:
        # Group-norm penalties require a group-norm convex surrogate whether
        # the caller supplied the historical factory or called this exported
        # solver directly without one.
        lla_penalty_factory = _group_surrogate_factory(scad_penalty)
        loss = _GroupFISTALossProxy(loss)

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
