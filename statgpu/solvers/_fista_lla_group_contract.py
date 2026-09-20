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

from statgpu.backends import _to_numpy
from statgpu.backends._array_ops import _xp, _xp_asarray
from statgpu.penalties import AdaptiveGroupLassoPenalty
from ._fista_lla import fista_lla_path as _base_fista_lla_path
from ._utils import _validate_sample_weight


_GROUP_NONCONVEX_NAMES = frozenset(
    {"group_mcp", "gmcp", "group_scad", "gscad"}
)
_LLA_NONCONVEX_NAMES = frozenset({"scad", "mcp"}) | _GROUP_NONCONVEX_NAMES


def _validate_quantile_xy_shapes(X, y) -> int:
    """Validate Quantile FISTA-LLA inputs without host-copying GPU arrays."""
    from statgpu.glm_core._validation import _as_native_array, _require_real_finite

    X_values = _as_native_array(X, name="X")
    y_values = _as_native_array(y, name="y")
    if int(X_values.ndim) != 2:
        raise ValueError("X must be two-dimensional for Quantile fista_lla_path")
    if int(y_values.ndim) != 1:
        raise ValueError("y must be one-dimensional for Quantile fista_lla_path")
    n_samples = int(X_values.shape[0])
    if n_samples < 1:
        raise ValueError(
            "X must contain at least one observation for Quantile fista_lla_path"
        )
    if int(y_values.shape[0]) != n_samples:
        raise ValueError(
            "y must have the same number of observations as X for "
            "Quantile fista_lla_path"
        )
    _require_real_finite(X_values, name="X")
    _require_real_finite(y_values, name="y")
    return n_samples


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

    feature_count = max(int(indices.max()) for indices in group_indices) + 1
    representative_indices = np.asarray(
        [int(indices[0]) for indices in group_indices],
        dtype=np.int64,
    )
    feature_group = np.empty(feature_count, dtype=np.int64)
    for group_id, indices in enumerate(group_indices):
        feature_group[indices] = group_id
    sqrt_group_sizes = np.sqrt(group_sizes)

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
        xp = _xp(per_coordinate_derivatives)
        if xp.__name__ == "numpy":
            values = np.asarray(
                per_coordinate_derivatives, dtype=np.float64
            ).ravel()
            if values.size < feature_count:
                raise ValueError(
                    "LLA derivative vector is shorter than group indices"
                )
            values = values[:feature_count]
            references = values[representative_indices]
            reference_per_feature = references[feature_group]
            finite = bool(np.all(np.isfinite(values)))
            constant = bool(
                np.all(
                    np.abs(values - reference_per_feature)
                    <= 1e-12 + 1e-10 * np.abs(reference_per_feature)
                )
            )
            nonnegative = bool(np.all(references >= -1e-12))
            group_weights = np.maximum(references, 0.0) / sqrt_group_sizes
        else:
            values = per_coordinate_derivatives.reshape(-1)
            size = int(values.numel()) if hasattr(values, "numel") else int(values.size)
            if size < feature_count:
                raise ValueError(
                    "LLA derivative vector is shorter than group indices"
                )
            values = values[:feature_count]
            rep_idx = _xp_asarray(
                representative_indices, xp.int64, values
            )
            group_map = _xp_asarray(feature_group, xp.int64, values)
            sqrt_sizes = _xp_asarray(
                sqrt_group_sizes, values.dtype, values
            )
            references = values[rep_idx]
            reference_per_feature = references[group_map]
            finite = xp.all(xp.isfinite(values))
            constant = xp.all(
                xp.abs(values - reference_per_feature)
                <= 1e-12 + 1e-10 * xp.abs(reference_per_feature)
            )
            nonnegative = xp.all(references >= -1e-12)
            group_weights_native = xp.maximum(
                references,
                _xp_asarray(0.0, values.dtype, values),
            ) / sqrt_sizes
            status = xp.stack([finite, constant, nonnegative]).to(
                dtype=values.dtype
            ) if xp.__name__ == "torch" else xp.stack(
                [finite, constant, nonnegative]
            ).astype(values.dtype, copy=False)
            payload = xp.concatenate([group_weights_native, status])
            payload_np = np.asarray(
                _to_numpy(payload), dtype=np.float64
            ).reshape(-1)
            group_weights = payload_np[:-3]
            finite, constant, nonnegative = (
                bool(payload_np[-3] != 0.0),
                bool(payload_np[-2] != 0.0),
                bool(payload_np[-1] != 0.0),
            )

        if not finite:
            raise FloatingPointError("group LLA derivatives must be finite")
        if not constant:
            raise ValueError(
                "group LLA derivatives must be constant within each group"
            )
        if not nonnegative:
            raise ValueError("group LLA derivatives must be non-negative")

        inner_penalty.set_weights(group_weights)
        return inner_penalty

    # The fused engine otherwise preserves the historical custom-factory
    # contract by supplying NumPy derivatives. This internal factory can reduce
    # device-native derivatives to a G+3 reporting payload before host transfer.
    factory._statgpu_accepts_native_derivatives = True
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

    if not isinstance(return_path, (bool, np.bool_)):
        raise ValueError("return_path must be boolean")
    return_path = bool(return_path)

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

    if isinstance(alpha_path, (list, tuple)):
        path_ndim = np.asarray(alpha_path, dtype=object).ndim
    else:
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

    loss_name = str(getattr(loss, "name", "") or "").lower().strip()
    n_samples = (
        _validate_quantile_xy_shapes(X, y)
        if loss_name == "quantile"
        else len(X)
    )

    if loss_name == "quantile":
        from statgpu.glm_core._validation import (
            _as_native_array,
            _is_boolean_array,
            _require_real_finite,
        )

        X_values = _as_native_array(X, name="X")
        n_features = int(X_values.shape[1])
        if init_coef is not None:
            init_values = _as_native_array(init_coef, name="init_coef")
            if int(init_values.ndim) != 1:
                raise ValueError("init_coef must be one-dimensional")
            if int(init_values.shape[0]) != n_features:
                raise ValueError("init_coef must have length n_features")
            if _is_boolean_array(init_values):
                raise ValueError("init_coef must contain real numeric values")
            _require_real_finite(init_values, name="init_coef")

        if init_intercept is not None:
            intercept_value = _as_native_array(
                init_intercept,
                name="init_intercept",
            )
            if int(intercept_value.ndim) != 0:
                raise ValueError("init_intercept must be a scalar")
            if _is_boolean_array(intercept_value):
                raise ValueError("init_intercept must be a finite real scalar")
            _require_real_finite(intercept_value, name="init_intercept")

    _validate_sample_weight(sample_weight, n_samples)

    penalty_name = str(getattr(scad_penalty, "name", "") or "").lower().strip()
    if penalty_name not in _LLA_NONCONVEX_NAMES:
        raise ValueError(
            "fista_lla_path requires SCAD/MCP or Group SCAD/MCP penalty"
        )

    if penalty_name in _GROUP_NONCONVEX_NAMES:
        validate_n_features = getattr(scad_penalty, "validate_n_features", None)
        if callable(validate_n_features):
            group_n_features = (
                n_features
                if loss_name == "quantile"
                else int(getattr(X, "shape")[1])
            )
            validate_n_features(group_n_features)

    if loss_name == "quantile":
        if "scad" in penalty_name:
            shape = getattr(scad_penalty, "a", None)
            if (
                isinstance(shape, (bool, np.bool_))
                or not isinstance(shape, Real)
                or not np.isfinite(float(shape))
                or float(shape) <= 2.0
            ):
                raise ValueError(
                    "SCAD penalty a must be a finite real number greater than 2"
                )
        else:
            shape = getattr(scad_penalty, "gamma", None)
            if (
                isinstance(shape, (bool, np.bool_))
                or not isinstance(shape, Real)
                or not np.isfinite(float(shape))
                or float(shape) <= 1.0
            ):
                raise ValueError(
                    "MCP penalty gamma must be a finite real number greater than 1"
                )

    # Quantile's weighted step scale must remain objective-consistent for both
    # scalar and group penalties, including direct public low-level calls.
    if loss_name == "quantile":
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
