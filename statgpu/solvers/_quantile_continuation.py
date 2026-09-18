"""Quantile continuation-path helpers shared by estimator and solver contracts.

Only estimator-generated Quantile continuation paths carry the marker defined
here. Direct low-level callers that provide their own ``alpha_path`` remain
authoritative and are never rewritten by these helpers.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._array_ops import _to_backend, _xp as _get_xp


class _AutoQuantileContinuationPath(np.ndarray):
    """Marker type for an estimator-generated Quantile continuation path."""


def mark_auto_quantile_continuation_path(alpha_path):
    """Mark an internally generated path without changing its numeric values."""
    return np.asarray(alpha_path, dtype=np.float64).view(_AutoQuantileContinuationPath)


def is_auto_quantile_continuation_path(alpha_path) -> bool:
    return isinstance(alpha_path, _AutoQuantileContinuationPath)


def _scalar_bool(value) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def _scalar_float(value) -> float:
    # Scalar control/continuation metadata may cross to host. Full X/y/weight
    # arrays remain on their declared backend/device throughout this resolver.
    return float(np.asarray(_to_numpy(value)).reshape(()))


def _weighted_lower_quantile_backend(y, sample_weight, tau: float, xp):
    """Return a deterministic weighted pinball intercept minimizer on-device."""
    order = xp.argsort(y)
    y_sorted = y[order]
    w_sorted = sample_weight[order]
    cumulative = (
        xp.cumsum(w_sorted, dim=0)
        if xp.__name__ == "torch"
        else xp.cumsum(w_sorted)
    )
    cutoff = float(tau) * xp.sum(w_sorted)
    if xp.__name__ == "torch":
        index = xp.searchsorted(cumulative, cutoff, right=False)
    else:
        index = xp.searchsorted(cumulative, cutoff, side="left")
    index_value = int(index.item() if hasattr(index, "item") else index)
    # Parallel/reassociated sum(w) can round a few ulps above the final
    # sequential cumsum value. For tau extremely close to one, searchsorted
    # may then return len(y) even though all validated weights are positive.
    # Clamp only that insertion-point overflow to the final observation.
    index_value = min(index_value, int(y_sorted.shape[0]) - 1)
    return y_sorted[index_value]


def quantile_penalty_alpha_start(score, penalty=None) -> float:
    """Map a Quantile slope score to the public penalty alpha scale."""
    xp = _get_xp(score)
    size = int(score.numel()) if hasattr(score, "numel") else int(score.size)
    if size == 0:
        return 0.0

    penalty_name = str(getattr(penalty, "name", "") or "").lower().strip()

    if penalty_name in {"adaptive_l1", "adaptive_lasso"}:
        weights = getattr(penalty, "_weights", None)
        if weights is not None:
            weights_np = np.asarray(_to_numpy(weights), dtype=np.float64).reshape(-1)
            if (
                weights_np.size == size
                and np.all(np.isfinite(weights_np))
                and np.all(weights_np > 0.0)
            ):
                score_np = np.asarray(_to_numpy(score), dtype=np.float64).reshape(-1)
                return float(np.max(np.abs(score_np) / weights_np))

    group_names = {
        "group_lasso", "gl", "adaptive_group_lasso",
        "group_scad", "gscad", "group_mcp", "gmcp"
    }
    groups = getattr(penalty, "_group_indices", None)
    if penalty_name in group_names and groups:
        adaptive_weights = (
            getattr(penalty, "_group_weights", None)
            if penalty_name == "adaptive_group_lasso"
            else None
        )
        if adaptive_weights is not None:
            adaptive_weights = np.asarray(
                _to_numpy(adaptive_weights), dtype=np.float64
            ).reshape(-1)
            if (
                adaptive_weights.size != len(groups)
                or not np.all(np.isfinite(adaptive_weights))
                or np.any(adaptive_weights <= 0.0)
            ):
                # A zero adaptive weight leaves a group unpenalized, so no
                # finite all-zero KKT threshold exists in general. Preserve
                # the historical heuristic rather than inventing infinity.
                return _scalar_float(xp.max(xp.abs(score)))

        thresholds = []
        for group_index, group in enumerate(groups):
            idx = np.asarray(group, dtype=np.int64).reshape(-1)
            if idx.size == 0:
                continue
            group_score = score[idx.tolist()]
            denominator = float(np.sqrt(idx.size))
            if adaptive_weights is not None:
                denominator *= float(adaptive_weights[group_index])
            thresholds.append(
                _scalar_float(xp.linalg.norm(group_score)) / denominator
            )
        return max(thresholds, default=0.0)

    return _scalar_float(xp.max(xp.abs(score)))


def _continuation_path_from_start(lambda_start, target_alpha, n_cont):
    lambda_start = float(lambda_start)
    target_alpha = float(target_alpha)
    n_cont = int(n_cont)
    alpha_start = max(lambda_start, target_alpha * 1.1)
    if (
        not np.isfinite(alpha_start)
        or alpha_start <= 0.0
        or target_alpha <= 0.0
    ):
        return np.linspace(max(lambda_start, 0.0), target_alpha, n_cont)
    return np.geomspace(alpha_start, target_alpha, n_cont)


def resolve_auto_quantile_continuation_path(
    loss,
    X,
    y,
    alpha_path,
    *,
    sample_weight=None,
    fit_intercept=True,
    penalty=None,
):
    """Align an estimator-generated Quantile continuation start with its objective.

    The maintained weighted Quantile objective is normalized by ``sum(weights)``.
    For a non-uniform weighted fit, the continuation start therefore uses a
    weighted intercept-only Quantile solution and the corresponding normalized
    weighted pinball score. Weight rescaling leaves the start unchanged.

    This is a ``lambda_max``-style continuation start rather than a claim that
    the chosen subgradient at zero residual is the unique exact KKT-minimal
    lambda. Its contract is objective consistency: the same analytic weights,
    intercept policy, and normalization used by the fitted Quantile objective
    also define the automatically generated continuation score.

    Equal weights with an intercept preserve the historical scalar-penalty
    path bit-for-bit. Group penalties use their public group-alpha scale and
    therefore recompute the continuation start even in the unweighted case.
    With ``fit_intercept=False`` the score is evaluated at the contractually
    fixed intercept zero. A plain user-supplied ``alpha_path`` is returned
    unchanged.

    The full design, response, and weight arrays remain on the input
    NumPy/CuPy/Torch backend in float64 while the weighted score is formed.
    Only scalar control decisions and scalar continuation metadata (including
    ``lambda_start``) may synchronize to the host; no full-array host snapshot
    is required for the non-uniform weighted recomputation.
    """
    if not is_auto_quantile_continuation_path(alpha_path):
        return alpha_path
    if str(getattr(loss, "name", "")).lower() != "quantile":
        return alpha_path

    path_values = np.asarray(alpha_path, dtype=np.float64)
    if path_values.ndim != 1 or path_values.size == 0:
        return alpha_path

    xp_hint = _get_xp(X)
    float64 = xp_hint.float64
    X_dev = _to_backend(X, backend="auto", ref_tensor=X, dtype=float64)
    y_dev = _to_backend(y, backend="auto", ref_tensor=X_dev, dtype=float64)
    if int(getattr(y_dev, "ndim", 1)) != 1:
        y_dev = y_dev.reshape(-1)
    xp = _get_xp(X_dev)
    n = int(X_dev.shape[0])
    if int(y_dev.shape[0]) != n:
        raise ValueError("Quantile continuation response length must match X rows")

    weights_dev = None
    nonuniform_weight = False
    if sample_weight is not None:
        weights_dev = _to_backend(
            sample_weight,
            backend="auto",
            ref_tensor=X_dev,
            dtype=float64,
        ).reshape(-1)
        if int(weights_dev.shape[0]) != n:
            raise ValueError("sample_weight must have length n_samples")
        # Equal analytic weights define the same objective as the unweighted
        # fit up to a common scale. Preserve that exact historical path, but do
        # not classify merely-near-equal weights as uniform: any genuine
        # observation-weight difference belongs in the weighted objective.
        nonuniform_weight = not _scalar_bool(xp.all(weights_dev == weights_dev[0]))

    penalty_name = str(getattr(penalty, "name", "") or "").lower().strip()
    group_scaled = penalty_name in {
        "group_lasso", "gl", "group_scad", "gscad", "group_mcp", "gmcp"
    }

    # Preserve the exact historical scalar path when the objective is
    # effectively unweighted and an intercept is fitted. Group penalties use
    # a different public alpha scale and therefore must recompute their start.
    if bool(fit_intercept) and not nonuniform_weight and not group_scaled:
        return alpha_path

    tau = float(getattr(loss, "_tau", getattr(loss, "quantile", 0.5)))
    if bool(fit_intercept):
        if nonuniform_weight:
            intercept = _weighted_lower_quantile_backend(
                y_dev,
                weights_dev,
                tau,
                xp,
            )
        else:
            intercept = xp.quantile(y_dev, tau)
    else:
        intercept = 0.0

    residual = y_dev - intercept
    pos = xp.full_like(residual, tau, dtype=float64)
    neg = xp.full_like(residual, -(1.0 - tau), dtype=float64)
    psi = xp.where(residual >= 0.0, pos, neg)
    if weights_dev is None or not nonuniform_weight:
        score = X_dev.T @ psi / float(n)
    else:
        total_weight = xp.sum(weights_dev)
        score = X_dev.T @ (weights_dev * psi) / total_weight

    lambda_start = quantile_penalty_alpha_start(score, penalty)
    return _continuation_path_from_start(
        lambda_start,
        float(path_values[-1]),
        int(path_values.size),
    )


__all__ = [
    "mark_auto_quantile_continuation_path",
    "is_auto_quantile_continuation_path",
    "resolve_auto_quantile_continuation_path",
    "quantile_penalty_alpha_start",
]
