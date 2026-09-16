"""Quantile continuation-path helpers shared by estimator and solver contracts.

Only estimator-generated Quantile continuation paths carry the marker defined
here.  Direct low-level callers that provide their own ``alpha_path`` remain
authoritative and are never rewritten by these helpers.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_numpy


class _AutoQuantileContinuationPath(np.ndarray):
    """Marker type for an estimator-generated Quantile continuation path."""


def mark_auto_quantile_continuation_path(alpha_path):
    """Mark an internally generated path without changing its numeric values."""
    return np.asarray(alpha_path, dtype=np.float64).view(_AutoQuantileContinuationPath)


def is_auto_quantile_continuation_path(alpha_path) -> bool:
    return isinstance(alpha_path, _AutoQuantileContinuationPath)


def _weighted_lower_quantile(y, sample_weight, tau: float) -> float:
    """Return a deterministic minimizer of the weighted pinball intercept loss."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    w_sorted = w[order]
    cutoff = float(tau) * float(np.sum(w_sorted, dtype=np.float64))
    index = int(np.searchsorted(np.cumsum(w_sorted, dtype=np.float64), cutoff, side="left"))
    index = min(max(index, 0), y_sorted.size - 1)
    return float(y_sorted[index])


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
):
    """Align an estimator-generated Quantile continuation start with its objective.

    The maintained weighted Quantile objective is normalized by ``sum(weights)``.
    For a non-uniform weighted fit, the continuation start therefore uses a
    weighted intercept-only Quantile solution and the corresponding normalized
    weighted pinball score.  Weight rescaling leaves the start unchanged.

    Equal weights with an intercept preserve the historical unweighted path
    bit-for-bit.  With ``fit_intercept=False`` the score is evaluated at the
    contractually fixed intercept zero.  A plain user-supplied ``alpha_path``
    is returned unchanged.
    """
    if not is_auto_quantile_continuation_path(alpha_path):
        return alpha_path
    if str(getattr(loss, "name", "")).lower() != "quantile":
        return alpha_path

    path_values = np.asarray(alpha_path, dtype=np.float64)
    if path_values.ndim != 1 or path_values.size == 0:
        return alpha_path

    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y), dtype=np.float64).reshape(-1)
    n = int(X_np.shape[0])
    if y_np.shape[0] != n:
        raise ValueError("Quantile continuation response length must match X rows")

    weights_np = None
    nonuniform_weight = False
    if sample_weight is not None:
        weights_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
        if weights_np.shape[0] != n:
            raise ValueError("sample_weight must have length n_samples")
        nonuniform_weight = not bool(np.allclose(weights_np, weights_np[0]))

    # Preserve the exact historical path when the objective is effectively
    # unweighted and an intercept is fitted.  This avoids changing ordinary
    # unweighted/constant-weight fits solely because weighted quantiles need a
    # deterministic empirical-CDF convention.
    if bool(fit_intercept) and not nonuniform_weight:
        return alpha_path

    tau = float(getattr(loss, "_tau", getattr(loss, "quantile", 0.5)))
    if bool(fit_intercept):
        intercept = _weighted_lower_quantile(y_np, weights_np, tau)
    else:
        intercept = 0.0

    residual = y_np - intercept
    psi = np.where(residual >= 0.0, tau, -(1.0 - tau))
    if weights_np is None:
        score = X_np.T @ psi / float(n)
    else:
        total_weight = float(np.sum(weights_np, dtype=np.float64))
        score = X_np.T @ (weights_np * psi) / total_weight

    lambda_start = float(np.max(np.abs(score))) if score.size else 0.0
    return _continuation_path_from_start(
        lambda_start,
        float(path_values[-1]),
        int(path_values.size),
    )


__all__ = [
    "mark_auto_quantile_continuation_path",
    "is_auto_quantile_continuation_path",
    "resolve_auto_quantile_continuation_path",
]
