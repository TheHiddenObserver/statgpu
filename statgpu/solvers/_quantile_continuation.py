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
    # Only scalar continuation metadata crosses to host. X/y/weights stay on
    # their declared backend/device throughout the score calculation.
    return float(np.asarray(_to_numpy(value)).reshape(()))


def _weighted_lower_quantile_backend(y, sample_weight, tau: float, xp):
    """Return a deterministic weighted pinball intercept minimizer on-device."""
    order = xp.argsort(y)
    y_sorted = y[order]
    w_sorted = sample_weight[order]
    cumulative = xp.cumsum(w_sorted)
    cutoff = float(tau) * xp.sum(w_sorted)
    if xp.__name__ == "torch":
        index = xp.searchsorted(cumulative, cutoff, right=False)
    else:
        index = xp.searchsorted(cumulative, cutoff, side="left")
    return y_sorted[int(index.item() if hasattr(index, "item") else index)]


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
    weighted pinball score. Weight rescaling leaves the start unchanged.

    Equal weights with an intercept preserve the historical unweighted path
    bit-for-bit. With ``fit_intercept=False`` the score is evaluated at the
    contractually fixed intercept zero. A plain user-supplied ``alpha_path``
    is returned unchanged.

    The calculation remains on the input NumPy/CuPy/Torch backend in float64,
    matching the maintained Quantile numerical paths. Only the final scalar
    ``lambda_start`` is synchronized to the host to build the small NumPy
    continuation vector consumed by the existing solver contract.
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

    # Preserve the exact historical path when the objective is effectively
    # unweighted and an intercept is fitted. This avoids changing ordinary
    # unweighted/constant-weight fits solely because weighted quantiles need a
    # deterministic empirical-CDF convention.
    if bool(fit_intercept) and not nonuniform_weight:
        return alpha_path

    tau = float(getattr(loss, "_tau", getattr(loss, "quantile", 0.5)))
    if bool(fit_intercept):
        intercept = _weighted_lower_quantile_backend(
            y_dev,
            weights_dev,
            tau,
            xp,
        )
    else:
        intercept = 0.0

    residual = y_dev - intercept
    pos = xp.full_like(residual, tau, dtype=float64)
    neg = xp.full_like(residual, -(1.0 - tau), dtype=float64)
    psi = xp.where(residual >= 0.0, pos, neg)
    if weights_dev is None:
        score = X_dev.T @ psi / float(n)
    else:
        total_weight = xp.sum(weights_dev)
        score = X_dev.T @ (weights_dev * psi) / total_weight

    lambda_start = _scalar_float(xp.max(xp.abs(score))) if int(score.size) else 0.0
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
