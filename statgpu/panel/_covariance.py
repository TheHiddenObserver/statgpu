"""Covariance estimators and internal dispatch for panel data models.

Stage C of Issue #93 extends the Stage-A residual-OLS covariance registry with
HC0/HC2/HC3, explicit cluster group debiasing, and Driscoll-Kraay covariance.
Covariance breads are built from the design pseudoinverse rather than normal
equations so full-rank but ill-conditioned designs do not square the condition
number before inference.
"""
from __future__ import annotations

__all__ = [
    "clustered_covariance",
    "two_way_clustered_covariance",
    "hac_covariance",
    "driscoll_kraay_covariance",
    "normalize_covariance_type",
    "ols_covariance",
]

from typing import Optional

import numpy as np

from statgpu.backends import (
    _get_xp,
    _resolve_backend,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_zeros,
)
from statgpu.panel._linalg import (
    panel_svd_pseudoinverse,
    panel_svd_working_pseudoinverse,
)
from statgpu.panel._utils import factorize_panel_metadata


_COVARIANCE_ALIASES = {
    "hc1": "robust",
    "dk": "driscoll-kraay",
    "kernel": "driscoll-kraay",
}

_KERNEL_ALIASES = {
    "bartlett": "bartlett",
    "newey-west": "bartlett",
    "parzen": "parzen",
    "gallant": "parzen",
    "quadratic-spectral": "qs",
    "qs": "qs",
    "andrews": "qs",
}


def normalize_covariance_type(cov_type: str) -> str:
    """Return the canonical Stage-C covariance name."""
    name = str(cov_type).strip().lower()
    return _COVARIANCE_ALIASES.get(name, name)


def _ensure_xp(xp=None, *arrays):
    """Return an explicit array module or infer it from public inputs."""
    if xp is not None:
        return xp
    return _get_xp(_resolve_backend("auto", *arrays))


def _is_torch(xp) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def _design_pseudoinverse(X, xp):
    """Compatibility wrapper around the shared panel SVD policy."""
    return panel_svd_pseudoinverse(X, xp)


def _gram_inverse(X, xp, *, rank_aware: bool = False):
    """Compatibility helper returning a stable generalized inverse of X'X.

    ``rank_aware`` is retained for the existing internal signature.  The stable
    implementation always derives the bread from ``X+``; it never attempts
    ``inv(X'X)`` merely because ``X`` is still classified as full rank.
    """
    del rank_aware
    _X_pinv, bread, rank = _design_pseudoinverse(X, xp)
    return bread, rank


def _influence_rows(X, resid, xp):
    """Return stable observation influence rows plus working projection factors."""
    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(
        X, xp
    )
    # X+ = design_scale * X_work+.  Multiply residuals before restoring the
    # design scale so a tiny full-rank design does not overflow X+ even when the
    # final influence and covariance remain representable.
    influence = (X_pinv_work.T * resid[:, None]) * design_scale
    return influence, X_pinv_work, X_work, rank


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _grouped_score_sums(scores, codes_np, *, n_groups: int, xp):
    """Sum an observation-by-parameter score matrix by integer group code."""
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    if codes_np.shape[0] != int(scores.shape[0]):
        raise ValueError("group codes must match the number of score rows")
    if int(n_groups) <= 0:
        raise ValueError("at least one group is required")
    codes = xp_asarray(
        codes_np,
        dtype=xp.int64,
        xp=xp,
        ref_arr=scores,
    )
    out = xp_zeros(
        (int(n_groups), int(scores.shape[1])),
        dtype=xp.float64,
        xp=xp,
        ref_arr=scores,
    )
    if hasattr(out, "scatter_add_"):
        out.scatter_add_(0, codes.unsqueeze(1).expand_as(scores), scores)
    elif type(out).__module__.startswith("cupy"):
        xp.add.at(out, codes, scores)
    else:
        np.add.at(out, codes_np, scores)
    return out


def _factorize_1d_labels(values, *, nobs: int, name: str):
    labels, codes = factorize_panel_metadata(
        values, name=name, expected_n=int(nobs)
    )
    return labels, codes


def _paired_codes(left, right):
    pairs = np.column_stack(
        [np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)]
    )
    _, codes = np.unique(pairs, axis=0, return_inverse=True)
    return codes.astype(np.int64, copy=False)


def _group_debias_factor(n_groups: int, nobs: int) -> float:
    n_groups = int(n_groups)
    nobs = int(nobs)
    if n_groups < 2:
        raise ValueError("group_debias requires at least two groups")
    if nobs <= 0:
        raise ValueError("group_debias requires a positive observation count")
    return (n_groups / (n_groups - 1.0)) * ((nobs - 1.0) / nobs)


def _validate_group_debias(value) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("group_debias must be boolean")
    return bool(value)


def clustered_covariance(
    X,
    resid,
    clusters,
    xp=None,
    *,
    group_debias: bool = False,
    metadata: Optional[dict] = None,
):
    """One-way clustered robust covariance matrix."""
    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    n, _k = X.shape
    labels, cluster_idx = _factorize_1d_labels(
        clusters, nobs=int(n), name="clusters"
    )
    if resid.shape[0] != n:
        raise ValueError("X and resid must have the same number of observations")

    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    n_clusters = int(len(labels))
    if n_clusters < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(
        influence, cluster_idx, n_groups=n_clusters, xp=xp
    )
    correction = 1.0
    if group_debias:
        correction = _group_debias_factor(n_clusters, int(n))
    cov = _symmetrize(grouped.T @ grouped * float(correction))
    if metadata is not None:
        metadata.update(
            {
                "cluster_dimensions": 1,
                "cluster_group_counts": [n_clusters],
                "group_debias": bool(group_debias),
                "group_debias_factors": [float(correction)],
            }
        )
    return cov


def two_way_clustered_covariance(
    X,
    resid,
    cluster1,
    cluster2,
    xp=None,
    *,
    group_debias: bool = False,
    metadata: Optional[dict] = None,
):
    """Two-way clustered covariance with exact intersection factorization."""
    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)
    n = int(X.shape[0])
    labels1, c1 = _factorize_1d_labels(cluster1, nobs=n, name="cluster1")
    labels2, c2 = _factorize_1d_labels(cluster2, nobs=n, name="cluster2")
    c12 = _paired_codes(c1, c2)
    n12 = int(np.max(c12)) + 1 if c12.size else 0

    meta1: dict = {}
    meta2: dict = {}
    meta12: dict = {}
    V1 = clustered_covariance(
        X,
        resid,
        c1,
        xp,
        group_debias=group_debias,
        metadata=meta1,
    )
    V2 = clustered_covariance(
        X,
        resid,
        c2,
        xp,
        group_debias=group_debias,
        metadata=meta2,
    )
    V12 = clustered_covariance(
        X,
        resid,
        c12,
        xp,
        group_debias=group_debias,
        metadata=meta12,
    )
    if metadata is not None:
        metadata.update(
            {
                "cluster_dimensions": 2,
                "cluster_group_counts": [
                    int(len(labels1)),
                    int(len(labels2)),
                    n12,
                ],
                "group_debias": bool(group_debias),
                "group_debias_factors": [
                    float(meta1["group_debias_factors"][0]),
                    float(meta2["group_debias_factors"][0]),
                    float(meta12["group_debias_factors"][0]),
                ],
            }
        )
    return _symmetrize(V1 + V2 - V12)


def hac_covariance(X, resid, bandwidth=None, kernel="bartlett", xp=None):
    """Historical row-order Newey-West HAC covariance using Bartlett weights."""
    xp = _ensure_xp(xp, X)
    if str(kernel).lower() != "bartlett":
        raise ValueError("kernel must be 'bartlett'")
    if bandwidth is not None:
        if isinstance(bandwidth, bool) or not isinstance(
            bandwidth, (int, np.integer)
        ):
            raise ValueError("bandwidth must be a non-negative integer or None")
        if int(bandwidth) < 0:
            raise ValueError("bandwidth must be a non-negative integer or None")

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])

    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(int(bandwidth), n - 1))

    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    cov = influence.T @ influence
    for h in range(1, bandwidth + 1):
        w = 1.0 - h / (bandwidth + 1.0)
        gamma_h = influence[h:].T @ influence[: n - h]
        cov = cov + w * (gamma_h + gamma_h.T)
    return _symmetrize(cov)


def _canonical_kernel(kernel: str) -> str:
    name = str(kernel).strip().lower()
    if name not in _KERNEL_ALIASES:
        choices = ", ".join(sorted(_KERNEL_ALIASES))
        raise ValueError(
            f"unsupported Driscoll-Kraay kernel {kernel!r}; expected one of: {choices}"
        )
    return _KERNEL_ALIASES[name]


def _validate_dk_bandwidth(bandwidth, n_periods: int) -> int:
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))
    if isinstance(bandwidth, bool) or not isinstance(
        bandwidth, (int, np.integer)
    ):
        raise ValueError(
            "Driscoll-Kraay bandwidth must be a non-negative integer or None"
        )
    bandwidth = int(bandwidth)
    if bandwidth < 0:
        raise ValueError(
            "Driscoll-Kraay bandwidth must be a non-negative integer or None"
        )
    # Explicit oversized bandwidths are smoothing parameters, not silently
    # capped lags. Only observed lags can contribute to the covariance.
    return bandwidth


def _dk_kernel_weights(
    kernel: str, bandwidth: int, max_lag: int
) -> tuple[str, np.ndarray]:
    """Return canonical DK kernel name and weights for lags 0..max_lag."""
    canonical = _canonical_kernel(kernel)
    max_lag = int(max_lag)
    bandwidth = int(bandwidth)
    weights = np.zeros(max_lag + 1, dtype=np.float64)
    weights[0] = 1.0
    if max_lag == 0 or bandwidth == 0:
        return canonical, weights

    if canonical == "bartlett":
        stop = min(bandwidth, max_lag)
        lag = np.arange(1, stop + 1, dtype=np.float64)
        weights[1 : stop + 1] = 1.0 - lag / (bandwidth + 1.0)
        return canonical, weights

    if canonical == "parzen":
        stop = min(bandwidth, max_lag)
        lag = np.arange(1, stop + 1, dtype=np.float64)
        z = lag / (bandwidth + 1.0)
        low = z <= 0.5
        w = np.empty_like(z)
        w[low] = 1.0 - 6.0 * z[low] ** 2 + 6.0 * z[low] ** 3
        w[~low] = 2.0 * (1.0 - z[~low]) ** 3
        weights[1 : stop + 1] = w
        return canonical, weights

    # Quadratic spectral is not truncated at bandwidth.  Its direct expression
    # catastrophically cancels for x -> 0, so use the analytic series there:
    # 3/x^2 (sin(x)/x - cos(x))
    #   = 1 - x^2/10 + x^4/280 - x^6/15120 + O(x^8).
    lag = np.arange(1, max_lag + 1, dtype=np.float64)
    x = 6.0 * np.pi * lag / (5.0 * bandwidth)
    small = np.abs(x) < 1.0e-3
    w = np.empty_like(x)
    x2 = x[small] * x[small]
    w[small] = 1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2 * x2 * x2 / 15120.0
    regular = ~small
    xr = x[regular]
    w[regular] = 3.0 / (xr * xr) * (np.sin(xr) / xr - np.cos(xr))
    weights[1:] = w
    return canonical, weights


def driscoll_kraay_covariance(
    X,
    resid,
    time_ids,
    *,
    bandwidth=None,
    kernel="bartlett",
    extra_df: int = 0,
    xp=None,
    metadata: Optional[dict] = None,
):
    """Driscoll-Kraay covariance on fit-space influence scores."""
    xp = _ensure_xp(xp, X)
    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])
    labels, time_codes = _factorize_1d_labels(
        time_ids, nobs=n, name="time_ids"
    )
    n_periods = int(len(labels))
    if n_periods < 2:
        raise ValueError(
            "Driscoll-Kraay covariance requires at least two time periods"
        )

    if isinstance(extra_df, bool) or not isinstance(extra_df, (int, np.integer)):
        raise ValueError("extra_df must be a non-negative integer")
    extra_df = int(extra_df)
    if extra_df < 0:
        raise ValueError("extra_df must be a non-negative integer")

    influence, _X_pinv, _bread, rank = _influence_rows(X, resid, xp)
    k_columns = int(X.shape[1])
    rank_deficient = rank < k_columns
    df_model = rank if rank_deficient else k_columns
    denom = n - extra_df - df_model
    if denom <= 0:
        raise ValueError(
            "Driscoll-Kraay covariance requires positive debiased residual degrees of freedom"
        )

    grouped = _grouped_score_sums(
        influence, time_codes, n_groups=n_periods, xp=xp
    )
    bw = _validate_dk_bandwidth(bandwidth, n_periods)
    canonical_kernel, weights_np = _dk_kernel_weights(
        kernel, bw, n_periods - 1
    )
    weights = xp_asarray(
        weights_np,
        dtype=xp.float64,
        xp=xp,
        ref_arr=grouped,
    )

    cov = grouped.T @ grouped
    for lag in range(1, n_periods):
        if weights_np[lag] == 0.0:
            continue
        gamma = grouped[lag:].T @ grouped[: n_periods - lag]
        cov = cov + weights[lag] * (gamma + gamma.T)

    scale = float(n) / float(denom)
    cov = _symmetrize(scale * cov)
    if metadata is not None:
        nonzero_lags = np.flatnonzero(np.abs(weights_np[1:]) > 0.0) + 1
        metadata.update(
            {
                "covariance": "driscoll-kraay",
                "kernel": canonical_kernel,
                "bandwidth": int(bw),
                "n_periods": n_periods,
                "max_weighted_lag": int(nonzero_lags.max())
                if nonzero_lags.size
                else 0,
                "all_observed_lags_weighted": bool(
                    canonical_kernel == "qs" and bw > 0
                ),
                "extra_df": int(extra_df),
                "design_rank": int(rank),
                "design_columns": int(k_columns),
                "rank_deficient_extension": bool(rank_deficient),
                "df_scale": float(scale),
            }
        )
    return cov


def _hc_covariance(
    X, resid, *, kind: str, xp, metadata: Optional[dict] = None
):
    influence, X_pinv_work, X_work, rank = _influence_rows(X, resid, xp)
    if kind == "hc0":
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "hc0",
                    "design_rank": int(rank),
                    "design_columns": int(X.shape[1]),
                }
            )
        return _symmetrize(influence.T @ influence)

    projection_rows = X_pinv_work.T
    # Leverage is invariant to the uniform working-design rescaling:
    # X X+ = X_work X_work+.  Evaluate it entirely at the safe working scale.
    if _is_torch(xp):
        leverage = xp.sum(X_work * projection_rows, dim=1)
    else:
        leverage = xp.sum(X_work * projection_rows, axis=1)
    leverage_min = _to_float_scalar(xp.min(leverage))
    leverage_max = _to_float_scalar(xp.max(leverage))
    tol = 4096.0 * np.finfo(np.float64).eps
    if leverage_min < -tol:
        raise ValueError("HC2/HC3 leverage is materially negative")
    if leverage_max > 1.0 + tol:
        raise ValueError("HC2/HC3 leverage is materially greater than one")
    if _is_torch(xp):
        leverage = xp.clamp(leverage, min=0.0, max=1.0)
    else:
        leverage = xp.clip(leverage, 0.0, 1.0)
    denominator = 1.0 - leverage
    denominator_min = _to_float_scalar(xp.min(denominator))
    if denominator_min <= tol:
        raise ValueError(
            "HC2/HC3 covariance is undefined when leverage is numerically one"
        )
    if kind == "hc2":
        adjusted_influence = influence / xp.sqrt(denominator)[:, None]
    elif kind == "hc3":
        adjusted_influence = influence / denominator[:, None]
    else:
        raise ValueError(f"unknown HC covariance kind {kind!r}")
    if metadata is not None:
        metadata.update(
            {
                "covariance": kind,
                "design_rank": int(rank),
                "design_columns": int(X.shape[1]),
                "leverage_min": float(leverage_min),
                "leverage_max": float(leverage_max),
            }
        )
    return _symmetrize(adjusted_influence.T @ adjusted_influence)


def ols_covariance(
    X,
    resid,
    *,
    cov_type,
    scale=None,
    df_resid=None,
    cluster=None,
    time_ids=None,
    bandwidth=None,
    kernel="bartlett",
    group_debias: bool = False,
    extra_df: int = 0,
    xp=None,
    allowed=None,
    hc1_correction=None,
    metadata: Optional[dict] = None,
):
    """Dispatch residual-based panel covariance definitions."""
    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)
    name = normalize_covariance_type(cov_type)
    if allowed is not None:
        allowed_names = {normalize_covariance_type(value) for value in allowed}
        if name not in allowed_names:
            choices = ", ".join(sorted(str(value) for value in allowed_names))
            raise ValueError(
                f"cov_type={cov_type!r} is not supported here; expected one of: {choices}"
            )

    if group_debias and name != "clustered":
        raise ValueError("group_debias=True requires cov_type='clustered'")

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])

    if metadata is not None:
        metadata.clear()
        metadata["covariance"] = name

    if name == "nonrobust":
        if scale is None:
            raise ValueError("scale is required for nonrobust covariance")
        scale_value = float(scale)
        if np.isnan(scale_value) or scale_value < 0.0:
            raise ValueError("scale must be non-negative and not NaN")
        _X_work, X_pinv_work, design_scale, _rank = (
            panel_svd_working_pseudoinverse(X, xp)
        )

        suspicious_scale = (
            not np.isfinite(scale_value)
            or (0.0 < scale_value < np.finfo(np.float64).tiny)
            or (scale_value == 0.0 and _to_float_scalar(xp.max(xp.abs(resid))) > 0.0)
        )
        if suspicious_scale:
            if df_resid is None or int(df_resid) <= 0:
                raise ValueError(
                    "positive df_resid is required when nonrobust scale must be "
                    "reconstructed from residuals"
                )
            resid_scale = xp.max(xp.abs(resid))
            resid_scale_value = _to_float_scalar(resid_scale)
            if resid_scale_value == 0.0:
                scaled_pinv = X_pinv_work * 0.0
            else:
                resid_unit = resid / resid_scale
                norm_sq = _to_float_scalar(xp.sum(resid_unit * resid_unit))
                rms_unit = float(np.sqrt(norm_sq / float(int(df_resid))))
                # sqrt(scale) = resid_scale * rms_unit.  Multiply the tiny/large
                # residual scale into the working pseudoinverse before restoring
                # design_scale; this lets opposite scales cancel while every
                # intermediate remains representable whenever the final covariance is.
                scaled_pinv = (
                    ((X_pinv_work * resid_scale) * design_scale) * rms_unit
                )
            if metadata is not None:
                metadata["nonrobust_scale_reconstructed"] = True
                metadata["residual_scale"] = float(resid_scale_value)
        else:
            # scale * X+ X+^T = A A^T with
            # A = sqrt(scale) * design_scale * X_work+.
            scaled_pinv = (
                X_pinv_work * float(np.sqrt(scale_value))
            ) * design_scale
        return _symmetrize(scaled_pinv @ scaled_pinv.T)

    if name == "robust":
        influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
        correction = hc1_correction
        if correction is None:
            if df_resid is None or int(df_resid) <= 0:
                raise ValueError(
                    "positive df_resid or hc1_correction is required for robust covariance"
                )
            correction = n / float(df_resid)
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "robust",
                    "hc_equivalent": "hc1",
                    "hc1_correction": float(correction),
                }
            )
        return _symmetrize(
            influence.T @ influence * float(correction)
        )

    if name in {"hc0", "hc2", "hc3"}:
        return _hc_covariance(
            X, resid, kind=name, xp=xp, metadata=metadata
        )

    if name == "clustered":
        if cluster is None:
            raise ValueError("cluster is required for cov_type='clustered'")
        cluster_np = np.asarray(_to_numpy(cluster))
        if cluster_np.ndim == 2 and cluster_np.shape[1] == 2:
            return two_way_clustered_covariance(
                X,
                resid,
                cluster_np[:, 0],
                cluster_np[:, 1],
                xp=xp,
                group_debias=group_debias,
                metadata=metadata,
            )
        if cluster_np.ndim == 2 and cluster_np.shape[1] == 1:
            cluster_np = cluster_np[:, 0]
        if cluster_np.ndim != 1:
            raise ValueError(
                "cluster must be one-dimensional, (n, 1), or (n, 2)"
            )
        return clustered_covariance(
            X,
            resid,
            cluster_np,
            xp=xp,
            group_debias=group_debias,
            metadata=metadata,
        )

    if name == "hac":
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "hac",
                    "kernel": "bartlett",
                    "bandwidth": bandwidth,
                    "row_order_hac": True,
                }
            )
        return hac_covariance(
            X, resid, bandwidth=bandwidth, kernel=kernel, xp=xp
        )

    if name == "driscoll-kraay":
        if time_ids is None:
            raise ValueError(
                "time_ids is required for Driscoll-Kraay covariance"
            )
        return driscoll_kraay_covariance(
            X,
            resid,
            time_ids,
            bandwidth=bandwidth,
            kernel=kernel,
            extra_df=extra_df,
            xp=xp,
            metadata=metadata,
        )

    raise ValueError(
        "cov_type must be one of 'nonrobust', 'robust', 'hc0', 'hc1', "
        "'hc2', 'hc3', 'clustered', 'hac', or 'driscoll-kraay'"
    )
