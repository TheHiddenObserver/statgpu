"""Shared Panel Tier-1 diagnostics and fit-statistic primitives.

Stage B of Issue #93 adds structured specification tests and parameter-based
fit statistics without changing the Stage-A estimator transformations or
covariance definitions. Observation-scale statistical accumulation stays on the
selected NumPy/CuPy/Torch backend. Hausman sample identity additionally uses a
chunked full-content SHA-256 over normalized float64 y/common-slope values so
different aligned samples cannot be accepted merely because low-order moments
collide, while an RE-only explicit intercept may be absorbed by FE.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray
from statgpu.inference._distributions_backend import get_distribution
from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank
from statgpu.panel._results import PanelFitStatistics, PanelTestResult
from statgpu.panel._utils import group_means, group_sizes

__all__ = [
    "hausman_test",
    "pooling_f_test",
    "breusch_pagan_lm_test",
]


def _inapplicable(
    *,
    null: str,
    alternative: str,
    distribution: Optional[str],
    reason: str,
    df=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PanelTestResult:
    return PanelTestResult(
        statistic=None,
        pvalue=None,
        distribution=distribution,
        df=df,
        null=null,
        alternative=alternative,
        applicable=False,
        reason=str(reason),
        metadata={} if metadata is None else dict(metadata),
    )


def _applicable(
    statistic: float,
    pvalue: float,
    *,
    null: str,
    alternative: str,
    distribution: str,
    df,
    metadata: Optional[Dict[str, Any]] = None,
) -> PanelTestResult:
    return PanelTestResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        distribution=distribution,
        df=df,
        null=null,
        alternative=alternative,
        applicable=True,
        reason=None,
        metadata={} if metadata is None else dict(metadata),
    )


def _matrix_rank(X, xp) -> int:
    return panel_matrix_rank(X, xp)


def _relative_tolerance(*values: float, factor: float = 256.0) -> float:
    """Return a float64 roundoff tolerance that preserves scale equivariance.

    Statistical quantities such as RSS and covariance matrices have physical
    scale. Using an absolute ``max(1, scale)`` floor makes F/Hausman decisions
    depend on arbitrary units. A zero scale therefore maps to an exact zero
    tolerance; otherwise the tolerance scales linearly with the compared value.
    """
    scale = max((abs(float(value)) for value in values), default=0.0)
    return float(factor) * np.finfo(np.float64).eps * scale


def _safe_r2(ss_res: float, ss_tot: float) -> Tuple[float, bool]:
    """Return linearmodels-style parameter R² and a degenerate-TSS flag."""
    ss_res = float(ss_res)
    ss_tot = float(ss_tot)
    if ss_tot <= 0.0:
        return 0.0, True
    return 1.0 - ss_res / ss_tot, False


def _scaled_residual_r2(resid, centered, xp) -> Tuple[float, bool]:
    """Return R² from a common scale without overflow in squared reductions."""
    resid_scale = xp.max(xp.abs(resid))
    centered_scale = xp.max(xp.abs(centered))
    scale = xp.maximum(resid_scale, centered_scale)
    scale_value = _to_float_scalar(scale)
    if scale_value == 0.0:
        return 0.0, True
    resid_scaled = resid / scale
    centered_scaled = centered / scale
    ss_res = _to_float_scalar(xp.sum(resid_scaled * resid_scaled))
    ss_tot = _to_float_scalar(xp.sum(centered_scaled * centered_scaled))
    return _safe_r2(ss_res, ss_tot)


def _scaled_mean(values, xp):
    """Return a mean with only the reduction-length scaling needed for safety."""
    n = int(values.shape[0])
    max_abs = xp.max(xp.abs(values))
    limit = np.finfo(np.float64).max / float(max(n, 1))
    factor = xp.where(
        max_abs > limit,
        xp.full_like(max_abs, float(n)),
        xp.ones_like(max_abs),
    )
    return xp.sum(values / factor) * (factor / float(n))


def _scaled_group_means(values, groups, xp):
    """Return group means without globally magnitude-normalizing safe groups."""
    sizes = group_sizes(groups, xp=xp)
    limit = np.finfo(np.float64).max / sizes
    dangerous = (xp.abs(values) > limit) * 1.0
    dangerous_group = group_means(dangerous, groups, xp=xp) > 0.0
    factor = xp.where(dangerous_group, sizes, xp.ones_like(sizes))
    # For a dangerous group of size m, group_means(values / m) * m equals
    # the original mean but the same-sign accumulation is bounded by max|x|.
    return group_means(values / factor, groups, xp=xp) * factor


def _demean_matrix(X, entity_codes, xp):
    out = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()
    for j in range(int(X.shape[1])):
        out[:, j] = X[:, j] - _scaled_group_means(
            X[:, j], entity_codes, xp
        )
    return out


def _first_group_indices(entity_codes, xp, ref_arr):
    codes_np = np.asarray(_to_numpy(entity_codes), dtype=np.int64).ravel()
    _, first = np.unique(codes_np, return_index=True)
    first.sort()
    return xp_asarray(first, dtype=xp.int64, xp=xp, ref_arr=ref_arr)


def _parameter_r2_components(
    y,
    X,
    params,
    *,
    xp,
    entity_codes=None,
    has_constant: bool,
) -> Tuple[Optional[float], Optional[float], float, Dict[str, bool]]:
    """Compute parameter-based within, between and overall R².

    ``has_constant`` refers to an actual identified constant in the level
    regressor design. Fixed effects alone do not set this flag. The pooling-F
    common-constant correction is a separate nested-test rule.
    """
    params = params.ravel()
    overall_resid = y - X @ params
    overall_center = y - _scaled_mean(y, xp) if has_constant else y
    overall, deg_o = _scaled_residual_r2(overall_resid, overall_center, xp)

    if entity_codes is None:
        return None, None, overall, {
            "within": False,
            "between": False,
            "overall": deg_o,
        }

    y_mean_aligned = _scaled_group_means(y, entity_codes, xp)
    X_mean_aligned = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()
    for j in range(int(X.shape[1])):
        X_mean_aligned[:, j] = _scaled_group_means(
            X[:, j], entity_codes, xp
        )
    first = _first_group_indices(entity_codes, xp, X)
    y_between = y_mean_aligned[first]
    X_between = X_mean_aligned[first]
    between_resid = y_between - X_between @ params
    between_center = (
        y_between - _scaled_mean(y_between, xp) if has_constant else y_between
    )
    between, deg_b = _scaled_residual_r2(between_resid, between_center, xp)

    y_within = y - y_mean_aligned
    X_within = _demean_matrix(X, entity_codes, xp)
    within_resid = y_within - X_within @ params
    within, deg_w = _scaled_residual_r2(within_resid, y_within, xp)

    return within, between, overall, {
        "within": deg_w,
        "between": deg_b,
        "overall": deg_o,
    }


def _adjusted_r2(
    *,
    rss: float,
    tss: float,
    df_resid: int,
    df_total: int,
) -> Optional[float]:
    if int(df_resid) <= 0 or int(df_total) <= 0:
        return None
    if float(tss) <= 0.0:
        return 0.0
    return 1.0 - (float(rss) / float(df_resid)) / (
        float(tss) / float(df_total)
    )


def _classical_model_f(
    y,
    X,
    params,
    *,
    xp,
    df_resid: int,
    has_constant: bool,
    restricted_X=None,
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]], Dict[str, Any]]:
    """Classical homoskedastic joint-slope F statistic in the fit space."""
    rank_u = _matrix_rank(X, xp)
    rank_r = _matrix_rank(restricted_X, xp) if restricted_X is not None else (
        1 if has_constant else 0
    )
    q = rank_u - rank_r
    metadata = {
        "classical_homoskedastic": True,
        "rank_unrestricted": rank_u,
        "rank_restricted": rank_r,
        "restriction_rank": q,
    }
    if restricted_X is not None:
        metadata["restricted_design_supplied"] = True
    if q <= 0 or int(df_resid) <= 0:
        metadata["unavailable_reason"] = "no estimable non-constant restrictions"
        return None, None, None, metadata

    resid = y - X @ params.ravel()
    rss_u = _to_float_scalar(xp.sum(resid * resid))
    if restricted_X is not None:
        if rank_r < int(restricted_X.shape[1]):
            beta_r, _ = panel_lstsq(restricted_X, y, xp)
        else:
            beta_r = xp.linalg.pinv(restricted_X) @ y
        resid_r = y - restricted_X @ beta_r
        rss_r = _to_float_scalar(xp.sum(resid_r * resid_r))
    elif has_constant:
        y_r = y - xp.mean(y)
        rss_r = _to_float_scalar(xp.sum(y_r * y_r))
    else:
        rss_r = _to_float_scalar(xp.sum(y * y))

    diff = rss_r - rss_u
    tol = _relative_tolerance(rss_r, rss_u)
    if diff < -tol:
        metadata["unavailable_reason"] = "restricted RSS is materially below unrestricted RSS"
        metadata["rss_restricted"] = float(rss_r)
        metadata["rss_unrestricted"] = float(rss_u)
        return None, None, None, metadata
    if diff < 0.0:
        diff = 0.0
        metadata["roundoff_normalized"] = True

    metadata["rss_restricted"] = float(rss_r)
    metadata["rss_unrestricted"] = float(rss_u)
    if rss_u <= tol:
        if diff > tol:
            metadata["exact_fit"] = True
            return (
                float("inf"),
                0.0,
                (float(q), float(df_resid)),
                metadata,
            )
        metadata["unavailable_reason"] = (
            "restricted and unrestricted residual sums of squares are both zero"
        )
        return None, None, None, metadata

    statistic = (diff / q) / (rss_u / int(df_resid))
    dist = get_distribution("f", backend="numpy")
    pvalue = _to_float_scalar(dist.sf(statistic, q, int(df_resid)))
    return float(statistic), float(pvalue), (float(q), float(df_resid)), metadata


def _build_fit_statistics(
    y,
    X,
    params,
    *,
    xp,
    entity_codes=None,
    has_constant: bool,
    rss_fit: float,
    tss_fit: float,
    df_resid: int,
    df_total: int,
    f_y=None,
    f_X=None,
    f_params=None,
    f_has_constant: Optional[bool] = None,
    f_restricted_X=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PanelFitStatistics:
    within, between, overall, degenerate = _parameter_r2_components(
        y,
        X,
        params,
        xp=xp,
        entity_codes=entity_codes,
        has_constant=bool(has_constant),
    )
    f_stat, f_pvalue, f_df, f_meta = _classical_model_f(
        y if f_y is None else f_y,
        X if f_X is None else f_X,
        params if f_params is None else f_params,
        xp=xp,
        df_resid=int(df_resid),
        has_constant=(
            bool(has_constant) if f_has_constant is None else bool(f_has_constant)
        ),
        restricted_X=f_restricted_X,
    )
    meta = {} if metadata is None else dict(metadata)
    meta.setdefault("r2_definition", "parameter-based")
    meta["degenerate_total_ss"] = degenerate
    meta["rsquared_adj_basis"] = {
        "df_total": int(df_total),
        "df_resid": int(df_resid),
    }
    meta["model_f"] = f_meta
    if entity_codes is None:
        meta.setdefault("unavailable", {})["within_between_r2"] = (
            "entity_ids were not supplied"
        )
    return PanelFitStatistics(
        rsquared_within=within,
        rsquared_between=between,
        rsquared_overall=overall,
        rsquared_adj=_adjusted_r2(
            rss=float(rss_fit),
            tss=float(tss_fit),
            df_resid=int(df_resid),
            df_total=int(df_total),
        ),
        f_statistic=f_stat,
        f_pvalue=f_pvalue,
        f_df=f_df,
        metadata=meta,
    )


def _pooling_f_from_sums(
    *,
    rss_pooled: float,
    rss_effects: float,
    df_num: int,
    df_denom: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> PanelTestResult:
    null = "all included fixed effects are jointly zero"
    alternative = "at least one included fixed effect is nonzero"
    meta = {} if metadata is None else dict(metadata)
    meta.update(
        {
            "rss_pooled": float(rss_pooled),
            "rss_effects": float(rss_effects),
            "classical_homoskedastic": True,
        }
    )
    if int(df_num) <= 0 or int(df_denom) <= 0:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="F",
            df=(float(df_num), float(df_denom)),
            reason="pooling F requires positive numerator and denominator degrees of freedom",
            metadata=meta,
        )

    diff = float(rss_pooled) - float(rss_effects)
    tol = _relative_tolerance(rss_pooled, rss_effects)
    if diff < -tol:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="F",
            df=(float(df_num), float(df_denom)),
            reason="pooled RSS is materially below fixed-effects RSS; nested-model contract failed",
            metadata=meta,
        )
    if diff < 0.0:
        diff = 0.0
        meta["roundoff_normalized"] = True

    if float(rss_effects) <= tol:
        if diff > tol:
            meta["exact_fit"] = True
            return _applicable(
                float("inf"),
                0.0,
                null=null,
                alternative=alternative,
                distribution="F",
                df=(float(df_num), float(df_denom)),
                metadata=meta,
            )
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="F",
            df=(float(df_num), float(df_denom)),
            reason="pooled and fixed-effects residual sums of squares are both zero",
            metadata=meta,
        )

    statistic = (diff / int(df_num)) / (float(rss_effects) / int(df_denom))
    dist = get_distribution("f", backend="numpy")
    pvalue = _to_float_scalar(dist.sf(statistic, int(df_num), int(df_denom)))
    return _applicable(
        statistic,
        pvalue,
        null=null,
        alternative=alternative,
        distribution="F",
        df=(float(df_num), float(df_denom)),
        metadata=meta,
    )


def _bp_lm_from_components(
    *,
    nobs: int,
    residual_ss: float,
    group_residual_sums: Sequence[float],
    group_counts: Sequence[float],
) -> PanelTestResult:
    null = "the entity random-effect variance is zero"
    alternative = "a nonzero entity random-effect variance component is present"
    sums = np.asarray(group_residual_sums, dtype=np.float64).ravel()
    counts = np.asarray(group_counts, dtype=np.float64).ravel()
    meta = {
        "n_entities": int(counts.size),
        "nobs": int(nobs),
        "residual_ss": float(residual_ss),
        "definition": "Baltagi-Li one-way unbalanced Breusch-Pagan LM",
    }
    if sums.size != counts.size or counts.size < 2:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="Breusch-Pagan LM requires at least two entities",
            metadata=meta,
        )
    if float(residual_ss) <= 0.0:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="pooled residual sum of squares must be positive",
            metadata=meta,
        )
    m11 = float(np.sum(counts * counts))
    if m11 <= int(nobs):
        meta["M11"] = m11
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="Breusch-Pagan LM requires at least one entity with repeated observations",
            metadata=meta,
        )

    a1 = float(np.sum(sums * sums) / float(residual_ss) - 1.0)
    lm1 = int(nobs) * np.sqrt(1.0 / (2.0 * (m11 - int(nobs)))) * a1
    statistic = float(lm1 * lm1)
    dist = get_distribution("chi2", backend="numpy")
    pvalue = _to_float_scalar(dist.sf(statistic, 1.0))
    meta.update({"A1": a1, "M11": m11, "LM1": float(lm1)})
    return _applicable(
        statistic,
        pvalue,
        null=null,
        alternative=alternative,
        distribution="chi2",
        df=1.0,
        metadata=meta,
    )


def _hausman_quadratic(
    difference: Sequence[float],
    covariance_difference,
) -> PanelTestResult:
    null = "the random-effects estimator is consistent"
    alternative = "the random-effects estimator is inconsistent"
    d = np.asarray(difference, dtype=np.float64).ravel()
    D = np.asarray(covariance_difference, dtype=np.float64)
    if D.shape != (d.size, d.size):
        raise ValueError("covariance difference shape must match coefficient difference")
    if d.size == 0:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=0.0,
            reason="Hausman test has no common estimable slope coefficients",
        )

    D = 0.5 * (D + D.T)
    eigvals, eigvecs = np.linalg.eigh(D)
    norm_D = float(np.linalg.norm(D, ord=2)) if D.size else 0.0
    tol = _relative_tolerance(norm_D, factor=256.0 * max(1, d.size))
    meta = {
        "eigen_tolerance": tol,
        "minimum_eigenvalue": float(eigvals.min()),
        "maximum_eigenvalue": float(eigvals.max()),
    }
    if float(eigvals.min()) < -tol:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="covariance difference is not positive semidefinite",
            metadata=meta,
        )

    positive = eigvals > tol
    rank = int(np.count_nonzero(positive))
    meta["rank"] = rank
    if rank == 0:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=0.0,
            reason="covariance difference has zero numerical rank",
            metadata=meta,
        )

    basis = eigvecs[:, positive]
    projected = basis @ (basis.T @ d)
    null_component = d - projected
    range_tol = _relative_tolerance(np.linalg.norm(d), factor=1024.0)
    meta["range_tolerance"] = float(range_tol)
    meta["nullspace_component_norm"] = float(np.linalg.norm(null_component))
    if float(np.linalg.norm(null_component)) > range_tol:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=float(rank),
            reason="coefficient difference has a component outside the identified covariance-difference range",
            metadata=meta,
        )

    inv_eigs = 1.0 / eigvals[positive]
    statistic = float((basis.T @ d).T @ (inv_eigs * (basis.T @ d)))
    stat_tol = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(statistic))
    if statistic < -stat_tol:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=float(rank),
            reason="Hausman quadratic form is materially negative",
            metadata=meta,
        )
    if statistic < 0.0:
        statistic = 0.0
        meta["roundoff_normalized"] = True

    if rank < d.size:
        meta["used_pinv"] = True
        meta["definition_extension"] = "singular PSD generalized-inverse Hausman"
    else:
        meta["used_pinv"] = False

    dist = get_distribution("chi2", backend="numpy")
    pvalue = _to_float_scalar(dist.sf(statistic, float(rank)))
    return _applicable(
        statistic,
        pvalue,
        null=null,
        alternative=alternative,
        distribution="chi2",
        df=float(rank),
        metadata=meta,
    )


def _row_weights(n: int, xp, ref_arr):
    if getattr(xp, "__name__", "") == "torch":
        return xp.arange(
            1,
            int(n) + 1,
            dtype=xp.float64,
            device=ref_arr.device,
        )
    return xp.arange(1, int(n) + 1, dtype=xp.float64)


def _full_content_digest(X, y) -> str:
    """Hash every aligned canonical slope-X/y value with bounded host transfers."""
    h = hashlib.sha256()
    h.update(b"statgpu-panel-diagnostic-identity-v3")

    for label, array, shape in (
        (b"X", X, (int(X.shape[0]), int(X.shape[1]))),
        (b"y", y, (int(y.shape[0]),)),
    ):
        h.update(label)
        h.update(np.asarray(shape, dtype="<i8").tobytes())
        width = int(X.shape[1]) if label == b"X" else 1
        rows_per_chunk = max(1, 1_000_000 // max(width, 1))
        for start in range(0, int(shape[0]), rows_per_chunk):
            stop = min(int(shape[0]), start + rows_per_chunk)
            chunk = np.array(
                _to_numpy(array[start:stop]),
                dtype=np.float64,
                copy=True,
                order="C",
            )
            chunk[chunk == 0.0] = 0.0
            chunk = np.ascontiguousarray(chunk.astype("<f8", copy=False))
            h.update(chunk.tobytes())
    return h.hexdigest()


def _numerical_fingerprint(X, y, *, xp) -> Dict[str, Any]:
    """Retain audit moments plus an authoritative full-content digest."""
    n = int(X.shape[0])
    weights = _row_weights(n, xp, X)
    X_sum = xp.sum(X, axis=0)
    X_sq = xp.sum(X * X, axis=0)
    X_weighted = xp.sum(X * weights.reshape(-1, 1), axis=0)
    y_sum = xp.sum(y)
    y_sq = xp.sum(y * y)
    y_weighted = xp.sum(y * weights)
    return {
        "content_digest": _full_content_digest(X, y),
        "X_sum": np.asarray(_to_numpy(X_sum), dtype=np.float64).ravel(),
        "X_sq": np.asarray(_to_numpy(X_sq), dtype=np.float64).ravel(),
        "X_weighted": np.asarray(_to_numpy(X_weighted), dtype=np.float64).ravel(),
        "y": np.asarray(
            [
                _to_float_scalar(y_sum),
                _to_float_scalar(y_sq),
                _to_float_scalar(y_weighted),
            ],
            dtype=np.float64,
        ),
    }


def _metadata_signature(codes) -> Optional[str]:
    if codes is None:
        return None
    arr = np.asarray(_to_numpy(codes), dtype=np.int64).ravel()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _identity_constant_index(X, *, xp, has_constant: bool) -> Optional[int]:
    """Locate the constant column for canonical Hausman slope identity."""
    if not has_constant or int(X.shape[1]) == 0:
        return None
    if getattr(xp, "__name__", "") == "torch":
        col_min_native = xp.amin(X, dim=0)
        col_max_native = xp.amax(X, dim=0)
    else:
        col_min_native = xp.min(X, axis=0)
        col_max_native = xp.max(X, axis=0)
    col_min = np.asarray(_to_numpy(col_min_native), dtype=np.float64).ravel()
    col_max = np.asarray(_to_numpy(col_max_native), dtype=np.float64).ravel()
    magnitude = np.maximum(np.abs(col_min), np.abs(col_max))
    span = np.abs(col_max - col_min)
    tol = 256.0 * np.finfo(np.float64).eps * magnitude
    candidates = np.flatnonzero((span <= tol) & (magnitude > 0.0))
    return None if candidates.size == 0 else int(candidates[0])


def _diagnostic_identity(
    X,
    y,
    *,
    xp,
    entity_codes=None,
    feature_names: Optional[Sequence[str]] = None,
    has_constant: bool = False,
    constant_column_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Build canonical sample/design identity for a Hausman slope comparison.

    FE absorbs a common intercept, while RE may estimate the same intercept as an
    explicit design column. Identity therefore hashes y and only the slope
    design. ``coefficient_indices`` maps canonical slope positions back to each
    fitted model's original coefficient/covariance positions.
    """
    k_raw = int(X.shape[1])
    constant_index = constant_column_index
    if constant_index is None:
        constant_index = _identity_constant_index(
            X, xp=xp, has_constant=bool(has_constant)
        )
    if constant_index is not None and not (0 <= int(constant_index) < k_raw):
        raise ValueError("constant_column_index is out of range")

    coefficient_indices = tuple(
        index for index in range(k_raw) if index != constant_index
    )
    if len(coefficient_indices) == k_raw:
        X_slopes = X
    else:
        index_dev = xp_asarray(
            np.asarray(coefficient_indices, dtype=np.int64),
            dtype=xp.int64,
            xp=xp,
            ref_arr=X,
        )
        X_slopes = X[:, index_dev]

    if feature_names is None:
        names = tuple(f"x{i + 1}" for i in range(len(coefficient_indices)))
    else:
        raw_names = tuple(feature_names)
        if len(raw_names) == k_raw:
            names = tuple(raw_names[index] for index in coefficient_indices)
        elif len(raw_names) == len(coefficient_indices):
            names = raw_names
        else:
            raise ValueError(
                "feature_names must match the raw or canonical slope feature count"
            )

    return {
        "nobs": int(X.shape[0]),
        "n_features": int(len(coefficient_indices)),
        "feature_names": names,
        "coefficient_indices": coefficient_indices,
        "has_constant": bool(has_constant),
        "constant_column_index": constant_index,
        "entity_signature": _metadata_signature(entity_codes),
        "fingerprint": _numerical_fingerprint(X_slopes, y, xp=xp),
    }


def _fingerprints_match(left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[bool, str]:
    # ``has_constant`` deliberately is not compared: FE may absorb the common
    # intercept while RE estimates it explicitly. Canonical slope X/y identity
    # plus entity and feature metadata is authoritative.
    scalar_keys = ("nobs", "n_features", "feature_names", "entity_signature")
    for key in scalar_keys:
        if left.get(key) != right.get(key):
            return False, f"diagnostic identity mismatch: {key}"
    lf = left.get("fingerprint")
    rf = right.get("fingerprint")
    if not isinstance(lf, dict) or not isinstance(rf, dict):
        return False, "diagnostic identity is missing numerical fingerprint metadata"
    left_digest = lf.get("content_digest")
    right_digest = rf.get("content_digest")
    if not isinstance(left_digest, str) or not isinstance(right_digest, str):
        return False, "diagnostic fingerprint is missing full-content digest"
    if left_digest != right_digest:
        return False, "diagnostic numerical fingerprint mismatch: content_digest"
    return True, ""


def pooling_f_test(fe_model) -> PanelTestResult:
    """Return the classical fixed-effect poolability F test."""
    from statgpu.panel._fixed_effects import PanelOLS

    if not isinstance(fe_model, PanelOLS):
        raise TypeError("pooling_f_test requires a fitted PanelOLS model")
    fe_model._check_is_fitted()
    if not (bool(fe_model.entity_effects) or bool(fe_model.time_effects)):
        return _inapplicable(
            null="all included fixed effects are jointly zero",
            alternative="at least one included fixed effect is nonzero",
            distribution="F",
            reason="pooling F requires at least one fixed-effect dimension",
        )
    result = getattr(fe_model, "_pooling_f_result", None)
    if result is None:
        return _inapplicable(
            null="all included fixed effects are jointly zero",
            alternative="at least one included fixed effect is nonzero",
            distribution="F",
            reason="pooling F sufficient statistics were not retained by this fitted model",
        )
    return result


def breusch_pagan_lm_test(pooled_model) -> PanelTestResult:
    """Return the one-way entity error-components Breusch-Pagan LM test."""
    from statgpu.panel._pooled import PooledOLS

    if not isinstance(pooled_model, PooledOLS):
        raise TypeError("breusch_pagan_lm_test requires a fitted PooledOLS model")
    pooled_model._check_is_fitted()
    result = getattr(pooled_model, "_bp_lm_result", None)
    if result is None:
        return _inapplicable(
            null="the entity random-effect variance is zero",
            alternative="a nonzero entity random-effect variance component is present",
            distribution="chi2",
            df=1.0,
            reason="entity_ids were not supplied to the pooled fit",
        )
    return result


def hausman_test(fe_model, re_model) -> PanelTestResult:
    """Return the classical one-way FE-vs-RE Hausman specification test."""
    from statgpu.panel._fixed_effects import PanelOLS
    from statgpu.panel._random_effects import RandomEffects

    if not isinstance(fe_model, PanelOLS):
        raise TypeError("hausman_test fe_model must be a fitted PanelOLS")
    if not isinstance(re_model, RandomEffects):
        raise TypeError("hausman_test re_model must be a fitted RandomEffects")
    fe_model._check_is_fitted()
    re_model._check_is_fitted()

    null = "the random-effects estimator is consistent"
    alternative = "the random-effects estimator is inconsistent"
    if not bool(fe_model.entity_effects) or bool(fe_model.time_effects):
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="classical Stage-B Hausman requires one-way entity PanelOLS",
        )
    if str(getattr(fe_model, "_cov_type", "nonrobust")).lower() != "nonrobust":
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="classical Hausman requires nonrobust FE covariance; robust auxiliary Hausman is not implemented in Stage B",
        )
    if str(getattr(re_model, "_cov_type", "nonrobust")).lower() != "nonrobust":
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="classical Hausman requires nonrobust RE covariance; robust auxiliary Hausman is not implemented in Stage C",
        )

    unavailable = []
    for label, model in (("FE", fe_model), ("RE", re_model)):
        if not bool(getattr(model, "_coefficient_inference_available", True)):
            model_reason = getattr(model, "_coefficient_inference_reason", None)
            unavailable.append(
                f"{label}: {model_reason or 'coefficient vector is not uniquely identified'}"
            )
    if unavailable:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason=(
                "classical Hausman requires uniquely identified coefficient vectors; "
                + "; ".join(unavailable)
            ),
            metadata={"coefficient_inference_unavailable": tuple(unavailable)},
        )

    left_id = getattr(fe_model, "_panel_diagnostic_identity", None)
    right_id = getattr(re_model, "_panel_diagnostic_identity", None)
    if not isinstance(left_id, dict) or not isinstance(right_id, dict):
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="fitted models do not contain Stage-B sample/design identity metadata",
        )
    matched, reason = _fingerprints_match(left_id, right_id)
    if not matched:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason=reason,
        )

    fe_cov = getattr(fe_model, "_panel_cov_params", None)
    re_cov = getattr(re_model, "_panel_cov_params", None)
    if fe_cov is None or re_cov is None:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="fitted models do not contain the small covariance matrices required for Hausman",
        )

    fe_names = tuple(left_id.get("feature_names", ()))
    re_names = tuple(right_id.get("feature_names", ()))
    common = [name for name in fe_names if name in set(re_names)]
    if not common:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="Hausman test has no common estimable slope coefficients",
        )

    fe_positions = tuple(
        int(value)
        for value in left_id.get("coefficient_indices", range(len(fe_names)))
    )
    re_positions = tuple(
        int(value)
        for value in right_id.get("coefficient_indices", range(len(re_names)))
    )
    if len(fe_positions) != len(fe_names) or len(re_positions) != len(re_names):
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="diagnostic identity has inconsistent coefficient-index metadata",
        )
    fe_lookup = dict(zip(fe_names, fe_positions))
    re_lookup = dict(zip(re_names, re_positions))
    fe_idx = np.asarray([fe_lookup[name] for name in common], dtype=np.int64)
    re_idx = np.asarray([re_lookup[name] for name in common], dtype=np.int64)
    fe_coef = np.asarray(fe_model.coef_, dtype=np.float64).ravel()[fe_idx]
    re_coef = np.asarray(re_model.coef_, dtype=np.float64).ravel()[re_idx]
    fe_cov = np.asarray(fe_cov, dtype=np.float64)[np.ix_(fe_idx, fe_idx)]
    re_cov = np.asarray(re_cov, dtype=np.float64)[np.ix_(re_idx, re_idx)]
    result = _hausman_quadratic(fe_coef - re_coef, fe_cov - re_cov)
    meta = dict(result.metadata)
    meta["common_features"] = tuple(common)
    meta["fe_coefficient_indices"] = tuple(int(value) for value in fe_idx)
    meta["re_coefficient_indices"] = tuple(int(value) for value in re_idx)
    meta["re_explicit_constant_excluded"] = bool(
        right_id.get("constant_column_index") is not None
    )
    return PanelTestResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        distribution=result.distribution,
        df=result.df,
        null=result.null,
        alternative=result.alternative,
        applicable=result.applicable,
        reason=result.reason,
        metadata=meta,
    )
