"""Estimator-integration helpers for Panel Tier-1 Stage B diagnostics.

This module keeps observation-scale sufficient-statistic work backend-native and
bridges fitted estimator arrays to the structured primitives in
``statgpu.panel._diagnostics``.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar
from statgpu.inference._distributions_backend import get_distribution
from statgpu.panel._diagnostics import (
    _applicable,
    _build_fit_statistics,
    _diagnostic_identity,
    _inapplicable,
    _matrix_rank,
    _pooling_f_from_sums,
)
from statgpu.panel._utils import group_means, group_sizes


def effect_rank_standard(
    *,
    n_entities: int,
    n_times: int,
    entity_effects: bool,
    time_effects: bool,
    has_constant: bool = False,
) -> int:
    """Return the nuisance-effect rank under the standard FE parameterization."""
    n_entities = int(n_entities)
    n_times = int(n_times)
    if not entity_effects and not time_effects:
        return 0

    if has_constant:
        rank = 0
        if entity_effects:
            rank += max(n_entities - 1, 0)
        if time_effects:
            rank += max(n_times - 1, 0)
        return int(rank)

    if entity_effects and time_effects:
        return int(max(n_entities + n_times - 1, 0))
    if entity_effects:
        return int(max(n_entities, 0))
    return int(max(n_times, 0))


def fixed_effect_diagnostic_df(
    X_transformed,
    *,
    xp,
    nobs: int,
    n_entities: int,
    n_times: int,
    entity_effects: bool,
    time_effects: bool,
    has_constant: bool = False,
):
    """Return rank-consistent Stage-B FE diagnostic df without changing legacy df."""
    rank_x = _matrix_rank(X_transformed, xp)
    effect_rank = effect_rank_standard(
        n_entities=n_entities,
        n_times=n_times,
        entity_effects=entity_effects,
        time_effects=time_effects,
        has_constant=has_constant,
    )
    df_resid = int(nobs) - int(rank_x) - int(effect_rank)
    df_total = int(nobs) - int(effect_rank)
    return {
        "rank_x": int(rank_x),
        "effect_rank": int(effect_rank),
        "df_resid": int(df_resid),
        "df_total": int(df_total),
        "legacy_df_unchanged": True,
    }


def pooling_f_from_level_arrays(
    y,
    X,
    *,
    xp,
    rss_effects: float,
    df_resid_effects: int,
    has_constant: bool = False,
):
    """Construct the nested pooled null on the exact aligned FE level sample."""
    n = int(y.shape[0])
    if has_constant:
        y_pool = y
        X_pool = X
        constant_projection_df = 0
    else:
        y_pool = y - xp.mean(y)
        X_pool = X - xp.mean(X, axis=0)
        constant_projection_df = 1

    rank_pool = _matrix_rank(X_pool, xp)
    beta_pool = xp.linalg.pinv(X_pool) @ y_pool
    resid_pool = y_pool - X_pool @ beta_pool
    rss_pool = _to_float_scalar(xp.sum(resid_pool * resid_pool))
    df_resid_pool = n - rank_pool - constant_projection_df
    df_num = int(df_resid_pool) - int(df_resid_effects)
    return _pooling_f_from_sums(
        rss_pooled=float(rss_pool),
        rss_effects=float(rss_effects),
        df_num=int(df_num),
        df_denom=int(df_resid_effects),
        metadata={
            "rank_pooled": int(rank_pool),
            "df_resid_pooled": int(df_resid_pool),
            "df_resid_effects_standard": int(df_resid_effects),
            "constant_projection_df": int(constant_projection_df),
            "has_explicit_level_constant": bool(has_constant),
        },
    )


def bp_lm_from_residuals(resid, entity_codes, *, xp):
    """Compute the one-way Baltagi-Li BP-LM from pooled residuals on backend."""
    null = "the entity random-effect variance is zero"
    alternative = "a nonzero entity random-effect variance component is present"
    if entity_codes is None:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="entity_ids were not supplied to the pooled fit",
        )

    nobs = int(resid.shape[0])
    codes_np = np.asarray(entity_codes if xp is np else entity_codes.get() if getattr(xp, "__name__", "") == "cupy" else entity_codes.detach().cpu().numpy()).ravel()
    n_entities = int(np.unique(codes_np).size)
    meta = {
        "n_entities": n_entities,
        "nobs": nobs,
        "definition": "Baltagi-Li one-way unbalanced Breusch-Pagan LM",
    }
    if n_entities < 2:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="Breusch-Pagan LM requires at least two entities",
            metadata=meta,
        )

    residual_ss = _to_float_scalar(xp.sum(resid * resid))
    if residual_ss <= 0.0:
        meta["residual_ss"] = float(residual_ss)
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="pooled residual sum of squares must be positive",
            metadata=meta,
        )

    mean_aligned = group_means(resid, entity_codes, xp=xp)
    sizes_aligned = group_sizes(entity_codes, xp=xp)
    # Repeated aligned values allow scalar reductions without transferring the
    # entity-level residual-sum vector to CPU.  sum_i s_i^2 equals
    # sum_obs mean_i^2 * T_i because each group contributes T_i copies.
    sum_group_sums_sq = _to_float_scalar(
        xp.sum(mean_aligned * mean_aligned * sizes_aligned)
    )
    # Likewise sum_i T_i^2 = sum_obs T_i.
    m11 = _to_float_scalar(xp.sum(sizes_aligned))
    if m11 <= nobs:
        meta.update({"residual_ss": float(residual_ss), "M11": float(m11)})
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=1.0,
            reason="Breusch-Pagan LM requires at least one entity with repeated observations",
            metadata=meta,
        )

    a1 = float(sum_group_sums_sq / residual_ss - 1.0)
    lm1 = float(nobs * np.sqrt(1.0 / (2.0 * (m11 - nobs))) * a1)
    statistic = float(lm1 * lm1)
    dist = get_distribution("chi2", backend="numpy")
    pvalue = _to_float_scalar(dist.sf(statistic, 1.0))
    meta.update(
        {
            "residual_ss": float(residual_ss),
            "A1": a1,
            "M11": float(m11),
            "LM1": lm1,
        }
    )
    return _applicable(
        statistic,
        pvalue,
        null=null,
        alternative=alternative,
        distribution="chi2",
        df=1.0,
        metadata=meta,
    )


def build_model_fit_statistics(*args, **kwargs):
    """Thin estimator-facing wrapper around the shared fit-statistics builder."""
    return _build_fit_statistics(*args, **kwargs)


def build_diagnostic_identity(*args, **kwargs):
    """Thin estimator-facing wrapper around the shared numerical fingerprint."""
    return _diagnostic_identity(*args, **kwargs)
