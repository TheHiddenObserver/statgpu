"""Estimator-integration helpers for Panel Tier-1 Stage B diagnostics.

This module keeps observation-scale sufficient-statistic work backend-native and
bridges fitted estimator arrays to the structured primitives in
``statgpu.panel._diagnostics``.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar, _to_numpy
from statgpu.inference._distributions_backend import get_distribution
from statgpu.panel._diagnostics import (
    _applicable,
    _build_fit_statistics,
    _diagnostic_identity,
    _inapplicable,
    _matrix_rank,
    _pooling_f_from_sums,
)
from statgpu.panel._linalg import panel_lstsq
from statgpu.panel._utils import group_means, group_sizes


def _two_way_incidence_components(
    entity_codes,
    time_codes,
    *,
    n_entities: int,
    n_times: int,
) -> int:
    """Count connected components in the observed entity-time incidence graph."""
    if entity_codes is None or time_codes is None:
        raise ValueError(
            "two-way fixed-effect diagnostic rank requires entity and time codes"
        )

    entity = np.asarray(_to_numpy(entity_codes), dtype=np.int64).ravel()
    time = np.asarray(_to_numpy(time_codes), dtype=np.int64).ravel()
    if entity.shape != time.shape:
        raise ValueError("entity and time codes must have the same length")

    n_entities = int(n_entities)
    n_times = int(n_times)
    total = n_entities + n_times
    parent = np.arange(total, dtype=np.int64)
    rank = np.zeros(total, dtype=np.int8)

    def find(node: int) -> int:
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            nxt = int(parent[node])
            parent[node] = root
            node = nxt
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for entity_code, time_code in zip(entity, time):
        entity_code = int(entity_code)
        time_code = int(time_code)
        if not (0 <= entity_code < n_entities):
            raise ValueError("entity diagnostic code is out of range")
        if not (0 <= time_code < n_times):
            raise ValueError("time diagnostic code is out of range")
        union(entity_code, n_entities + time_code)

    active = np.zeros(total, dtype=bool)
    active[entity] = True
    active[n_entities + time] = True
    return len({find(int(node)) for node in np.flatnonzero(active)})


def effect_rank_standard(
    *,
    n_entities: int,
    n_times: int,
    entity_effects: bool,
    time_effects: bool,
    has_constant: bool = False,
    n_components: int = 1,
) -> int:
    """Return the nuisance-effect rank under the standard FE parameterization."""
    n_entities = int(n_entities)
    n_times = int(n_times)
    n_components = int(n_components)
    if not entity_effects and not time_effects:
        return 0

    if entity_effects and time_effects:
        if n_components <= 0:
            raise ValueError("two-way incidence component count must be positive")
        full_dummy_rank = max(n_entities + n_times - n_components, 0)
        if has_constant:
            return int(max(full_dummy_rank - 1, 0))
        return int(full_dummy_rank)

    if has_constant:
        if entity_effects:
            return int(max(n_entities - 1, 0))
        return int(max(n_times - 1, 0))

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
    entity_codes=None,
    time_codes=None,
    rank_x=None,
):
    """Return the standard rank-consistent fixed-effect df decomposition."""
    if rank_x is None:
        rank_x = _matrix_rank(X_transformed, xp)
    rank_x = int(rank_x)
    n_components = 1
    if entity_effects and time_effects:
        n_components = _two_way_incidence_components(
            entity_codes,
            time_codes,
            n_entities=n_entities,
            n_times=n_times,
        )
    effect_rank = effect_rank_standard(
        n_entities=n_entities,
        n_times=n_times,
        entity_effects=entity_effects,
        time_effects=time_effects,
        has_constant=has_constant,
        n_components=n_components,
    )
    df_resid = int(nobs) - int(rank_x) - int(effect_rank)
    level_constant_rank = int(
        bool(has_constant) and not entity_effects and not time_effects
    )
    df_total = int(nobs) - int(effect_rank) - level_constant_rank
    return {
        "rank_x": int(rank_x),
        "effect_rank": int(effect_rank),
        "df_resid": int(df_resid),
        "df_total": int(df_total),
        "incidence_components": int(n_components),
        "legacy_df_unchanged": False,
        "public_df_uses_standard_effect_rank": True,
    }


def explicit_constant_column(X, *, xp):
    """Return an identified explicit constant-column index, if one is present.

    The classification is relative to each column's own magnitude so multiplying
    a valid design column by a nonzero unit-conversion factor does not change
    whether it is recognized as an explicit constant.
    """
    if int(X.shape[1]) == 0:
        return None
    if getattr(xp, "__name__", "") == "torch":
        min_native = xp.amin(X, dim=0)
        max_native = xp.amax(X, dim=0)
    else:
        min_native = xp.min(X, axis=0)
        max_native = xp.max(X, axis=0)
    col_min = np.asarray(_to_numpy(min_native), dtype=np.float64).ravel()
    col_max = np.asarray(_to_numpy(max_native), dtype=np.float64).ravel()
    magnitude = np.maximum(np.abs(col_min), np.abs(col_max))
    tol = 256.0 * np.finfo(np.float64).eps * magnitude
    span = np.abs(col_max - col_min)
    candidates = np.flatnonzero((span <= tol) & (magnitude > 0.0))
    if candidates.size == 0:
        return None
    return int(candidates[0])


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
    if rank_pool < int(X_pool.shape[1]):
        beta_pool, _ = panel_lstsq(X_pool, y_pool, xp)
    else:
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
    # Entity codes are diagnostic metadata, so the common explicit metadata
    # conversion is allowed. Observation-scale residuals remain backend-native.
    codes_np = np.asarray(_to_numpy(entity_codes), dtype=np.int64).ravel()
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
    # entity-level residual-sum vector to CPU. sum_i s_i^2 equals
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
