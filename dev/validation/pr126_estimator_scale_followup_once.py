from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Group projection is estimator infrastructure, not merely diagnostics. Make the
# compact mean itself overflow-safe so Between/RE/within/two-way consumers all
# share one implementation and ordinary groups retain the original arithmetic.
replace_once(
    "statgpu/panel/_utils.py",
    '''def _compact_group_means(values, projection, xp):\n    idx, n_groups, _labels, _counts, inv_counts = projection\n    sums = _scatter_add(xp, idx, values, n_groups)\n    return sums * inv_counts\n''',
    '''def _compact_group_means(values, projection, xp):\n    idx, n_groups, _labels, counts, inv_counts = projection\n    # A group sum can overflow even though its mean is finite. Only groups whose\n    # raw accumulation is at risk are divided by their own count before the\n    # scatter-add; safe groups keep the historical arithmetic exactly.\n    counts_aligned = counts[idx]\n    limit = np.finfo(np.float64).max / xp_maximum(counts_aligned, 1.0, xp)\n    dangerous_obs = (xp.abs(values) > limit) * 1.0\n    dangerous_count = _scatter_add(xp, idx, dangerous_obs, n_groups)\n    factor_compact = xp.where(\n        dangerous_count > 0.0, counts, xp.ones_like(counts)\n    )\n    factor_aligned = factor_compact[idx]\n    sums = _scatter_add(xp, idx, values / factor_aligned, n_groups)\n    return sums * inv_counts * factor_compact\n''',
    "safe compact group means",
)

# Shared stable residual variance. Unlike forming RSS first, this can represent
# RSS/df when RSS itself is outside float64 because division by df is applied at
# the RMS stage before the final square.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''def _scaled_residual_r2(resid, centered, xp) -> Tuple[float, bool]:\n''',
    '''def _scaled_residual_variance(resid, df_resid: int, xp) -> float:\n    """Return ``sum(resid**2) / df_resid`` without avoidable RSS overflow."""\n    if int(df_resid) <= 0:\n        raise ValueError("df_resid must be positive")\n    scale = xp.max(xp.abs(resid))\n    scale_value = _to_float_scalar(scale)\n    if scale_value == 0.0:\n        return 0.0\n    unit = resid / scale\n    norm_sq = _to_float_scalar(xp.sum(unit * unit))\n    root = scale_value * float(\n        np.sqrt(norm_sq / float(int(df_resid)))\n    )\n    return float(np.float64(root) * np.float64(root))\n\n\ndef _scaled_residual_r2(resid, centered, xp) -> Tuple[float, bool]:\n''',
    "stable residual variance helper",
)

# PooledOLS: stable scale, mean and legacy R2.
replace_once(
    "statgpu/panel/_pooled.py",
    '''        resid = y_arr - X_arr @ params\n        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid\n''',
    '''        resid = y_arr - X_arr @ params\n        from statgpu.panel._diagnostics import (\n            _restore_squared_scale,\n            _scaled_mean,\n            _scaled_residual_r2,\n            _scaled_residual_variance,\n        )\n\n        scale = _scaled_residual_variance(resid, df_resid, xp)\n''',
    "Pooled stable scale",
)
replace_once(
    "statgpu/panel/_pooled.py",
    '''        y_mean = xp.mean(y_arr)\n        ss_tot = _to_float_scalar(xp.sum((y_arr - y_mean) ** 2))\n        ss_res = _to_float_scalar(xp.sum(resid * resid))\n        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")\n''',
    '''        y_centered = y_arr - _scaled_mean(y_arr, xp)\n        self.rsquared, r2_degenerate = _scaled_residual_r2(\n            resid, y_centered, xp\n        )\n        if r2_degenerate:\n            self.rsquared = float("nan")\n        resid_scale = xp.max(xp.abs(resid))\n        centered_scale = xp.max(xp.abs(y_centered))\n        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        centered_unit = y_centered / xp.where(\n            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)\n        )\n        ss_res = _restore_squared_scale(\n            _to_float_scalar(xp.sum(resid_unit * resid_unit)),\n            _to_float_scalar(resid_scale),\n        )\n        ss_tot = _restore_squared_scale(\n            _to_float_scalar(xp.sum(centered_unit * centered_unit)),\n            _to_float_scalar(centered_scale),\n        )\n''',
    "Pooled stable legacy R2",
)

# BetweenOLS: shared group means are now safe; use stable variance/R2 too.
replace_once(
    "statgpu/panel/_between.py",
    '''        df_resid = n - int(rank_mean)\n        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid\n''',
    '''        df_resid = n - int(rank_mean)\n        from statgpu.panel._diagnostics import (\n            _restore_squared_scale,\n            _scaled_mean,\n            _scaled_residual_r2,\n            _scaled_residual_variance,\n        )\n\n        scale = _scaled_residual_variance(resid, df_resid, xp)\n''',
    "Between stable scale",
)
replace_once(
    "statgpu/panel/_between.py",
    '''        y_bar = xp.mean(y_mean)\n        ss_tot = _to_float_scalar(xp.sum((y_mean - y_bar) ** 2))\n        ss_res = _to_float_scalar(xp.sum(resid * resid))\n        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")\n''',
    '''        y_centered = y_mean - _scaled_mean(y_mean, xp)\n        self.rsquared, r2_degenerate = _scaled_residual_r2(\n            resid, y_centered, xp\n        )\n        if r2_degenerate:\n            self.rsquared = float("nan")\n        resid_scale = xp.max(xp.abs(resid))\n        centered_scale = xp.max(xp.abs(y_centered))\n        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        centered_unit = y_centered / xp.where(\n            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)\n        )\n        ss_res = _restore_squared_scale(\n            _to_float_scalar(xp.sum(resid_unit * resid_unit)),\n            _to_float_scalar(resid_scale),\n        )\n        ss_tot = _restore_squared_scale(\n            _to_float_scalar(xp.sum(centered_unit * centered_unit)),\n            _to_float_scalar(centered_scale),\n        )\n''',
    "Between stable legacy R2",
)

# FirstDifferenceOLS preserves its historical centered legacy R2 definition.
replace_once(
    "statgpu/panel/_first_diff.py",
    '''        df_resid = n - int(rank_diff)\n        resid = y_diff - X_diff @ params\n        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid\n''',
    '''        df_resid = n - int(rank_diff)\n        resid = y_diff - X_diff @ params\n        from statgpu.panel._diagnostics import (\n            _restore_squared_scale,\n            _scaled_mean,\n            _scaled_residual_r2,\n            _scaled_residual_variance,\n        )\n\n        scale = _scaled_residual_variance(resid, df_resid, xp)\n''',
    "FD stable scale",
)
replace_once(
    "statgpu/panel/_first_diff.py",
    '''        y_bar = xp.mean(y_diff)\n        ss_tot = _to_float_scalar(xp.sum((y_diff - y_bar) ** 2))\n        ss_res = _to_float_scalar(xp.sum(resid * resid))\n        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")\n''',
    '''        y_centered = y_diff - _scaled_mean(y_diff, xp)\n        self.rsquared, r2_degenerate = _scaled_residual_r2(\n            resid, y_centered, xp\n        )\n        if r2_degenerate:\n            self.rsquared = float("nan")\n        resid_scale = xp.max(xp.abs(resid))\n        centered_scale = xp.max(xp.abs(y_centered))\n        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        centered_unit = y_centered / xp.where(\n            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)\n        )\n        ss_res = _restore_squared_scale(\n            _to_float_scalar(xp.sum(resid_unit * resid_unit)),\n            _to_float_scalar(resid_scale),\n        )\n        ss_tot = _restore_squared_scale(\n            _to_float_scalar(xp.sum(centered_unit * centered_unit)),\n            _to_float_scalar(centered_scale),\n        )\n''',
    "FD stable legacy R2",
)

# PanelOLS: stable inference scale, grand mean, legacy within R2 and diagnostic
# metadata sums. The transformed definition is unchanged.
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''        y_pred = X_d @ coef\n        resid = y_d - y_pred\n        scale = _to_float_scalar(xp.sum(resid ** 2)) / self.df_resid\n        self._scale = scale\n''',
    '''        y_pred = X_d @ coef\n        resid = y_d - y_pred\n        from statgpu.panel._diagnostics import (\n            _restore_squared_scale,\n            _scaled_mean,\n            _scaled_residual_r2,\n            _scaled_residual_variance,\n        )\n\n        scale = _scaled_residual_variance(resid, self.df_resid, xp)\n        self._scale = scale\n''',
    "Panel stable scale",
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''        resid_orig = y_arr - X_arr @ coef\n        grand_mean = float(xp.mean(resid_orig))\n''',
    '''        resid_orig = y_arr - X_arr @ coef\n        grand_mean = _to_float_scalar(_scaled_mean(resid_orig, xp))\n''',
    "Panel stable grand mean",
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''        ss_res = _to_float_scalar(xp.sum(resid ** 2))\n        if has_level_constant or self.entity_effects or self.time_effects:\n            y_d_mean = _to_float_scalar(xp.mean(y_d))\n            ss_tot = _to_float_scalar(xp.sum((y_d - y_d_mean) ** 2))\n        else:\n            ss_tot = _to_float_scalar(xp.sum(y_d ** 2))\n        self.rsquared_within = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0\n\n        if has_level_constant:\n            y_d_centered = y_d - xp.mean(y_d)\n            ss_tot_diag = _to_float_scalar(xp.sum(y_d_centered * y_d_centered))\n        else:\n            ss_tot_diag = _to_float_scalar(xp.sum(y_d * y_d))\n''',
    '''        if has_level_constant or self.entity_effects or self.time_effects:\n            y_d_centered = y_d - _scaled_mean(y_d, xp)\n        else:\n            y_d_centered = y_d\n        self.rsquared_within, _r2_degenerate = _scaled_residual_r2(\n            resid, y_d_centered, xp\n        )\n        resid_scale = xp.max(xp.abs(resid))\n        total_scale = xp.max(xp.abs(y_d_centered))\n        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        total_unit = y_d_centered / xp.where(\n            total_scale > 0.0, total_scale, xp.ones_like(total_scale)\n        )\n        ss_res = _restore_squared_scale(\n            _to_float_scalar(xp.sum(resid_unit * resid_unit)),\n            _to_float_scalar(resid_scale),\n        )\n        ss_tot_diag = _restore_squared_scale(\n            _to_float_scalar(xp.sum(total_unit * total_unit)),\n            _to_float_scalar(total_scale),\n        )\n''',
    "Panel stable within R2",
)

# RandomEffects: compute the Swamy-Arora variance *ratios* on one common residual
# scale. Public variance components are restored to original units only after the
# transformation parameters have been determined safely.
replace_once(
    "statgpu/panel/_random_effects.py",
    '''        rss_between = float(xp.sum(resid_between ** 2))\n\n        # --- Step 2: Within estimation ---\n''',
    '''        # Delay quadratic accumulation until both auxiliary residual series\n        # are available so they can share one safe working scale.\n\n        # --- Step 2: Within estimation ---\n''',
    "RE defer between RSS",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    '''        rss_within = float(xp.sum(resid_within ** 2))\n\n        # --- Step 3: Swamy-Arora variance components ---\n''',
    '''        from statgpu.panel._diagnostics import (\n            _common_scaled_sumsquares,\n            _restore_squared_scale,\n            _scaled_residual_variance,\n        )\n        rss_within_work, rss_between_work, variance_scale = (\n            _common_scaled_sumsquares(resid_within, resid_between, xp)\n        )\n\n        # --- Step 3: Swamy-Arora variance components ---\n''',
    "RE common residual scale",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    '''        sigma2_e = rss_within / df_within\n\n        df_between = n_entities - int(rank_between)\n''',
    '''        sigma2_e_work = rss_within_work / df_within\n\n        df_between = n_entities - int(rank_between)\n''',
    "RE within variance working scale",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    '''        s_b_sq = rss_between / df_between\n        sigma2_a_raw = (s_b_sq - sigma2_e) / T_bar\n        sigma2_a = max(0.0, sigma2_a_raw)\n        self.variance_components_ = {\n            "sigma2_e": sigma2_e,\n            "sigma2_a": sigma2_a,\n        }\n\n        # --- Step 4: GLS transformation ---\n''',
    '''        s_b_sq_work = rss_between_work / df_between\n        sigma2_a_work = max(\n            0.0, (s_b_sq_work - sigma2_e_work) / T_bar\n        )\n        self.variance_components_ = {\n            "sigma2_e": _restore_squared_scale(\n                sigma2_e_work, variance_scale\n            ),\n            "sigma2_a": _restore_squared_scale(\n                sigma2_a_work, variance_scale\n            ),\n        }\n\n        # --- Step 4: GLS transformation ---\n''',
    "RE variance components working scale",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    '''            denom = sigma2_e + Ti * sigma2_a\n            if denom > 0:\n                theta_map[Ti] = 1.0 - np.sqrt(sigma2_e / denom)\n''',
    '''            denom = sigma2_e_work + Ti * sigma2_a_work\n            if denom > 0:\n                theta_map[Ti] = 1.0 - np.sqrt(sigma2_e_work / denom)\n''',
    "RE theta working variances",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    '''        self._scale = _to_float_scalar(xp.sum(resid_gls ** 2)) / df_resid\n''',
    '''        self._scale = _scaled_residual_variance(\n            resid_gls, df_resid, xp\n        )\n''',
    "RE final stable scale",
)

# Tests: global group mean, public legacy R2 and end-to-end RE scale invariance.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel._utils import demean_variables\n",
    "from statgpu.panel._utils import demean_variables, group_means\n",
    1,
)
marker = "def test_group_means_avoid_finite_same_sign_overflow():"
if marker in text:
    raise RuntimeError("estimator-scale regressions already present")
text += r'''


def test_group_means_avoid_finite_same_sign_overflow():
    groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    values = np.asarray([6.0e307, 6.0e307, 6.0e307, 6.0e307, 2.0, 4.0])
    actual = group_means(values, groups, xp=np)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual[:4], np.full(4, 6.0e307), rtol=2e-15)
    assert_allclose(actual[4:], np.full(2, 3.0), rtol=0.0, atol=0.0)


def test_pooled_legacy_rsquared_survives_overflowing_raw_rss():
    rng = np.random.default_rng(20260818)
    n = 160
    x = rng.normal(size=(n, 1))
    noise = rng.normal(scale=0.2, size=n)
    base_y = 0.7 + 0.5 * x[:, 0] + noise
    reference = PooledOLS(cov_type="nonrobust").fit(x, base_y)
    scale = 1.0e154
    candidate = PooledOLS(cov_type="nonrobust").fit(x, scale * base_y)
    assert np.isfinite(candidate.rsquared)
    assert_allclose(candidate.rsquared, reference.rsquared, rtol=5e-13, atol=5e-15)
    assert np.all(np.isfinite(candidate.cov_params_))


def test_random_effects_transformation_is_invariant_when_raw_auxiliary_rss_overflows():
    rng = np.random.default_rng(20260819)
    n_entities, n_times = 24, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    x = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.5, size=n_entities), n_times)
    noise = rng.normal(scale=0.15, size=entity.size)
    y = x @ np.asarray([0.6, -0.3]) + alpha + noise
    reference = RandomEffects(cov_type="nonrobust").fit(
        x, y, entity_ids=entity
    )
    scale = 1.0e200
    candidate = RandomEffects(cov_type="nonrobust").fit(
        scale * x, scale * y, entity_ids=entity
    )
    assert np.isfinite(candidate.theta_)
    assert_allclose(candidate.theta_, reference.theta_, rtol=3e-12, atol=3e-14)
    assert_allclose(candidate.coef_, reference.coef_, rtol=2e-11, atol=2e-12)
    assert np.all(np.isfinite(candidate.cov_params_))
'''
test_path.write_text(text, encoding="utf-8")

print("PR126 estimator-scale follow-up staged")
