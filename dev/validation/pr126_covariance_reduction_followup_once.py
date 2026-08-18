from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---- Shared covariance: recover suspicious nonrobust variance scalars from
# the still-available residual vector before combining with the working-design
# pseudoinverse. This preserves final covariance when the standalone variance
# underflows/overflows but the product with bread is representable.
replace_once(
    "statgpu/panel/_covariance.py",
    '''    if name == "nonrobust":\n        if scale is None:\n            raise ValueError("scale is required for nonrobust covariance")\n        scale_value = float(scale)\n        if not np.isfinite(scale_value) or scale_value < 0.0:\n            raise ValueError("scale must be finite and non-negative")\n        _X_work, X_pinv_work, design_scale, _rank = (\n            panel_svd_working_pseudoinverse(X, xp)\n        )\n        # scale * X+ X+^T = A A^T with\n        # A = sqrt(scale) * design_scale * X_work+.  Apply sqrt(scale) before\n        # restoring the potentially large design scale so a representable final\n        # covariance is not rejected merely because the raw bread overflows.\n        scaled_pinv = (X_pinv_work * float(np.sqrt(scale_value))) * design_scale\n        return _symmetrize(scaled_pinv @ scaled_pinv.T)\n''',
    '''    if name == "nonrobust":\n        if scale is None:\n            raise ValueError("scale is required for nonrobust covariance")\n        scale_value = float(scale)\n        if np.isnan(scale_value) or scale_value < 0.0:\n            raise ValueError("scale must be non-negative and not NaN")\n        _X_work, X_pinv_work, design_scale, _rank = (\n            panel_svd_working_pseudoinverse(X, xp)\n        )\n\n        suspicious_scale = (\n            not np.isfinite(scale_value)\n            or (0.0 < scale_value < np.finfo(np.float64).tiny)\n            or (scale_value == 0.0 and _to_float_scalar(xp.max(xp.abs(resid))) > 0.0)\n        )\n        if suspicious_scale:\n            if df_resid is None or int(df_resid) <= 0:\n                raise ValueError(\n                    "positive df_resid is required when nonrobust scale must be "\n                    "reconstructed from residuals"\n                )\n            resid_scale = xp.max(xp.abs(resid))\n            resid_scale_value = _to_float_scalar(resid_scale)\n            if resid_scale_value == 0.0:\n                scaled_pinv = X_pinv_work * 0.0\n            else:\n                resid_unit = resid / resid_scale\n                norm_sq = _to_float_scalar(xp.sum(resid_unit * resid_unit))\n                rms_unit = float(np.sqrt(norm_sq / float(int(df_resid))))\n                # sqrt(scale) = resid_scale * rms_unit.  Multiply the tiny/large\n                # residual scale into the working pseudoinverse before restoring\n                # design_scale; this lets opposite scales cancel while every\n                # intermediate remains representable whenever the final covariance is.\n                scaled_pinv = (\n                    ((X_pinv_work * resid_scale) * design_scale) * rms_unit\n                )\n            if metadata is not None:\n                metadata["nonrobust_scale_reconstructed"] = True\n                metadata["residual_scale"] = float(resid_scale_value)\n        else:\n            # scale * X+ X+^T = A A^T with\n            # A = sqrt(scale) * design_scale * X_work+.\n            scaled_pinv = (\n                X_pinv_work * float(np.sqrt(scale_value))\n            ) * design_scale\n        return _symmetrize(scaled_pinv @ scaled_pinv.T)\n''',
    "nonrobust reconstructed scale",
)

# ---- Diagnostics helpers for stable squared-scale restoration and adjusted R2.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''def _safe_r2(ss_res: float, ss_tot: float) -> Tuple[float, bool]:\n''',
    '''def _restore_squared_scale(value: float, scale: float) -> float:\n    """Restore ``value * scale**2`` without avoidable intermediate overflow."""\n    value = float(value)\n    scale = float(scale)\n    if value < 0.0 or scale < 0.0 or np.isnan(value) or np.isnan(scale):\n        return float("nan")\n    if value == 0.0 or scale == 0.0:\n        return 0.0\n    root = float(np.sqrt(value))\n    scaled_root = root * scale\n    return float(scaled_root * scaled_root)\n\n\ndef _common_scaled_sumsquares(left, right, xp):\n    """Return two sums of squares normalized by one common backend scale."""\n    scale = xp.maximum(xp.max(xp.abs(left)), xp.max(xp.abs(right)))\n    scale_value = _to_float_scalar(scale)\n    if scale_value == 0.0:\n        return 0.0, 0.0, 0.0\n    left_scaled = left / scale\n    right_scaled = right / scale\n    left_ss = _to_float_scalar(xp.sum(left_scaled * left_scaled))\n    right_ss = _to_float_scalar(xp.sum(right_scaled * right_scaled))\n    return float(left_ss), float(right_ss), float(scale_value)\n\n\ndef _safe_r2(ss_res: float, ss_tot: float) -> Tuple[float, bool]:\n''',
    "stable squared-scale helpers",
)

# Model-F already computes normalized RSS. Keep that numerical computation, but
# restore the pre-existing metadata keys to their original response-squared units.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''        metadata["rss_restricted"] = float(rss_r)\n        metadata["rss_unrestricted"] = float(rss_u)\n        metadata["rss_common_scale"] = float(common_scale_value)\n        metadata["rss_values_are_common_scale_normalized"] = True\n        return None, None, None, metadata\n''',
    '''        metadata["rss_restricted"] = _restore_squared_scale(\n            rss_r, common_scale_value\n        )\n        metadata["rss_unrestricted"] = _restore_squared_scale(\n            rss_u, common_scale_value\n        )\n        metadata["rss_restricted_normalized"] = float(rss_r)\n        metadata["rss_unrestricted_normalized"] = float(rss_u)\n        metadata["rss_common_scale"] = float(common_scale_value)\n        return None, None, None, metadata\n''',
    "model F unavailable metadata units",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    metadata["rss_restricted"] = float(rss_r)\n    metadata["rss_unrestricted"] = float(rss_u)\n    metadata["rss_common_scale"] = float(common_scale_value)\n    metadata["rss_values_are_common_scale_normalized"] = True\n    # The common-scale reduction already prevents underflow. A merely small\n''',
    '''    metadata["rss_restricted"] = _restore_squared_scale(\n        rss_r, common_scale_value\n    )\n    metadata["rss_unrestricted"] = _restore_squared_scale(\n        rss_u, common_scale_value\n    )\n    metadata["rss_restricted_normalized"] = float(rss_r)\n    metadata["rss_unrestricted_normalized"] = float(rss_u)\n    metadata["rss_common_scale"] = float(common_scale_value)\n    # The common-scale reduction already prevents underflow. A merely small\n''',
    "model F metadata units",
)

# Structured adjusted R2 must not consume raw RSS/TSS scalars that may already
# be Inf or zero from a quadratic reduction. Reconstruct the same fit-space
# residual/TSS ratio directly from arrays on one common scale.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    meta = {} if metadata is None else dict(metadata)\n    meta.setdefault("r2_definition", "parameter-based")\n''',
    '''    fit_y = y if f_y is None else f_y\n    fit_X = X if f_X is None else f_X\n    fit_params = params if f_params is None else f_params\n    fit_has_constant = (\n        bool(has_constant) if f_has_constant is None else bool(f_has_constant)\n    )\n    fit_resid = fit_y - fit_X @ fit_params.ravel()\n    fit_centered = (\n        fit_y - _scaled_mean(fit_y, xp) if fit_has_constant else fit_y\n    )\n    rss_adj, tss_adj, adjusted_scale = _common_scaled_sumsquares(\n        fit_resid, fit_centered, xp\n    )\n\n    meta = {} if metadata is None else dict(metadata)\n    meta.setdefault("r2_definition", "parameter-based")\n''',
    "adjusted R2 fit-space reduction",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    meta["model_f"] = f_meta\n    if entity_codes is None:\n''',
    '''    meta["model_f"] = f_meta\n    meta["adjusted_r2_common_scale"] = float(adjusted_scale)\n    meta["adjusted_r2_uses_common_scale"] = True\n    if entity_codes is None:\n''',
    "adjusted R2 metadata",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''        rsquared_adj=_adjusted_r2(\n            rss=float(rss_fit),\n            tss=float(tss_fit),\n            df_resid=int(df_resid),\n            df_total=int(df_total),\n        ),\n''',
    '''        rsquared_adj=_adjusted_r2(\n            rss=float(rss_adj),\n            tss=float(tss_adj),\n            df_resid=int(df_resid),\n            df_total=int(df_total),\n        ),\n''',
    "adjusted R2 normalized inputs",
)

# Let low-level pooling-F keep direct-call semantics while optionally recording
# a common scale for callers that normalize both nested residual series first.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''def _pooling_f_from_sums(\n    *,\n    rss_pooled: float,\n    rss_effects: float,\n    df_num: int,\n    df_denom: int,\n    metadata: Optional[Dict[str, Any]] = None,\n) -> PanelTestResult:\n''',
    '''def _pooling_f_from_sums(\n    *,\n    rss_pooled: float,\n    rss_effects: float,\n    df_num: int,\n    df_denom: int,\n    metadata: Optional[Dict[str, Any]] = None,\n    rss_common_scale: Optional[float] = None,\n) -> PanelTestResult:\n''',
    "pooling F scale signature",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    meta.update(\n        {\n            "rss_pooled": float(rss_pooled),\n            "rss_effects": float(rss_effects),\n            "classical_homoskedastic": True,\n        }\n    )\n''',
    '''    if rss_common_scale is None:\n        rss_pooled_public = float(rss_pooled)\n        rss_effects_public = float(rss_effects)\n    else:\n        rss_pooled_public = _restore_squared_scale(\n            rss_pooled, rss_common_scale\n        )\n        rss_effects_public = _restore_squared_scale(\n            rss_effects, rss_common_scale\n        )\n        meta["rss_pooled_normalized"] = float(rss_pooled)\n        meta["rss_effects_normalized"] = float(rss_effects)\n        meta["rss_common_scale"] = float(rss_common_scale)\n    meta.update(\n        {\n            "rss_pooled": float(rss_pooled_public),\n            "rss_effects": float(rss_effects_public),\n            "classical_homoskedastic": True,\n        }\n    )\n''',
    "pooling F public metadata units",
)

# ---- Integration diagnostics: normalize pooling F and BP-LM before quadratic
# accumulation. The statistics are scale invariant; metadata is restored safely.
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''from statgpu.panel._diagnostics import (\n    _applicable,\n    _build_fit_statistics,\n    _diagnostic_identity,\n    _inapplicable,\n    _matrix_rank,\n    _pooling_f_from_sums,\n)\n''',
    '''from statgpu.panel._diagnostics import (\n    _applicable,\n    _build_fit_statistics,\n    _diagnostic_identity,\n    _inapplicable,\n    _matrix_rank,\n    _pooling_f_from_sums,\n    _restore_squared_scale,\n)\n''',
    "diagnostic context scale helper import",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''    rss_effects: float,\n    df_resid_effects: int,\n    has_constant: bool = False,\n):\n''',
    '''    rss_effects: float,\n    df_resid_effects: int,\n    has_constant: bool = False,\n    resid_effects=None,\n):\n''',
    "pooling F residual signature",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''    resid_pool = y_pool - X_pool @ beta_pool\n    rss_pool = _to_float_scalar(xp.sum(resid_pool * resid_pool))\n    df_resid_pool = n - rank_pool - constant_projection_df\n    df_num = int(df_resid_pool) - int(df_resid_effects)\n    return _pooling_f_from_sums(\n        rss_pooled=float(rss_pool),\n        rss_effects=float(rss_effects),\n        df_num=int(df_num),\n        df_denom=int(df_resid_effects),\n        metadata={\n''',
    '''    resid_pool = y_pool - X_pool @ beta_pool\n    df_resid_pool = n - rank_pool - constant_projection_df\n    df_num = int(df_resid_pool) - int(df_resid_effects)\n    rss_common_scale = None\n    if resid_effects is None:\n        rss_pool = _to_float_scalar(xp.sum(resid_pool * resid_pool))\n        rss_effects_work = float(rss_effects)\n    else:\n        resid_effects = xp.asarray(resid_effects, dtype=xp.float64).ravel()\n        if int(resid_effects.shape[0]) != n:\n            raise ValueError("resid_effects must match the pooled sample length")\n        common_scale = xp.maximum(\n            xp.max(xp.abs(resid_pool)), xp.max(xp.abs(resid_effects))\n        )\n        rss_common_scale = _to_float_scalar(common_scale)\n        if rss_common_scale == 0.0:\n            rss_pool = 0.0\n            rss_effects_work = 0.0\n        else:\n            pool_unit = resid_pool / common_scale\n            effects_unit = resid_effects / common_scale\n            rss_pool = _to_float_scalar(xp.sum(pool_unit * pool_unit))\n            rss_effects_work = _to_float_scalar(\n                xp.sum(effects_unit * effects_unit)\n            )\n    return _pooling_f_from_sums(\n        rss_pooled=float(rss_pool),\n        rss_effects=float(rss_effects_work),\n        df_num=int(df_num),\n        df_denom=int(df_resid_effects),\n        rss_common_scale=rss_common_scale,\n        metadata={\n''',
    "pooling F normalized residual pair",
)

replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''    residual_ss = _to_float_scalar(xp.sum(resid * resid))\n    if residual_ss <= 0.0:\n        meta["residual_ss"] = float(residual_ss)\n''',
    '''    residual_scale = xp.max(xp.abs(resid))\n    residual_scale_value = _to_float_scalar(residual_scale)\n    if residual_scale_value == 0.0:\n        residual_ss = 0.0\n        meta["residual_ss"] = 0.0\n''',
    "BP exact-zero residual scale",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''            metadata=meta,\n        )\n\n    mean_aligned = group_means(resid, entity_codes, xp=xp)\n    sizes_aligned = group_sizes(entity_codes, xp=xp)\n''',
    '''            metadata=meta,\n        )\n\n    resid_work = resid / residual_scale\n    residual_ss = _to_float_scalar(xp.sum(resid_work * resid_work))\n    mean_aligned = group_means(resid_work, entity_codes, xp=xp)\n    sizes_aligned = group_sizes(entity_codes, xp=xp)\n''',
    "BP normalized residual work",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''        meta.update({"residual_ss": float(residual_ss), "M11": float(m11)})\n''',
    '''        meta.update(\n            {\n                "residual_ss": _restore_squared_scale(\n                    residual_ss, residual_scale_value\n                ),\n                "residual_ss_normalized": float(residual_ss),\n                "residual_scale": float(residual_scale_value),\n                "M11": float(m11),\n            }\n        )\n''',
    "BP repeated-observation metadata",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''            "residual_ss": float(residual_ss),\n            "A1": a1,\n''',
    '''            "residual_ss": _restore_squared_scale(\n                residual_ss, residual_scale_value\n            ),\n            "residual_ss_normalized": float(residual_ss),\n            "residual_scale": float(residual_scale_value),\n            "A1": a1,\n''',
    "BP success metadata",
)

# PanelOLS can provide the exact transformed residual vector to pooling-F, so
# no already-rounded RSS scalar is needed for the ratio.
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''                rss_effects=ss_res,\n                df_resid_effects=diagnostic_df["df_resid"],\n                has_constant=False,\n''',
    '''                rss_effects=ss_res,\n                df_resid_effects=diagnostic_df["df_resid"],\n                has_constant=False,\n                resid_effects=resid,\n''',
    "PanelOLS pooling residual integration",
)

# ---- Maintained regressions.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel._diagnostics import _classical_model_f\n",
    "from statgpu.panel._diagnostic_context import bp_lm_from_residuals\n"
    "from statgpu.panel._diagnostics import _build_fit_statistics, _classical_model_f\n",
    1,
)
marker = "def test_nonrobust_scale_underflow_recovers_representable_covariance():"
if marker in text:
    raise RuntimeError("reduction follow-up regressions already present")
text += r'''


def test_nonrobust_scale_underflow_recovers_representable_covariance():
    X = np.full((4, 1), 1.0e-200, dtype=np.float64)
    resid = np.asarray([1.0e-200, -1.0e-200, 1.0e-200, -1.0e-200])
    # Raw RSS/df underflows to zero, but variance * bread is 1/3.
    actual = ols_covariance(
        X,
        resid,
        cov_type="nonrobust",
        scale=0.0,
        df_resid=3,
    )
    assert_allclose(actual, np.asarray([[1.0 / 3.0]]), rtol=4e-13, atol=0.0)


def test_nonrobust_scale_overflow_recovers_representable_covariance():
    X = np.full((4, 1), 1.0e200, dtype=np.float64)
    resid = np.asarray([1.0e200, -1.0e200, 1.0e200, -1.0e200])
    # Raw RSS/df overflows, while the tiny bread makes final covariance 1/3.
    actual = ols_covariance(
        X,
        resid,
        cov_type="nonrobust",
        scale=float("inf"),
        df_resid=3,
    )
    assert_allclose(actual, np.asarray([[1.0 / 3.0]]), rtol=4e-13, atol=0.0)


def test_model_f_preserves_original_unit_rss_metadata_after_common_scaling():
    x = np.linspace(-1.0, 1.0, 12)
    X = np.column_stack([np.ones(x.size), x])
    y = 1.0e150 * (
        0.8
        + 0.45 * x
        + np.asarray([0.08, -0.04, 0.03, -0.06, 0.05, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.05])
    )
    params = panel_lstsq(X, y, np)[0]
    statistic, pvalue, _df, metadata = _classical_model_f(
        y,
        X,
        params,
        xp=np,
        df_resid=10,
        has_constant=True,
    )
    assert np.isfinite(statistic)
    assert np.isfinite(pvalue)
    assert metadata["rss_restricted"] > 1.0e298
    assert metadata["rss_unrestricted"] > 0.0
    assert metadata["rss_restricted_normalized"] < 20.0
    assert metadata["rss_unrestricted_normalized"] < 20.0


def test_adjusted_r2_uses_fit_space_common_scale_when_raw_sums_overflow():
    x = np.linspace(-1.0, 1.0, 12)
    X = np.column_stack([np.ones(x.size), x])
    base_y = 0.8 + 0.45 * x + np.asarray(
        [0.08, -0.04, 0.03, -0.06, 0.05, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.05]
    )
    params = panel_lstsq(X, base_y, np)[0]
    reference = _build_fit_statistics(
        base_y,
        X,
        params,
        xp=np,
        has_constant=True,
        rss_fit=float(np.sum((base_y - X @ params) ** 2)),
        tss_fit=float(np.sum((base_y - np.mean(base_y)) ** 2)),
        df_resid=10,
        df_total=11,
    )
    scale = 1.0e200
    y = scale * base_y
    params_big = scale * params
    huge = _build_fit_statistics(
        y,
        X,
        params_big,
        xp=np,
        has_constant=True,
        rss_fit=float("inf"),
        tss_fit=float("inf"),
        df_resid=10,
        df_total=11,
    )
    assert np.isfinite(huge.rsquared_adj)
    assert_allclose(huge.rsquared_adj, reference.rsquared_adj, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("scale", [1.0e150, 1.0e-150])
def test_bp_lm_is_invariant_when_raw_residual_ss_would_over_or_underflow(scale):
    entity = np.repeat(np.arange(4), 3)
    resid = np.asarray(
        [0.8, -0.2, 0.1, 0.5, -0.1, 0.4, -0.7, 0.3, -0.2, 0.2, -0.4, 0.6]
    )
    reference = bp_lm_from_residuals(resid, entity, xp=np)
    candidate = bp_lm_from_residuals(scale * resid, entity, xp=np)
    assert reference.applicable and candidate.applicable
    assert_allclose(candidate.statistic, reference.statistic, rtol=3e-13, atol=1e-14)
    assert_allclose(candidate.pvalue, reference.pvalue, rtol=3e-13, atol=1e-14)
    assert candidate.metadata["residual_ss_normalized"] > 0.0
'''
test_path.write_text(text, encoding="utf-8")

print("PR126 covariance reduction follow-up staged")
