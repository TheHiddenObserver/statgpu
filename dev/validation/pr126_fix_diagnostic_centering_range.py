from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Shared diagnostic centering primitive. Ordinary-scale subtraction stays exact;
# only a mathematically finite mean difference that cannot be represented on the
# original level scale switches to dimensionless working coordinates.
p = Path("statgpu/panel/_diagnostics.py")
text = p.read_text(encoding="utf-8")
anchor = '''def _scaled_group_means(values, groups, xp):\n'''
helper = '''def _centered_working_values(values, xp):\n    """Center ``values`` without materializing an overflowing level difference.\n\n    Returns ``(centered, scale)`` where ``scale is None`` preserves the ordinary\n    physical-scale path.  Only when ``max|value| + |mean|`` can exceed DBL_MAX do\n    we center normalized values; callers of this helper use scale-invariant RSS\n    or R-squared reductions and can put comparison residuals on the same scale.\n    """\n    mean = _scaled_mean(values, xp)\n    maximum = xp.max(xp.abs(values))\n    safe_limit = float(np.finfo(np.float64).max) - xp.abs(mean)\n    if bool(_to_float_scalar(maximum <= safe_limit)):\n        return values - mean, None\n    if _to_float_scalar(maximum) == 0.0:\n        return values, None\n    unit = _scaled_unit_values(values, maximum, xp)\n    return unit - _scaled_mean(unit, xp), maximum\n\n\ndef _residual_on_centering_scale(resid, scale, xp):\n    if scale is None:\n        return resid\n    return _scaled_unit_values(resid, scale, xp)\n\n\ndef _physical_common_scale(common_scale, centering_scale) -> float:\n    value = _to_float_scalar(common_scale)\n    if centering_scale is None:\n        return float(value)\n    level = _to_float_scalar(centering_scale)\n    if value == 0.0 or level == 0.0:\n        return 0.0\n    if value > float(np.finfo(np.float64).max) / level:\n        return float("inf")\n    return float(value * level)\n\n\n'''
if helper not in text:
    if anchor not in text:
        raise RuntimeError("diagnostic helper anchor not found")
    text = text.replace(anchor, helper + anchor, 1)

# Parameter-based overall/between R2: compare residuals to centered response in
# the same working units when physical centering would overflow.
old = '''    overall_resid = y - X @ params\n    overall_center = y - _scaled_mean(y, xp) if has_constant else y\n    overall, deg_o = _scaled_residual_r2(overall_resid, overall_center, xp)\n'''
new = '''    overall_resid = y - X @ params\n    if has_constant:\n        overall_center, overall_center_scale = _centered_working_values(y, xp)\n        overall_resid_work = _residual_on_centering_scale(\n            overall_resid, overall_center_scale, xp\n        )\n    else:\n        overall_center = y\n        overall_resid_work = overall_resid\n    overall, deg_o = _scaled_residual_r2(\n        overall_resid_work, overall_center, xp\n    )\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("overall R2 centering anchor not found")
    text = text.replace(old, new, 1)
old = '''    between_resid = y_between - X_between @ params\n    between_center = (\n        y_between - _scaled_mean(y_between, xp) if has_constant else y_between\n    )\n    between, deg_b = _scaled_residual_r2(between_resid, between_center, xp)\n'''
new = '''    between_resid = y_between - X_between @ params\n    if has_constant:\n        between_center, between_center_scale = _centered_working_values(\n            y_between, xp\n        )\n        between_resid_work = _residual_on_centering_scale(\n            between_resid, between_center_scale, xp\n        )\n    else:\n        between_center = y_between\n        between_resid_work = between_resid\n    between, deg_b = _scaled_residual_r2(\n        between_resid_work, between_center, xp\n    )\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("between R2 centering anchor not found")
    text = text.replace(old, new, 1)

# Classical constant-only restricted model. Keep unrestricted and restricted
# residuals on one response scale before the common RSS normalization.
old = '''    resid = y - X @ params.ravel()\n    if restricted_X is not None:\n        beta_r, _ = panel_lstsq(restricted_X, y, xp)\n        resid_r = y - restricted_X @ beta_r\n    elif has_constant:\n        resid_r = y - _scaled_mean(y, xp)\n    else:\n        resid_r = y\n\n    # F is invariant to a common positive residual scale. Work with one shared\n'''
new = '''    resid = y - X @ params.ravel()\n    centering_scale = None\n    if restricted_X is not None:\n        beta_r, _ = panel_lstsq(restricted_X, y, xp)\n        resid_r = y - restricted_X @ beta_r\n    elif has_constant:\n        resid_r, centering_scale = _centered_working_values(y, xp)\n        resid = _residual_on_centering_scale(resid, centering_scale, xp)\n    else:\n        resid_r = y\n\n    # F is invariant to a common positive residual scale. Work with one shared\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("classical F centering anchor not found")
    text = text.replace(old, new, 1)
old = '''    common_scale_value = _to_float_scalar(common_scale)\n'''
new = '''    common_scale_value = _physical_common_scale(\n        common_scale, centering_scale\n    )\n'''
# This exact line appears only in _classical_model_f before current edit context.
idx = text.find(old, text.find("def _classical_model_f"))
if idx < 0:
    raise RuntimeError("classical F common scale anchor not found")
if new not in text[idx:idx + len(old) + len(new) + 40]:
    text = text[:idx] + new + text[idx + len(old):]

# Adjusted R2 uses the same response centering contract.
old = '''    fit_resid = fit_y - fit_X @ fit_params.ravel()\n    fit_centered = (\n        fit_y - _scaled_mean(fit_y, xp) if fit_has_constant else fit_y\n    )\n    rss_adj, tss_adj, adjusted_scale = _common_scaled_sumsquares(\n        fit_resid, fit_centered, xp\n    )\n'''
new = '''    fit_resid = fit_y - fit_X @ fit_params.ravel()\n    if fit_has_constant:\n        fit_centered, fit_centering_scale = _centered_working_values(\n            fit_y, xp\n        )\n        fit_resid_work = _residual_on_centering_scale(\n            fit_resid, fit_centering_scale, xp\n        )\n    else:\n        fit_centered = fit_y\n        fit_centering_scale = None\n        fit_resid_work = fit_resid\n    rss_adj, tss_adj, adjusted_scale_work = _common_scaled_sumsquares(\n        fit_resid_work, fit_centered, xp\n    )\n    adjusted_scale = _physical_common_scale(\n        adjusted_scale_work, fit_centering_scale\n    )\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("adjusted R2 centering anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Pooling-F integration: y may need dimensionless centering. X retains the
# already-shared stable column means; only columns whose physical subtraction
# would overflow are normalized independently, which preserves their span.
p = Path("statgpu/panel/_diagnostic_context.py")
text = p.read_text(encoding="utf-8")
old_import = '''    _build_fit_statistics,\n    _diagnostic_identity,\n'''
new_import = '''    _build_fit_statistics,\n    _centered_working_values,\n    _diagnostic_identity,\n'''
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("diagnostic context import anchor not found")
    text = text.replace(old_import, new_import, 1)
old = '''    if has_constant:\n        y_pool = y\n        X_pool = X\n        constant_projection_df = 0\n    else:\n        y_pool = y - _scaled_mean(y, xp)\n        X_pool = X - _scaled_column_means(X, xp)\n        constant_projection_df = 1\n\n    rank_pool = _matrix_rank(X_pool, xp)\n'''
new = '''    y_centering_scale = None\n    if has_constant:\n        y_pool = y\n        X_pool = X\n        constant_projection_df = 0\n    else:\n        y_pool, y_centering_scale = _centered_working_values(y, xp)\n        X_pool = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()\n        for j in range(int(X.shape[1])):\n            centered_column, _column_scale = _centered_working_values(\n                X[:, j], xp\n            )\n            X_pool[:, j] = centered_column\n        constant_projection_df = 1\n\n    rank_pool = _matrix_rank(X_pool, xp)\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("pooling F centering anchor not found")
    text = text.replace(old, new, 1)
old = '''    if resid_effects is None:\n        rss_pool = _to_float_scalar(xp.sum(resid_pool * resid_pool))\n        rss_effects_work = float(rss_effects)\n    else:\n        resid_effects = xp.asarray(resid_effects, dtype=xp.float64).ravel()\n        if int(resid_effects.shape[0]) != n:\n            raise ValueError("resid_effects must match the pooled sample length")\n        common_scale = xp.maximum(\n            xp.max(xp.abs(resid_pool)), xp.max(xp.abs(resid_effects))\n        )\n        rss_common_scale = _to_float_scalar(common_scale)\n        if rss_common_scale == 0.0:\n            rss_pool = 0.0\n            rss_effects_work = 0.0\n        else:\n            pool_unit = _scaled_unit_values(resid_pool, common_scale, xp)\n            effects_unit = _scaled_unit_values(resid_effects, common_scale, xp)\n            rss_pool = _to_float_scalar(xp.sum(pool_unit * pool_unit))\n            rss_effects_work = _to_float_scalar(\n                xp.sum(effects_unit * effects_unit)\n            )\n'''
new = '''    if resid_effects is None:\n        rss_pool = _to_float_scalar(xp.sum(resid_pool * resid_pool))\n        if y_centering_scale is None:\n            rss_effects_work = float(rss_effects)\n        else:\n            if not np.isfinite(float(rss_effects)):\n                raise FloatingPointError(\n                    "extreme-scale pooling F requires residual-level fixed-effect "\n                    "values when the physical RSS is non-finite"\n                )\n            level = _to_float_scalar(y_centering_scale)\n            root = float(np.sqrt(max(float(rss_effects), 0.0)))\n            rss_effects_work = float((root / level) ** 2)\n            rss_common_scale = level\n    else:\n        resid_effects = xp.asarray(resid_effects, dtype=xp.float64).ravel()\n        if int(resid_effects.shape[0]) != n:\n            raise ValueError("resid_effects must match the pooled sample length")\n        if y_centering_scale is not None:\n            resid_effects = _scaled_unit_values(\n                resid_effects, y_centering_scale, xp\n            )\n        common_scale = xp.maximum(\n            xp.max(xp.abs(resid_pool)), xp.max(xp.abs(resid_effects))\n        )\n        common_scale_work = _to_float_scalar(common_scale)\n        if y_centering_scale is None:\n            rss_common_scale = common_scale_work\n        else:\n            level = _to_float_scalar(y_centering_scale)\n            rss_common_scale = (\n                float("inf")\n                if common_scale_work > float(np.finfo(np.float64).max) / level\n                else float(common_scale_work * level)\n            )\n        if common_scale_work == 0.0:\n            rss_pool = 0.0\n            rss_effects_work = 0.0\n        else:\n            pool_unit = _scaled_unit_values(resid_pool, common_scale, xp)\n            effects_unit = _scaled_unit_values(resid_effects, common_scale, xp)\n            rss_pool = _to_float_scalar(xp.sum(pool_unit * pool_unit))\n            rss_effects_work = _to_float_scalar(\n                xp.sum(effects_unit * effects_unit)\n            )\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("pooling F RSS anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# CPU/Torch regressions cover both the public pooling F integration and the
# shared fit-statistics path where constant centering used to overflow.
p = Path("dev/tests/test_panel_diagnostic_cancellation_precision.py")
text = p.read_text(encoding="utf-8")
old_import = '''from statgpu.panel._diagnostic_context import _scaled_column_means\n'''
new_import = '''from statgpu.panel._diagnostic_context import (\n    _scaled_column_means,\n    pooling_f_from_level_arrays,\n)\n'''
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("diagnostic precision context import anchor not found")
    text = text.replace(old_import, new_import, 1)
old_import = '''from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean\n'''
new_import = '''from statgpu.panel._diagnostics import (\n    _build_fit_statistics,\n    _scaled_group_means,\n    _scaled_mean,\n)\n'''
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("diagnostic precision diagnostics import anchor not found")
    text = text.replace(old_import, new_import, 1)
marker = "def test_extreme_constant_centering_keeps_diagnostics_on_finite_working_scale():"
if marker not in text:
    text += '''\n\n\ndef _extreme_centering_fixture(xp):\n    y_np = np.asarray([1.0e308] + [-1.0e308] * 10, dtype=np.float64)\n    z_np = np.asarray([1.0] + [-1.0] * 10, dtype=np.float64)\n    X_np = np.column_stack([np.ones(11), z_np])\n    params_np = np.asarray([0.0, 1.0e308], dtype=np.float64)\n    if xp is np:\n        return y_np, X_np, params_np\n    return (\n        xp.as_tensor(y_np, dtype=xp.float64),\n        xp.as_tensor(X_np, dtype=xp.float64),\n        xp.as_tensor(params_np, dtype=xp.float64),\n    )\n\n\n@pytest.mark.parametrize("backend", ["numpy", "torch"])\ndef test_extreme_constant_centering_keeps_diagnostics_on_finite_working_scale(backend):\n    xp = np if backend == "numpy" else pytest.importorskip("torch")\n    y, X, params = _extreme_centering_fixture(xp)\n    result = _build_fit_statistics(\n        y, X, params, xp=xp, entity_codes=None, has_constant=True,\n        rss_fit=0.0, tss_fit=float("inf"), df_resid=9, df_total=10,\n    )\n    assert result.rsquared_overall == 1.0\n    assert result.rsquared_adj == 1.0\n    assert np.isposinf(result.f_statistic)\n    assert result.f_pvalue == 0.0\n\n\n@pytest.mark.parametrize("backend", ["numpy", "torch"])\ndef test_extreme_pooling_f_centering_avoids_level_overflow(backend):\n    xp = np if backend == "numpy" else pytest.importorskip("torch")\n    y_np = np.asarray([1.0e308] + [-1.0e308] * 10, dtype=np.float64)\n    X_np = np.linspace(-1.0, 1.0, 11, dtype=np.float64)[:, None]\n    if xp is np:\n        y, X, resid_effects = y_np, X_np, np.zeros(11, dtype=np.float64)\n    else:\n        y = xp.as_tensor(y_np, dtype=xp.float64)\n        X = xp.as_tensor(X_np, dtype=xp.float64)\n        resid_effects = xp.zeros(11, dtype=xp.float64)\n    result = pooling_f_from_level_arrays(\n        y, X, xp=xp, rss_effects=0.0, df_resid_effects=8,\n        has_constant=False, resid_effects=resid_effects,\n    )\n    assert result.applicable\n    assert np.isposinf(result.statistic)\n    assert result.pvalue == 0.0\n'''
    p.write_text(text, encoding="utf-8")

# Physical runner locks the same two scale-invariant diagnostics on CuPy/Torch.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
audit_anchor = '''    # Pooling F: naive column/scalar means overflow after this common scaling,\n'''
audit_insert = '''    # Constant centering itself can exceed DBL_MAX although the F/R2 statistics\n    # are scale invariant. Keep the restricted response in safe working units.\n    extreme_y_np = np.asarray([1.0e308] + [-1.0e308] * 10, dtype=np.float64)\n    extreme_x_np = np.linspace(-1.0, 1.0, 11, dtype=np.float64)[:, None]\n    if backend == "cupy":\n        extreme_y = xp.asarray(extreme_y_np)\n        extreme_x = xp.asarray(extreme_x_np)\n        extreme_resid = xp.zeros(11, dtype=xp.float64)\n    elif backend == "torch":\n        extreme_y = xp.as_tensor(extreme_y_np, dtype=xp.float64, device="cuda")\n        extreme_x = xp.as_tensor(extreme_x_np, dtype=xp.float64, device="cuda")\n        extreme_resid = xp.zeros(11, dtype=xp.float64, device="cuda")\n    else:\n        extreme_y, extreme_x = extreme_y_np, extreme_x_np\n        extreme_resid = np.zeros(11, dtype=np.float64)\n    extreme_pooling = pooling_f_from_level_arrays(\n        extreme_y, extreme_x, xp=xp, rss_effects=0.0,\n        df_resid_effects=8, has_constant=False, resid_effects=extreme_resid,\n    )\n    if not extreme_pooling.applicable or not np.isposinf(extreme_pooling.statistic):\n        raise AssertionError(f"{backend}: extreme pooling-F centering did not stay valid")\n\n    # Pooling F: naive column/scalar means overflow after this common scaling,\n'''
if audit_insert not in text:
    if audit_anchor not in text:
        raise RuntimeError("physical diagnostic centering anchor not found")
    text = text.replace(audit_anchor, audit_insert, 1)
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_runner_covers_extreme_diagnostic_centering():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_covers_extreme_diagnostic_centering():\n    source = inspect.getsource(_MOD._diagnostic_scale_audit)\n    for token in ("1.0e308", "extreme_pooling", "extreme pooling-F centering"):\n        assert token in source\n'''
    p.write_text(text, encoding="utf-8")

# Changelog note for the diagnostic working-centering boundary.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
needle = "- **Panel diagnostics extreme-scale correctness**:"
idx = text.find(needle)
if idx < 0:
    raise RuntimeError("CHANGELOG diagnostic bullet not found")
line_end = text.find("\n", idx)
line = text[idx:line_end]
if "range-safe response centering" not in line:
    line += " Constant-only restricted diagnostics and parameter/adjusted R-squared now use range-safe response centering only when the physical value-minus-mean difference would exceed float64 range, keeping scale-invariant statistics finite without perturbing ordinary-scale subtraction."
    text = text[:idx] + line + text[line_end:]
    p.write_text(text, encoding="utf-8")
