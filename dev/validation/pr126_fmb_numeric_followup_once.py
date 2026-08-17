from pathlib import Path

# Rewrite the first-pass helper's small-output finite checks so the existing
# direct-fit performance contract continues to forbid only repeated full X/y
# scans. Coefficient and covariance checks reduce tiny output arrays in place.
p = Path('dev/validation/pr126_fmb_numeric_stability_once.py')
text = p.read_text(encoding='utf-8')
text = text.replace(
    'if not _finite_all(avg_beta, xp):',
    'if not bool(_to_float_scalar(xp.all(xp.isfinite(avg_beta)))):',
)
text = text.replace(
    'if not _finite_all(beta_centered, xp):',
    'if not bool(_to_float_scalar(xp.all(xp.isfinite(beta_centered)))):',
)
text = text.replace(
    'if not _finite_all(cov_params, xp):',
    'if not bool(_to_float_scalar(xp.all(xp.isfinite(cov_params)))):',
)
p.write_text(text, encoding='utf-8')

# Parameter R^2 is scale invariant. Compute its RSS/TSS ratio after one common
# backend-native rescaling so finite large-level data cannot turn the public fit
# statistic into Inf/Inf -> NaN.
p = Path('statgpu/panel/_diagnostics.py')
text = p.read_text(encoding='utf-8')
anchor = '''def _safe_r2(ss_res: float, ss_tot: float) -> Tuple[float, bool]:
    """Return linearmodels-style parameter R² and a degenerate-TSS flag."""
    ss_res = float(ss_res)
    ss_tot = float(ss_tot)
    if ss_tot <= 0.0:
        return 0.0, True
    return 1.0 - ss_res / ss_tot, False


'''
addition = anchor + '''def _scaled_residual_r2(resid, centered, xp) -> Tuple[float, bool]:
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
    """Return a backend-native mean without overflowing the raw sum."""
    scale = xp.max(xp.abs(values))
    safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
    return xp.mean(values / safe_scale) * safe_scale


'''
if anchor not in text:
    raise RuntimeError('diagnostics R2 helper anchor not found')
text = text.replace(anchor, addition, 1)

old = '''    overall_resid = y - X @ params
    overall_center = y - xp.mean(y) if has_constant else y
    overall_ss_res = _to_float_scalar(xp.sum(overall_resid * overall_resid))
    overall_ss_tot = _to_float_scalar(xp.sum(overall_center * overall_center))
    overall, deg_o = _safe_r2(overall_ss_res, overall_ss_tot)
'''
new = '''    overall_resid = y - X @ params
    overall_center = y - _scaled_mean(y, xp) if has_constant else y
    overall, deg_o = _scaled_residual_r2(overall_resid, overall_center, xp)
'''
if old not in text:
    raise RuntimeError('overall R2 block anchor not found')
text = text.replace(old, new, 1)

old = '''    between_resid = y_between - X_between @ params
    between_center = y_between - xp.mean(y_between) if has_constant else y_between
    between_ss_res = _to_float_scalar(xp.sum(between_resid * between_resid))
    between_ss_tot = _to_float_scalar(xp.sum(between_center * between_center))
    between, deg_b = _safe_r2(between_ss_res, between_ss_tot)
'''
new = '''    between_resid = y_between - X_between @ params
    between_center = (
        y_between - _scaled_mean(y_between, xp) if has_constant else y_between
    )
    between, deg_b = _scaled_residual_r2(between_resid, between_center, xp)
'''
if old not in text:
    raise RuntimeError('between R2 block anchor not found')
text = text.replace(old, new, 1)

old = '''    within_resid = y_within - X_within @ params
    within_ss_res = _to_float_scalar(xp.sum(within_resid * within_resid))
    within_ss_tot = _to_float_scalar(xp.sum(y_within * y_within))
    within, deg_w = _safe_r2(within_ss_res, within_ss_tot)
'''
new = '''    within_resid = y_within - X_within @ params
    within, deg_w = _scaled_residual_r2(within_resid, y_within, xp)
'''
if old not in text:
    raise RuntimeError('within R2 block anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Strengthen the large finite-level regression to cover fit_statistics_.
p = Path('dev/tests/test_fama_macbeth_inference_matrix.py')
text = p.read_text(encoding='utf-8')
anchor = '''    assert np.all(np.isfinite(_to_numpy(actual.cov_params_)))
    np.testing.assert_allclose(
        _to_numpy(actual.betas_)[:, 0],
'''
replacement = '''    assert np.all(np.isfinite(_to_numpy(actual.cov_params_)))
    assert actual.fit_statistics_.rsquared_overall == 0.0
    assert actual.fit_statistics_.metadata["degenerate_total_ss"]["overall"] is True
    np.testing.assert_allclose(
        _to_numpy(actual.betas_)[:, 0],
'''
if anchor not in text:
    raise RuntimeError('large-level fit-stat regression anchor not found')
p.write_text(text.replace(anchor, replacement, 1), encoding='utf-8')
