from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Avoid evaluating max_abs / target on already-large designs. xp.where is eager
# on maintained backends, so form the ratio only from a bounded numerator.
replace_once(
    "statgpu/panel/_linalg.py",
    '''    def _factor(max_abs):\n        relative = max_abs / float(target)\n        safe_relative = xp.where(\n            relative > 0.0, relative, xp.ones_like(relative)\n        )\n        return xp.where(\n            (max_abs > 0.0) & (max_abs < target),\n            1.0 / safe_relative,\n            xp.ones_like(max_abs),\n        )\n''',
    '''    def _factor(max_abs):\n        needs_scale = (max_abs > 0.0) & (max_abs < target)\n        bounded_max = xp.where(\n            needs_scale, max_abs, xp.full_like(max_abs, float(target))\n        )\n        relative = bounded_max / float(target)\n        safe_relative = xp.where(\n            relative > 0.0, relative, xp.ones_like(relative)\n        )\n        return xp.where(\n            needs_scale,\n            1.0 / safe_relative,\n            xp.ones_like(max_abs),\n        )\n''',
    "bounded working-design ratio",
)

# If the variance itself is genuinely outside float64, return Inf deliberately
# instead of creating it through an overflowing multiplication warning.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    root = scale_value * float(\n        np.sqrt(norm_sq / float(int(df_resid)))\n    )\n    return float(np.float64(root) * np.float64(root))\n''',
    '''    root = scale_value * float(\n        np.sqrt(norm_sq / float(int(df_resid)))\n    )\n    if root > np.sqrt(np.finfo(np.float64).max):\n        return float("inf")\n    return float(root * root)\n''',
    "explicit unrepresentable variance",
)

# The structured fit-stat path now recomputes its scale-free RSS/TSS from arrays,
# but keep the legacy scalar arguments free of avoidable raw-square warnings.
replace_once(
    "statgpu/panel/_random_effects.py",
    '''        diagnostic_df_resid = n - rank_star\n        ss_res_diag = _to_float_scalar(xp.sum(resid_gls * resid_gls))\n\n        restricted_X = None\n''',
    '''        diagnostic_df_resid = n - rank_star\n        resid_gls_scale = xp.max(xp.abs(resid_gls))\n        resid_gls_scale_value = _to_float_scalar(resid_gls_scale)\n        if resid_gls_scale_value == 0.0:\n            ss_res_diag = 0.0\n        else:\n            resid_gls_unit = resid_gls / resid_gls_scale\n            ss_res_diag = _restore_squared_scale(\n                _to_float_scalar(xp.sum(resid_gls_unit * resid_gls_unit)),\n                resid_gls_scale_value,\n            )\n\n        restricted_X = None\n''',
    "RE stable diagnostic RSS",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    '''            restricted_resid = y_star - restricted_X @ restricted_params\n            ss_tot_diag = _to_float_scalar(\n                xp.sum(restricted_resid * restricted_resid)\n            )\n        else:\n            ss_tot_diag = _to_float_scalar(xp.sum(y_star * y_star))\n''',
    '''            restricted_resid = y_star - restricted_X @ restricted_params\n            restricted_scale = xp.max(xp.abs(restricted_resid))\n            restricted_scale_value = _to_float_scalar(restricted_scale)\n            if restricted_scale_value == 0.0:\n                ss_tot_diag = 0.0\n            else:\n                restricted_unit = restricted_resid / restricted_scale\n                ss_tot_diag = _restore_squared_scale(\n                    _to_float_scalar(xp.sum(restricted_unit * restricted_unit)),\n                    restricted_scale_value,\n                )\n        else:\n            y_star_scale = xp.max(xp.abs(y_star))\n            y_star_scale_value = _to_float_scalar(y_star_scale)\n            if y_star_scale_value == 0.0:\n                ss_tot_diag = 0.0\n            else:\n                y_star_unit = y_star / y_star_scale\n                ss_tot_diag = _restore_squared_scale(\n                    _to_float_scalar(xp.sum(y_star_unit * y_star_unit)),\n                    y_star_scale_value,\n                )\n''',
    "RE stable diagnostic TSS",
)

# New regression assertions should use the established panel inference surface.
replace_once(
    "dev/tests/test_panel_stage_c_final_review_fixes.py",
    '''    assert np.all(np.isfinite(candidate.cov_params_))\n\n\ndef test_random_effects_transformation_is_invariant_when_raw_auxiliary_rss_overflows():\n''',
    '''    assert np.all(np.isfinite(candidate._panel_cov_params_raw))\n\n\ndef test_random_effects_transformation_is_invariant_when_raw_auxiliary_rss_overflows():\n''',
    "Pooled covariance assertion",
)
replace_once(
    "dev/tests/test_panel_stage_c_final_review_fixes.py",
    '''    assert_allclose(candidate.coef_, reference.coef_, rtol=2e-11, atol=2e-12)\n    assert np.all(np.isfinite(candidate.cov_params_))\n''',
    '''    assert_allclose(candidate.coef_, reference.coef_, rtol=2e-11, atol=2e-12)\n    assert np.all(np.isfinite(candidate._panel_cov_params_raw))\n''',
    "RE covariance assertion",
)

print("PR126 estimator-scale gate fixups staged")
