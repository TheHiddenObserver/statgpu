from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Shared model-F normalization must use the same subnormal-safe scalar
# normalization already used by residual variance/R2 reductions.
replace_once(
    "statgpu/panel/_diagnostics.py",
    """        resid_u_scaled = resid / common_scale\n        resid_r_scaled = resid_r / common_scale\n""",
    """        resid_u_scaled = _scaled_unit_values(resid, common_scale, xp)\n        resid_r_scaled = _scaled_unit_values(resid_r, common_scale, xp)\n""",
    "classical F subnormal normalization",
)

# Diagnostic integration needs the same overflow-safe means and subnormal-safe
# normalization as the shared fit-statistic primitives.
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    """    _pooling_f_from_sums,\n    _restore_squared_scale,\n)\n""",
    """    _pooling_f_from_sums,\n    _restore_squared_scale,\n    _scaled_mean,\n    _scaled_unit_values,\n)\n""",
    "diagnostic context scaled helper imports",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    """def pooling_f_from_level_arrays(\n""",
    """def _scaled_column_means(values, xp):\n    \"\"\"Return column means without overflowing a finite reduction.\"\"\"\n    n = int(values.shape[0])\n    if getattr(xp, \"__name__\", \"\") == \"torch\":\n        max_abs = xp.max(xp.abs(values), dim=0).values\n    else:\n        max_abs = xp.max(xp.abs(values), axis=0)\n    limit = np.finfo(np.float64).max / float(max(n, 1))\n    factor = xp.where(\n        max_abs > limit,\n        xp.full_like(max_abs, float(n)),\n        xp.ones_like(max_abs),\n    )\n    if getattr(xp, \"__name__\", \"\") == \"torch\":\n        summed = xp.sum(values / factor, dim=0)\n    else:\n        summed = xp.sum(values / factor, axis=0)\n    return summed * (factor / float(n))\n\n\ndef pooling_f_from_level_arrays(\n""",
    "diagnostic scaled column means helper",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    """    else:\n        y_pool = y - xp.mean(y)\n        X_pool = X - xp.mean(X, axis=0)\n        constant_projection_df = 1\n""",
    """    else:\n        y_pool = y - _scaled_mean(y, xp)\n        X_pool = X - _scaled_column_means(X, xp)\n        constant_projection_df = 1\n""",
    "pooling F overflow-safe centering",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    """            pool_unit = resid_pool / common_scale\n            effects_unit = resid_effects / common_scale\n""",
    """            pool_unit = _scaled_unit_values(resid_pool, common_scale, xp)\n            effects_unit = _scaled_unit_values(resid_effects, common_scale, xp)\n""",
    "pooling F subnormal normalization",
)
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    """    resid_work = resid / residual_scale\n""",
    """    resid_work = _scaled_unit_values(resid, residual_scale, xp)\n""",
    "BP-LM subnormal normalization",
)

# Estimator-side RSS/TSS reconstruction is metadata/reporting work, but it must
# not reintroduce backend-specific subnormal division after the shared R2 path
# has already succeeded. Reuse _scaled_unit_values consistently.
for path in (
    "statgpu/panel/_pooled.py",
    "statgpu/panel/_between.py",
    "statgpu/panel/_first_diff.py",
    "statgpu/panel/_fixed_effects.py",
):
    replace_once(
        path,
        """            _scaled_residual_r2,\n            _scaled_residual_variance,\n""",
        """            _scaled_residual_r2,\n            _scaled_residual_variance,\n            _scaled_unit_values,\n""",
        f"{path} scaled-unit import",
    )

replace_once(
    "statgpu/panel/_pooled.py",
    """        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        centered_unit = y_centered / xp.where(\n            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)\n        )\n""",
    """        resid_unit = _scaled_unit_values(resid, resid_scale, xp)\n        centered_unit = _scaled_unit_values(y_centered, centered_scale, xp)\n""",
    "Pooled metadata subnormal normalization",
)
replace_once(
    "statgpu/panel/_between.py",
    """        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        centered_unit = y_centered / xp.where(\n            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)\n        )\n""",
    """        resid_unit = _scaled_unit_values(resid, resid_scale, xp)\n        centered_unit = _scaled_unit_values(y_centered, centered_scale, xp)\n""",
    "Between metadata subnormal normalization",
)
replace_once(
    "statgpu/panel/_first_diff.py",
    """        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        centered_unit = y_centered / xp.where(\n            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)\n        )\n""",
    """        resid_unit = _scaled_unit_values(resid, resid_scale, xp)\n        centered_unit = _scaled_unit_values(y_centered, centered_scale, xp)\n""",
    "FD metadata subnormal normalization",
)
replace_once(
    "statgpu/panel/_first_diff.py",
    """        diagnostic_df = n - rank_diff\n        ss_tot_diag = _to_float_scalar(xp.sum(y_diff * y_diff))\n""",
    """        diagnostic_df = n - rank_diff\n        y_diff_scale = xp.max(xp.abs(y_diff))\n        y_diff_unit = _scaled_unit_values(y_diff, y_diff_scale, xp)\n        ss_tot_diag = _restore_squared_scale(\n            _to_float_scalar(xp.sum(y_diff_unit * y_diff_unit)),\n            _to_float_scalar(y_diff_scale),\n        )\n""",
    "FD diagnostic TSS reconstruction",
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    """        resid_unit = resid / xp.where(\n            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)\n        )\n        total_unit = y_d_centered / xp.where(\n            total_scale > 0.0, total_scale, xp.ones_like(total_scale)\n        )\n""",
    """        resid_unit = _scaled_unit_values(resid, resid_scale, xp)\n        total_unit = _scaled_unit_values(y_d_centered, total_scale, xp)\n""",
    "Panel metadata subnormal normalization",
)

replace_once(
    "statgpu/panel/_random_effects.py",
    """            _common_scaled_sumsquares,\n            _restore_squared_scale,\n            _scaled_residual_variance,\n""",
    """            _common_scaled_sumsquares,\n            _restore_squared_scale,\n            _scaled_residual_variance,\n            _scaled_unit_values,\n""",
    "RE scaled-unit import",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """            resid_gls_unit = resid_gls / resid_gls_scale\n""",
    """            resid_gls_unit = _scaled_unit_values(\n                resid_gls, resid_gls_scale, xp\n            )\n""",
    "RE residual diagnostic normalization",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """                restricted_unit = restricted_resid / restricted_scale\n""",
    """                restricted_unit = _scaled_unit_values(\n                    restricted_resid, restricted_scale, xp\n                )\n""",
    "RE restricted diagnostic normalization",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """                y_star_unit = y_star / y_star_scale\n""",
    """                y_star_unit = _scaled_unit_values(\n                    y_star, y_star_scale, xp\n                )\n""",
    "RE no-intercept diagnostic normalization",
)

# Add focused regressions for the two newly found production paths. The pooling
# fixture scales both X and y so raw finite means overflow while the centered
# regression and F statistic remain representable and scale-invariant.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel._diagnostic_context import bp_lm_from_residuals\n",
    "from statgpu.panel._diagnostic_context import (\n"
    "    bp_lm_from_residuals,\n"
    "    pooling_f_from_level_arrays,\n"
    ")\n",
    1,
)
marker = "def test_pooling_f_large_finite_centering_is_scale_invariant():"
if marker in text:
    raise RuntimeError("fresh numerical closeout regressions already present")
text += r'''


def test_pooling_f_large_finite_centering_is_scale_invariant():
    n = 24
    t = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([1.15 + 0.18 * t, 0.95 - 0.11 * t + 0.03 * t * t])
    y = 1.05 + 0.42 * t - 0.08 * t * t + 0.025 * np.sin(np.arange(n))

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    beta, _ = panel_lstsq(Xc, yc, np)
    pooled_resid = yc - Xc @ beta
    effects_resid = 0.55 * pooled_resid

    reference = pooling_f_from_level_arrays(
        y,
        X,
        xp=np,
        rss_effects=0.0,
        df_resid_effects=n - 6,
        has_constant=False,
        resid_effects=effects_resid,
    )
    scale = 1.0e307
    candidate = pooling_f_from_level_arrays(
        scale * y,
        scale * X,
        xp=np,
        rss_effects=0.0,
        df_resid_effects=n - 6,
        has_constant=False,
        resid_effects=scale * effects_resid,
    )
    assert reference.applicable and candidate.applicable
    assert np.isfinite(reference.statistic) and np.isfinite(candidate.statistic)
    assert_allclose(candidate.statistic, reference.statistic, rtol=2e-10, atol=1e-12)
    assert_allclose(candidate.pvalue, reference.pvalue, rtol=2e-10, atol=1e-14)


def test_torch_subnormal_classical_f_matches_numpy():
    torch = pytest.importorskip("torch")
    x = np.linspace(-1.0, 1.0, 12)
    X_np = np.column_stack([np.ones(x.size), x])
    base = 0.8 + 0.45 * x + np.asarray(
        [0.08, -0.04, 0.03, -0.06, 0.05, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.05]
    )
    scale = 1.0e-310
    y_np = scale * base
    params_np, _ = panel_lstsq(X_np, y_np, np)
    reference = _classical_model_f(
        y_np,
        X_np,
        params_np,
        xp=np,
        df_resid=10,
        has_constant=True,
    )

    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    params = torch.as_tensor(params_np, dtype=torch.float64)
    candidate = _classical_model_f(
        y,
        X,
        params,
        xp=torch,
        df_resid=10,
        has_constant=True,
    )
    assert reference[0] is not None and candidate[0] is not None
    assert np.isfinite(reference[0]) and np.isfinite(candidate[0])
    assert_allclose(candidate[0], reference[0], rtol=2e-4, atol=2e-6)
    assert_allclose(candidate[1], reference[1], rtol=2e-4, atol=2e-8)


def test_torch_subnormal_bp_lm_matches_numpy():
    torch = pytest.importorskip("torch")
    groups = np.repeat(np.arange(5), 4)
    pattern = np.asarray(
        [1.0, -0.4, 0.6, -0.3, 0.8, -0.2, 0.5, -0.7, 1.1, -0.5,
         0.3, -0.1, 0.7, -0.6, 0.4, -0.2, 0.9, -0.3, 0.2, -0.4]
    )
    resid_np = 1.0e-310 * pattern
    reference = bp_lm_from_residuals(resid_np, groups, xp=np)
    candidate = bp_lm_from_residuals(
        torch.as_tensor(resid_np, dtype=torch.float64),
        torch.as_tensor(groups, dtype=torch.int64),
        xp=torch,
    )
    assert reference.applicable and candidate.applicable
    assert np.isfinite(reference.statistic) and np.isfinite(candidate.statistic)
    assert_allclose(candidate.statistic, reference.statistic, rtol=2e-10, atol=1e-12)
    assert_allclose(candidate.pvalue, reference.pvalue, rtol=2e-10, atol=1e-14)
'''
test_path.write_text(text, encoding="utf-8")

print("PR126 fresh numerical closeout staged")
