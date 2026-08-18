from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Classical model-F is a ratio statistic. Evaluate both restricted and
# unrestricted residual sums of squares on one common scale, then use that
# scale-free pair for the nested-model comparison. This preserves tiny nonzero
# RSS and avoids Inf/Inf when both raw RSS values exceed float64 range.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    resid = y - X @ params.ravel()\n    rss_u = _to_float_scalar(xp.sum(resid * resid))\n    if restricted_X is not None:\n        beta_r, _ = panel_lstsq(restricted_X, y, xp)\n        resid_r = y - restricted_X @ beta_r\n    elif has_constant:\n        y_r = y - xp.mean(y)\n        rss_r = _to_float_scalar(xp.sum(y_r * y_r))\n    else:\n        rss_r = _to_float_scalar(xp.sum(y * y))\n\n    diff = rss_r - rss_u\n    tol = _relative_tolerance(rss_r, rss_u)\n''',
    '''    resid = y - X @ params.ravel()\n    if restricted_X is not None:\n        beta_r, _ = panel_lstsq(restricted_X, y, xp)\n        resid_r = y - restricted_X @ beta_r\n    elif has_constant:\n        resid_r = y - _scaled_mean(y, xp)\n    else:\n        resid_r = y\n\n    # F is invariant to a common positive residual scale.  Work with one shared\n    # scale so tiny nonzero unrestricted RSS is not rounded to zero and huge\n    # restricted/unrestricted RSS values do not become Inf/Inf before the ratio.\n    common_scale = xp.maximum(xp.max(xp.abs(resid)), xp.max(xp.abs(resid_r)))\n    common_scale_value = _to_float_scalar(common_scale)\n    if common_scale_value == 0.0:\n        rss_u = 0.0\n        rss_r = 0.0\n    else:\n        resid_u_scaled = resid / common_scale\n        resid_r_scaled = resid_r / common_scale\n        rss_u = _to_float_scalar(xp.sum(resid_u_scaled * resid_u_scaled))\n        rss_r = _to_float_scalar(xp.sum(resid_r_scaled * resid_r_scaled))\n\n    diff = rss_r - rss_u\n    tol = _relative_tolerance(rss_r, rss_u)\n''',
    "classical F scaled RSS",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''        metadata["rss_restricted"] = float(rss_r)\n        metadata["rss_unrestricted"] = float(rss_u)\n        return None, None, None, metadata\n''',
    '''        metadata["rss_restricted"] = float(rss_r)\n        metadata["rss_unrestricted"] = float(rss_u)\n        metadata["rss_common_scale"] = float(common_scale_value)\n        metadata["rss_values_are_common_scale_normalized"] = True\n        return None, None, None, metadata\n''',
    "classical F negative diff metadata",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    metadata["rss_restricted"] = float(rss_r)\n    metadata["rss_unrestricted"] = float(rss_u)\n    if rss_u <= tol:\n''',
    '''    metadata["rss_restricted"] = float(rss_r)\n    metadata["rss_unrestricted"] = float(rss_u)\n    metadata["rss_common_scale"] = float(common_scale_value)\n    metadata["rss_values_are_common_scale_normalized"] = True\n    if rss_u <= tol:\n''',
    "classical F common scale metadata",
)

# The pooling-F restricted fit is another panel least-squares problem. Never
# bypass the shared SVD rank/working-scale policy just because the design has
# already been classified as full rank.
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''    rank_pool = _matrix_rank(X_pool, xp)\n    if rank_pool < int(X_pool.shape[1]):\n        beta_pool, _ = panel_lstsq(X_pool, y_pool, xp)\n    else:\n        beta_pool = xp.linalg.pinv(X_pool) @ y_pool\n    resid_pool = y_pool - X_pool @ beta_pool\n''',
    '''    rank_pool = _matrix_rank(X_pool, xp)\n    beta_pool, _ = panel_lstsq(X_pool, y_pool, xp)\n    resid_pool = y_pool - X_pool @ beta_pool\n''',
    "pooling F shared solver",
)

print("PR126 scaled-RSS follow-up staged")
