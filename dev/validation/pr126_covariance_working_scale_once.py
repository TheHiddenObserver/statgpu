from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Expose a stable working-design pseudoinverse factorization for inference.
replace_once(
    "statgpu/panel/_linalg.py",
    "\ndef panel_svd_pseudoinverse(X, xp):\n",
    '''\ndef panel_svd_working_pseudoinverse(X, xp):\n    """Return a safe working-design pseudoinverse and its rescaling factor.\n\n    The returned pseudoinverse belongs to ``X_work = design_scale * X``.\n    Consumers that combine the pseudoinverse with residual or covariance scale\n    can therefore apply ``design_scale`` only after those smaller quantities\n    have reduced the dynamic range, instead of materializing an unrepresentable\n    original-scale ``X+`` or ``X+ X+^T`` first.  The rank policy is identical to\n    :func:`panel_lstsq`.\n    """\n    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)\n    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)\n    X_pinv_work = (Vh.T * inverse_values) @ U.T\n    return X_work, X_pinv_work, design_scale, rank\n\n\ndef panel_svd_pseudoinverse(X, xp):\n''',
    "working pseudoinverse helper",
)

# 2) Build covariance influence/bread from the safe working design instead of
# materializing an overflowing original-scale pseudoinverse first.
replace_once(
    "statgpu/panel/_covariance.py",
    "from statgpu.panel._linalg import panel_svd_pseudoinverse\n",
    "from statgpu.panel._linalg import (\n    panel_svd_pseudoinverse,\n    panel_svd_working_pseudoinverse,\n)\n",
    "covariance linalg import",
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''def _influence_rows(X, resid, xp):\n    """Return observation OLS influence rows e_i * ((X+)')_i."""\n    X_pinv, bread, rank = _design_pseudoinverse(X, xp)\n    influence = X_pinv.T * resid[:, None]\n    return influence, X_pinv, bread, rank\n''',
    '''def _influence_rows(X, resid, xp):\n    """Return stable observation influence rows plus working projection factors."""\n    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(\n        X, xp\n    )\n    # X+ = design_scale * X_work+.  Multiply residuals before restoring the\n    # design scale so a tiny full-rank design does not overflow X+ even when the\n    # final influence and covariance remain representable.\n    influence = (X_pinv_work.T * resid[:, None]) * design_scale\n    return influence, X_pinv_work, X_work, rank\n''',
    "stable influence rows",
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    influence, X_pinv, _bread, rank = _influence_rows(X, resid, xp)\n    if kind == "hc0":\n''',
    '''    influence, X_pinv_work, X_work, rank = _influence_rows(X, resid, xp)\n    if kind == "hc0":\n''',
    "HC working factors",
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    projection_rows = X_pinv.T\n    if _is_torch(xp):\n        leverage = xp.sum(X * projection_rows, dim=1)\n    else:\n        leverage = xp.sum(X * projection_rows, axis=1)\n''',
    '''    projection_rows = X_pinv_work.T\n    # Leverage is invariant to the uniform working-design rescaling:\n    # X X+ = X_work X_work+.  Evaluate it entirely at the safe working scale.\n    if _is_torch(xp):\n        leverage = xp.sum(X_work * projection_rows, dim=1)\n    else:\n        leverage = xp.sum(X_work * projection_rows, axis=1)\n''',
    "HC leverage working scale",
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    if kind == "hc2":\n        adjusted_resid = resid / xp.sqrt(denominator)\n    elif kind == "hc3":\n        adjusted_resid = resid / denominator\n    else:\n        raise ValueError(f"unknown HC covariance kind {kind!r}")\n    adjusted_influence = projection_rows * adjusted_resid[:, None]\n''',
    '''    if kind == "hc2":\n        adjusted_influence = influence / xp.sqrt(denominator)[:, None]\n    elif kind == "hc3":\n        adjusted_influence = influence / denominator[:, None]\n    else:\n        raise ValueError(f"unknown HC covariance kind {kind!r}")\n''',
    "HC adjusted influence",
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    if name == "nonrobust":\n        if scale is None:\n            raise ValueError("scale is required for nonrobust covariance")\n        _X_pinv, bread, _rank = _design_pseudoinverse(X, xp)\n        return _symmetrize(float(scale) * bread)\n''',
    '''    if name == "nonrobust":\n        if scale is None:\n            raise ValueError("scale is required for nonrobust covariance")\n        scale_value = float(scale)\n        if not np.isfinite(scale_value) or scale_value < 0.0:\n            raise ValueError("scale must be finite and non-negative")\n        _X_work, X_pinv_work, design_scale, _rank = (\n            panel_svd_working_pseudoinverse(X, xp)\n        )\n        # scale * X+ X+^T = A A^T with\n        # A = sqrt(scale) * design_scale * X_work+.  Apply sqrt(scale) before\n        # restoring the potentially large design scale so a representable final\n        # covariance is not rejected merely because the raw bread overflows.\n        scaled_pinv = (X_pinv_work * float(np.sqrt(scale_value))) * design_scale\n        return _symmetrize(scaled_pinv @ scaled_pinv.T)\n''',
    "stable nonrobust bread",
)

# 3) Restricted model-F fits must use the same shared SVD/rank/working-scale
# policy as every other panel least-squares problem.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    if restricted_X is not None:\n        if rank_r < int(restricted_X.shape[1]):\n            beta_r, _ = panel_lstsq(restricted_X, y, xp)\n        else:\n            beta_r = xp.linalg.pinv(restricted_X) @ y\n        resid_r = y - restricted_X @ beta_r\n''',
    '''    if restricted_X is not None:\n        beta_r, _ = panel_lstsq(restricted_X, y, xp)\n        resid_r = y - restricted_X @ beta_r\n''',
    "restricted model F shared solver",
)

# 4) Maintained regressions live in a file already executed by the Stage-C
# Torch CPU workflow and the complete CPU tree.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
old_import = "from statgpu.panel._utils import demean_variables\n"
new_import = '''from statgpu.panel._covariance import clustered_covariance, ols_covariance\nfrom statgpu.panel._diagnostics import _classical_model_f\nfrom statgpu.panel._linalg import panel_lstsq\nfrom statgpu.panel._utils import demean_variables\n'''
if old_import not in text:
    raise RuntimeError("test import anchor missing")
text = text.replace(old_import, new_import, 1)
marker = "\n\ndef test_tiny_design_nonrobust_covariance_preserves_representable_final_scale():"
if marker in text:
    raise RuntimeError("working-scale regressions already present")
text += r'''


def test_tiny_design_nonrobust_covariance_preserves_representable_final_scale():
    # X+X+^T is about 2.5e399 and cannot be materialized in float64, but
    # scale * bread is only 2.5e299 and is a valid public covariance.
    X = np.full((4, 1), 1.0e-200, dtype=np.float64)
    resid = np.zeros(4, dtype=np.float64)
    actual = ols_covariance(
        X,
        resid,
        cov_type="nonrobust",
        scale=1.0e-100,
    )
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, np.asarray([[2.5e299]]), rtol=8e-14, atol=0.0)


@pytest.mark.parametrize(
    ("cov_type", "multiplier"),
    [("hc0", 1.0), ("hc2", 1.0 / 0.75), ("hc3", 1.0 / (0.75 * 0.75))],
)
def test_subnormal_design_hc_covariance_avoids_raw_pseudoinverse_overflow(
    cov_type, multiplier
):
    # The original-scale X+ is above DBL_MAX, while each final influence row is
    # about 2.5e149 and the covariance remains representable.
    X = np.full((4, 1), 1.0e-310, dtype=np.float64)
    resid = np.asarray([1.0e-160, -1.0e-160, 1.0e-160, -1.0e-160])
    actual = ols_covariance(X, resid, cov_type=cov_type)
    expected = 2.5e299 * multiplier
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, np.asarray([[expected]]), rtol=2e-13, atol=0.0)


def test_subnormal_design_clustered_covariance_uses_stable_influence_rows():
    X = np.full((4, 1), 1.0e-310, dtype=np.float64)
    resid = np.asarray([1.0e-160, 1.0e-160, -1.0e-160, -1.0e-160])
    cluster = np.asarray([0, 0, 1, 1], dtype=np.int64)
    actual = clustered_covariance(X, resid, cluster)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, np.asarray([[5.0e299]]), rtol=2e-13, atol=0.0)


def test_subnormal_design_hc0_torch_cpu_matches_numpy_working_scale():
    torch = pytest.importorskip("torch")
    X_np = np.full((4, 1), 1.0e-310, dtype=np.float64)
    resid_np = np.asarray([1.0e-160, -1.0e-160, 1.0e-160, -1.0e-160])
    expected = ols_covariance(X_np, resid_np, cov_type="hc0")
    X = torch.as_tensor(X_np, dtype=torch.float64)
    resid = torch.as_tensor(resid_np, dtype=torch.float64)
    actual = ols_covariance(X, resid, cov_type="hc0", xp=torch)
    assert torch.all(torch.isfinite(actual))
    assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=3e-12, atol=0.0
    )


def test_classical_model_f_restricted_fit_uses_shared_tiny_design_solver():
    tiny = 1.0e-310
    z = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    X = tiny * np.column_stack([np.ones(z.size), z])
    beta = np.asarray([1.0e160, 2.0e160], dtype=np.float64)
    noise = 1.0e-160 * np.asarray([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
    y = X @ beta + noise
    params, rank = panel_lstsq(X, y, np)
    assert rank == 2
    restricted = X[:, :1]
    statistic, pvalue, df, metadata = _classical_model_f(
        y,
        X,
        params,
        xp=np,
        df_resid=6,
        has_constant=False,
        restricted_X=restricted,
    )
    assert statistic is not None and np.isfinite(statistic) and statistic > 0.0
    assert pvalue is not None and np.isfinite(pvalue)
    assert df == (1.0, 6.0)
    assert np.isfinite(metadata["rss_restricted"])
    assert np.isfinite(metadata["rss_unrestricted"])
'''
test_path.write_text(text, encoding="utf-8")

print("PR126 covariance working-scale review fix staged")
