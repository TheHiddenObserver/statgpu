from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# PooledOLS must not bypass the shared working-scale/rank policy after a separate
# full-rank classification. The shared solver already returns both params/rank.
replace_once(
    "statgpu/panel/_pooled.py",
    '''def _panel_lstsq(X, y, xp):\n    """Preserve the historical full-rank solver under an explicit rank policy."""\n    rank = panel_matrix_rank(X, xp)\n    if rank < int(X.shape[1]):\n        params, _ = panel_lstsq(X, y, xp)\n        return params, rank\n    if getattr(xp, "__name__", "") == "torch":\n        try:\n            return xp.linalg.pinv(X) @ y, rank\n        except RuntimeError:\n            params, _ = panel_lstsq(X, y, xp)\n            return params, rank\n    try:\n        return xp.linalg.lstsq(X, y, rcond=None)[0], rank\n    except (TypeError, AttributeError, np.linalg.LinAlgError):\n        params, _ = panel_lstsq(X, y, xp)\n        return params, rank\n''',
    '''def _panel_lstsq(X, y, xp):\n    """Use the shared panel SVD rank and working-scale policy."""\n    return panel_lstsq(X, y, xp)\n''',
    "Pooled shared solver",
)
# panel_matrix_rank becomes unused in pooled after the wrapper is unified.
replace_once(
    "statgpu/panel/_pooled.py",
    "from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank\n",
    "from statgpu.panel._linalg import panel_lstsq\n",
    "Pooled linalg import",
)

# Reuse the already-tested subnormal normalizer in the nonrobust recovery path.
replace_once(
    "statgpu/panel/_covariance.py",
    '''            else:\n                resid_unit = resid / resid_scale\n                norm_sq = _to_float_scalar(xp.sum(resid_unit * resid_unit))\n                rms_unit = float(np.sqrt(norm_sq / float(int(df_resid))))\n                # sqrt(scale) = resid_scale * rms_unit.  Multiply the tiny/large\n                # residual scale into the working pseudoinverse before restoring\n                # design_scale; this lets opposite scales cancel while every\n                # intermediate remains representable whenever the final covariance is.\n                scaled_pinv = (\n                    ((X_pinv_work * resid_scale) * design_scale) * rms_unit\n                )\n''',
    '''            else:\n                from statgpu.panel._diagnostics import _scaled_unit_values\n\n                resid_unit = _scaled_unit_values(resid, resid_scale, xp)\n                norm_sq = _to_float_scalar(xp.sum(resid_unit * resid_unit))\n                rms_unit = float(np.sqrt(norm_sq / float(int(df_resid))))\n                # sqrt(scale) = resid_scale * rms_unit. Apply the <=1 RMS factor\n                # before the potentially large residual/design restoration so a\n                # representable final covariance does not overflow an intermediate.\n                scaled_pinv = (\n                    ((X_pinv_work * rms_unit) * resid_scale) * design_scale\n                )\n''',
    "nonrobust subnormal reconstruction",
)

# Regressions: end-to-end Pooled tiny design and Torch covariance where both
# design and residual scales are subnormal but the final covariance is ordinary.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
marker = "def test_pooled_tiny_full_rank_design_uses_shared_working_scale():"
if marker in text:
    raise RuntimeError("solver covariance tail regressions already present")
text += r'''


def test_pooled_tiny_full_rank_design_uses_shared_working_scale():
    tiny = 1.0e-320
    X = tiny * np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0], [-1.0, 2.0], [2.0, 2.0]]
    )
    beta = np.asarray([1.5, -0.75])
    y = X @ beta
    fit = PooledOLS(cov_type="nonrobust").fit(X, y)
    assert_allclose(fit.coef_, beta, rtol=2e-3, atol=2e-3)
    assert np.all(np.isfinite(fit.coef_))


def test_torch_nonrobust_subnormal_design_and_residual_reconstruction():
    torch = pytest.importorskip("torch")
    X_np = np.full((4, 1), 1.0e-320, dtype=np.float64)
    resid_np = np.asarray([1.0e-320, -1.0e-320, 1.0e-320, -1.0e-320])
    expected = ols_covariance(
        X_np,
        resid_np,
        cov_type="nonrobust",
        scale=0.0,
        df_resid=3,
    )
    X = torch.as_tensor(X_np, dtype=torch.float64)
    resid = torch.as_tensor(resid_np, dtype=torch.float64)
    actual = ols_covariance(
        X,
        resid,
        cov_type="nonrobust",
        scale=0.0,
        df_resid=3,
        xp=torch,
    )
    assert torch.all(torch.isfinite(actual))
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=3e-11, atol=2e-12)
    assert_allclose(actual.detach().cpu().numpy(), np.asarray([[1.0 / 3.0]]), rtol=3e-3, atol=3e-3)
'''
test_path.write_text(text, encoding="utf-8")

print("PR126 solver/covariance tail review staged")
