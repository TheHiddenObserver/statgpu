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
    '''            else:\n                from statgpu.panel._diagnostics import _scaled_unit_values\n\n                resid_unit = _scaled_unit_values(resid, resid_scale, xp)\n                norm_sq = _to_float_scalar(xp.sum(resid_unit * resid_unit))\n                rms_unit = float(np.sqrt(norm_sq / float(int(df_resid))))\n                # sqrt(scale) = resid_scale * rms_unit. Apply the dimensionless\n                # RMS factor before the potentially large residual/design-scale\n                # restoration so a representable final covariance does not\n                # overflow an intermediate solely because of multiplication order.\n                scaled_pinv = (\n                    ((X_pinv_work * rms_unit) * resid_scale) * design_scale\n                )\n''',
    "nonrobust subnormal reconstruction",
)

# Regressions: prove Pooled no longer owns a backend-specific full-rank solve,
# plus Torch covariance where design/residual scales are subnormal but the final
# covariance is an ordinary representable number.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
marker = "def test_pooled_full_rank_fit_uses_shared_solver_ownership(monkeypatch):"
if marker in text:
    raise RuntimeError("solver covariance tail regressions already present")
text += r'''


def test_pooled_full_rank_fit_uses_shared_solver_ownership(monkeypatch):
    rng = np.random.default_rng(2026082001)
    X = rng.normal(size=(80, 2))
    y = 0.4 + X @ np.asarray([0.7, -0.25]) + rng.normal(scale=0.1, size=80)
    design = np.column_stack([np.ones(X.shape[0]), X])
    expected, expected_rank = panel_lstsq(design, y, np)

    def forbidden_backend_lstsq(*_args, **_kwargs):
        raise AssertionError("PooledOLS bypassed the shared panel_lstsq policy")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden_backend_lstsq)
    fit = PooledOLS(cov_type="hc0").fit(X, y)
    assert fit.rank_ == expected_rank
    assert_allclose(fit.coef_, expected, rtol=3e-12, atol=3e-12)


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
