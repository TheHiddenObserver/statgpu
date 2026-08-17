"""One-shot update for the stale full-rank PanelOLS solver-path regression."""

from pathlib import Path

path = Path("dev/tests/test_panel_stage_c_edge_contracts.py")
text = path.read_text()
old = '''def test_panel_full_rank_fit_preserves_historical_solver_path(monkeypatch):
    import statgpu.panel._fixed_effects as fixed_effects_module
    from statgpu.panel import PanelOLS

    rng = np.random.default_rng(12955)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([0.4, -0.2, 0.7]) + rng.normal(scale=0.1, size=80)
    expected = np.linalg.solve(X.T @ X, X.T @ y)

    def _forbid_rank_deficient_solve(*args, **kwargs):
        raise AssertionError("full-rank PanelOLS entered the SVD rank-deficient solve")

    monkeypatch.setattr(fixed_effects_module, "panel_lstsq", _forbid_rank_deficient_solve)
    model = PanelOLS().fit(X, y)
    np.testing.assert_allclose(model.coef_, expected, rtol=2e-12, atol=2e-14)
'''
new = '''def test_panel_full_rank_fit_uses_shared_svd_policy():
    from statgpu.panel import PanelOLS

    rng = np.random.default_rng(12955)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([0.4, -0.2, 0.7]) + rng.normal(scale=0.1, size=80)
    rcond = max(X.shape) * np.finfo(np.float64).eps
    expected = np.linalg.lstsq(X, y, rcond=rcond)[0]

    model = PanelOLS().fit(X, y)
    np.testing.assert_allclose(model.coef_, expected, rtol=2e-12, atol=2e-14)
    assert model._covariance_metadata["design_rank"] == X.shape[1]
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected one stale solver-path test, found {count}")
path.write_text(text.replace(old, new, 1))
