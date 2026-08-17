"""One-shot fix for saturated connected two-way prediction fixtures."""
from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    p.write_text(text.replace(old, new, 1))


old_test = '''def test_connected_two_way_prediction_rejects_one_sided_or_known_unknown_effects():
    rng = np.random.default_rng(2026082401)
    entity = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    time = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    X = rng.normal(size=(entity.size, 1))
    alpha = np.array([0.4, -0.3, 0.8])
    tau = np.array([0.25, -0.15, 0.45])
    y = 0.75 * X[:, 0] + alpha[entity] + tau[time]
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert model.fit_statistics_.metadata["diagnostic_df"]["incidence_components"] == 1
'''
new_test = '''def test_connected_two_way_prediction_rejects_one_sided_or_known_unknown_effects():
    rng = np.random.default_rng(2026082401)
    # Keep this prediction-contract fixture away from a saturated FE model.
    # With only the six cycle edges, N=T=3 plus one slope leaves zero standard
    # residual df; the historical N-1/T-1 shortcut accidentally hid that fact.
    entity = np.repeat(np.arange(3, dtype=np.int64), 3)
    time = np.tile(np.arange(3, dtype=np.int64), 3)
    X = rng.normal(size=(entity.size, 1))
    alpha = np.array([0.4, -0.3, 0.8])
    tau = np.array([0.25, -0.15, 0.45])
    y = 0.75 * X[:, 0] + alpha[entity] + tau[time]
    y = y + rng.normal(scale=0.02, size=entity.size)
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert model.df_resid > 0
    assert model.fit_statistics_.metadata["diagnostic_df"]["incidence_components"] == 1
'''
replace_once("dev/tests/test_panel_stage_c_final_review_fixes.py", old_test, new_test)

old_bench = '''def _connected_two_way_prediction_audit(backend):
    """Audit two-way normalization guards on a connected incidence graph."""
    rng = np.random.default_rng(20260824)
    entity = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    time = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    X = rng.normal(size=(entity.size, 1)).astype(np.float64)
    alpha = np.array([0.4, -0.3, 0.8], dtype=np.float64)
    tau = np.array([0.25, -0.15, 0.45], dtype=np.float64)
    y = (0.75 * X[:, 0] + alpha[entity] + tau[time]).astype(np.float64)
    Xb, yb, eb, tb = _to_backend(X, y, entity, time, backend)
    model = PanelOLS(
        entity_effects=True, time_effects=True, cov_type="hc0", device=_device(backend)
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
    if _backend_name(model) != backend:
        raise AssertionError("connected prediction audit fit backend provenance drifted")
    diagnostic = model.fit_statistics_.metadata.get("diagnostic_df", {})
    if int(diagnostic.get("incidence_components", -1)) != 1:
        raise AssertionError("connected prediction audit fixture is not connected")
'''
new_bench = '''def _connected_two_way_prediction_audit(backend):
    """Audit two-way normalization guards on a connected incidence graph."""
    rng = np.random.default_rng(20260824)
    # Use a connected, overidentified fixture. The former six-edge cycle was
    # saturated once standard FE nuisance rank was counted correctly (df=0).
    entity = np.repeat(np.arange(3, dtype=np.int64), 3)
    time = np.tile(np.arange(3, dtype=np.int64), 3)
    X = rng.normal(size=(entity.size, 1)).astype(np.float64)
    alpha = np.array([0.4, -0.3, 0.8], dtype=np.float64)
    tau = np.array([0.25, -0.15, 0.45], dtype=np.float64)
    y = 0.75 * X[:, 0] + alpha[entity] + tau[time]
    y = (y + rng.normal(scale=0.02, size=entity.size)).astype(np.float64)
    Xb, yb, eb, tb = _to_backend(X, y, entity, time, backend)
    model = PanelOLS(
        entity_effects=True, time_effects=True, cov_type="hc0", device=_device(backend)
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
    if _backend_name(model) != backend:
        raise AssertionError("connected prediction audit fit backend provenance drifted")
    if int(model.df_resid) <= 0:
        raise AssertionError("connected prediction audit fixture has no residual df")
    diagnostic = model.fit_statistics_.metadata.get("diagnostic_df", {})
    if int(diagnostic.get("incidence_components", -1)) != 1:
        raise AssertionError("connected prediction audit fixture is not connected")
'''
replace_once("dev/benchmarks/validate_panel_stage_c_gpu.py", old_bench, new_bench)
