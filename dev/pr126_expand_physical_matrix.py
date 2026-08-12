from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: replacement count={count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


runner = "dev/benchmarks/validate_panel_stage_c_gpu.py"
replace_once(
    runner,
    """    return X, resid, time, clusters\n\n\ndef _to_backend(X, y, entity, time, backend):\n""",
    """    return X, resid, time, clusters\n\n\ndef _rank_deficient_estimator_inputs(seed=20260816):\n    \"\"\"Return an unbalanced exact-collinearity panel for estimator integration.\"\"\"\n    rng = np.random.default_rng(seed)\n    n_entities, n_times = 15, 4\n    entity = np.repeat(np.arange(n_entities), n_times)\n    time = np.tile(np.arange(n_times), n_entities)\n    keep = np.ones(entity.size, dtype=bool)\n    keep[[1, 10, 19, 33, 46, 57]] = False\n    entity = entity[keep]\n    time = time[keep]\n    x = rng.normal(size=entity.size)\n    X = np.column_stack([x, 2.0 * x]).astype(np.float64)\n    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)[keep]\n    y = 0.4 + 0.7 * x + alpha + rng.normal(scale=0.2, size=entity.size)\n    return X, y.astype(np.float64), entity, time\n\n\ndef _to_backend(X, y, entity, time, backend):\n""",
)
replace_once(
    runner,
    """    X_rank, y_rank, time_rank, _cluster_rank = _rank_boundary_inputs()\n""",
    """    X_rd, y_rd, entity_rd, time_rd = _rank_deficient_estimator_inputs()\n    X_rd_b, y_rd_b, entity_rd_b, time_rd_b = _to_backend(\n        X_rd, y_rd, entity_rd, time_rd, backend\n    )\n    X_rd_re = np.column_stack([np.ones(len(y_rd)), X_rd])\n    X_rd_re_b, y_rd_re_b, entity_rd_re_b, _time_rd_re_b = _to_backend(\n        X_rd_re, y_rd, entity_rd, time_rd, backend\n    )\n    for cov in (\"nonrobust\", \"robust\"):\n        cases[f\"panel_entity_rank_deficient_{cov}\"] = PanelOLS(\n            entity_effects=True, cov_type=cov, device=device\n        ).fit(X_rd_b, y_rd_b, entity_ids=entity_rd_b)\n        cases[f\"between_rank_deficient_{cov}\"] = BetweenOLS(\n            cov_type=cov, device=device\n        ).fit(X_rd_b, y_rd_b, entity_ids=entity_rd_b)\n        cases[f\"first_difference_rank_deficient_{cov}\"] = FirstDifferenceOLS(\n            cov_type=cov, device=device\n        ).fit(X_rd_b, y_rd_b, entity_ids=entity_rd_b, time_ids=time_rd_b)\n        cases[f\"random_effects_rank_deficient_{cov}\"] = RandomEffects(\n            cov_type=cov, device=device\n        ).fit(X_rd_re_b, y_rd_re_b, entity_ids=entity_rd_re_b)\n\n    X_rank, y_rank, time_rank, _cluster_rank = _rank_boundary_inputs()\n""",
)
replace_once(
    runner,
    """        \"first_difference_hc0\", \"first_difference_hc2\", \"first_difference_hc3\",\n        \"panel_rank_boundary_dk\",\n""",
    """        \"first_difference_hc0\", \"first_difference_hc2\", \"first_difference_hc3\",\n        \"panel_entity_rank_deficient_nonrobust\", \"panel_entity_rank_deficient_robust\",\n        \"between_rank_deficient_nonrobust\", \"between_rank_deficient_robust\",\n        \"first_difference_rank_deficient_nonrobust\", \"first_difference_rank_deficient_robust\",\n        \"random_effects_rank_deficient_nonrobust\", \"random_effects_rank_deficient_robust\",\n        \"panel_rank_boundary_dk\",\n""",
)
replace_once(
    runner,
    """    if len(reference) != 27:\n        raise AssertionError(f\"expected 27 Stage-C physical cases, got {len(reference)}\")\n""",
    """    if len(reference) != 35:\n        raise AssertionError(f\"expected 35 Stage-C physical cases, got {len(reference)}\")\n""",
)

contract = "dev/tests/test_panel_stage_c_physical_runner_contract.py"
replace_once(
    contract,
    """        \"first_difference_hc0\", \"first_difference_hc2\", \"first_difference_hc3\",\n        \"panel_rank_boundary_dk\",\n""",
    """        \"first_difference_hc0\", \"first_difference_hc2\", \"first_difference_hc3\",\n        \"panel_entity_rank_deficient_nonrobust\", \"panel_entity_rank_deficient_robust\",\n        \"between_rank_deficient_nonrobust\", \"between_rank_deficient_robust\",\n        \"first_difference_rank_deficient_nonrobust\", \"first_difference_rank_deficient_robust\",\n        \"random_effects_rank_deficient_nonrobust\", \"random_effects_rank_deficient_robust\",\n        \"panel_rank_boundary_dk\",\n""",
)
p = Path(contract)
text = p.read_text(encoding="utf-8")
addition = r'''


def test_stage_c_runner_rank_deficient_estimators_exercise_identified_rank_df():
    X, y, entity, time, clusters = _MOD._dataset()
    cases = _MOD._fit_cases(X, y, entity, time, clusters, "numpy")
    names = {
        "panel_entity_rank_deficient_nonrobust",
        "panel_entity_rank_deficient_robust",
        "between_rank_deficient_nonrobust",
        "between_rank_deficient_robust",
        "first_difference_rank_deficient_nonrobust",
        "first_difference_rank_deficient_robust",
        "random_effects_rank_deficient_nonrobust",
        "random_effects_rank_deficient_robust",
    }
    for name in names:
        model = cases[name]
        meta = model._covariance_metadata
        assert meta.get("design_rank", 0) < meta.get("design_columns", 0), name
        assert model.df_resid > 0, name
        assert np.all(np.isfinite(model.bse_)), name
'''
if "test_stage_c_runner_rank_deficient_estimators_exercise_identified_rank_df" in text:
    raise SystemExit("physical rank estimator contract already exists")
p.write_text(text + addition, encoding="utf-8")

perf = "dev/benchmarks/benchmark_panel_stage_c_covariance.py"
replace_once(
    perf,
    """def _version(name):\n    try:\n        return importlib.metadata.version(name)\n    except importlib.metadata.PackageNotFoundError:\n        return None\n\n\ndef _parse_scales(text):\n""",
    """def _version(name):\n    try:\n        return importlib.metadata.version(name)\n    except importlib.metadata.PackageNotFoundError:\n        return None\n\n\ndef _package_version(name):\n    if name == \"cupy\":\n        return (\n            _version(\"cupy\")\n            or _version(\"cupy-cuda11x\")\n            or _version(\"cupy-cuda12x\")\n        )\n    return _version(name)\n\n\ndef _parse_scales(text):\n""",
)
replace_once(
    perf,
    """            \"packages\": {\n                name: _version(name)\n                for name in (\"statgpu\", \"numpy\", \"cupy\", \"torch\")\n            },\n""",
    """            \"packages\": {\n                name: _package_version(name)\n                for name in (\"statgpu\", \"numpy\", \"cupy\", \"torch\")\n            },\n""",
)
p = Path("dev/tests/test_panel_stage_c_performance_runner_contract.py")
text = p.read_text(encoding="utf-8")
addition = r'''


def test_cupy_package_version_accepts_cuda_specific_distribution(monkeypatch):
    mod = _runner()
    versions = {"cupy-cuda12x": "13.6.0"}
    monkeypatch.setattr(mod, "_version", lambda name: versions.get(name))
    assert mod._package_version("cupy") == "13.6.0"
'''
if "test_cupy_package_version_accepts_cuda_specific_distribution" in text:
    raise SystemExit("performance package provenance test already exists")
p.write_text(text + addition, encoding="utf-8")
