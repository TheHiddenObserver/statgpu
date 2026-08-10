from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


PARSER = "dev/benchmarks/frontend_data/parsers/panel_stage_c.py"
TEST = "dev/tests/test_panel_stage_c_frontend_source.py"

replace_once(
    PARSER,
    '''_EXPECTED_PRIMITIVES = {"cluster_group_debias", "driscoll_kraay_qs"}\n_HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}\n''',
    '''_EXPECTED_PRIMITIVES = {"cluster_group_debias", "driscoll_kraay_qs"}\n_BASE_CASES = {\n    "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",\n    "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",\n    "random_effects_nonrobust", "random_effects_hc3",\n}\n_BASE_SCALES = {(10000, 2, 20), (100000, 2, 20), (100000, 10, 20)}\n_HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}\n''',
)

replace_once(
    PARSER,
    '''def _scale(n_samples: int, n_features: int, *, suffix: str | None = None) -> dict[str, Any]:\n    label = make_scale_label(int(n_samples), int(n_features))\n    if suffix:\n        label = f"{label} · {suffix}"\n    return {\n        "scale_key": make_scale_key(int(n_samples), int(n_features)),\n        "n_samples": int(n_samples),\n        "n_features": int(n_features),\n        "label": label,\n    }\n''',
    '''def _scale(\n    n_samples: int,\n    n_features: int,\n    *,\n    suffix: str | None = None,\n    key_suffix: str | None = None,\n) -> dict[str, Any]:\n    label = make_scale_label(int(n_samples), int(n_features))\n    if suffix:\n        label = f"{label} · {suffix}"\n    scale_key = make_scale_key(int(n_samples), int(n_features))\n    if key_suffix:\n        scale_key = f"{scale_key}_{key_suffix}"\n    return {\n        "scale_key": scale_key,\n        "n_samples": int(n_samples),\n        "n_features": int(n_features),\n        "label": label,\n    }\n''',
)

replace_once(
    PARSER,
    '''    high_t = [row for row in rows if row.get("scenario") == "high_t_qs"]\n    if len(high_t) != 4:\n        raise ValueError("PR126 Stage-C performance requires four high-T QS rows")\n    if {row.get("case") for row in high_t} != _HIGH_T_CASES:\n        raise ValueError("PR126 Stage-C high-T QS case identity drifted")\n    if any(\n        int(row.get("n_samples", 0)) != 10000\n        or int(row.get("n_features", 0)) != 2\n        or int(row.get("n_times", 0)) != 200\n        for row in high_t\n    ):\n        raise ValueError("PR126 Stage-C high-T QS dimensions drifted")\n''',
    '''    base_rows = [row for row in rows if row.get("scenario") == "base"]\n    expected_base = {\n        (backend, case_name, n_samples, n_features, n_times)\n        for backend in ("cupy", "torch")\n        for case_name in _BASE_CASES\n        for n_samples, n_features, n_times in _BASE_SCALES\n    }\n    actual_base = [\n        (\n            row.get("backend"),\n            row.get("case"),\n            int(row.get("n_samples", 0)),\n            int(row.get("n_features", 0)),\n            int(row.get("n_times", 0)),\n        )\n        for row in base_rows\n    ]\n    if len(actual_base) != len(expected_base) or set(actual_base) != expected_base:\n        raise ValueError("PR126 Stage-C performance base matrix drifted")\n\n    high_t = [row for row in rows if row.get("scenario") == "high_t_qs"]\n    expected_high_t = {\n        (backend, case_name, 10000, 2, 200)\n        for backend in ("cupy", "torch")\n        for case_name in _HIGH_T_CASES\n    }\n    actual_high_t = [\n        (\n            row.get("backend"),\n            row.get("case"),\n            int(row.get("n_samples", 0)),\n            int(row.get("n_features", 0)),\n            int(row.get("n_times", 0)),\n        )\n        for row in high_t\n    ]\n    if len(actual_high_t) != len(expected_high_t) or set(actual_high_t) != expected_high_t:\n        raise ValueError("PR126 Stage-C performance high-T QS matrix drifted")\n''',
)

replace_once(
    PARSER,
    '''        scale = _scale(n_samples, n_features, suffix=f"T={n_times}")\n''',
    '''        scale = _scale(\n            n_samples,\n            n_features,\n            suffix=f"T={n_times}",\n            key_suffix=f"t{n_times}",\n        )\n''',
)

# Strengthen the canonical parser tests around the two review findings.
test_path = Path(TEST)
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    '''    high_t = [run for run in runs if run["parameters"]["scenario"] == "high_t_qs"]\n    assert len(high_t) == 4\n    assert {run["backend"] for run in high_t} == {"cupy", "torch"}\n    assert {run["parameters"]["n_times"] for run in high_t} == {200}\n    assert {run["model_id"] for run in high_t} == {"PooledOLS", "PanelOLS"}\n''',
    '''    base = [run for run in runs if run["parameters"]["scenario"] == "base"]\n    high_t = [run for run in runs if run["parameters"]["scenario"] == "high_t_qs"]\n    assert len(base) == 54\n    assert len(high_t) == 4\n    assert {run["backend"] for run in high_t} == {"cupy", "torch"}\n    assert {run["parameters"]["n_times"] for run in high_t} == {200}\n    assert {run["model_id"] for run in high_t} == {"PooledOLS", "PanelOLS"}\n    assert {run["scale"]["scale_key"] for run in base} == {\n        "n10000_p2_t20", "n100000_p2_t20", "n100000_p10_t20"\n    }\n    assert {run["scale"]["scale_key"] for run in high_t} == {"n10000_p2_t200"}\n    assert {run["scale"]["label"] for run in high_t} == {"10K×2 · T=200"}\n''',
    1,
)
if "test_stage_c_performance_parser_fails_closed_on_base_matrix_drift" not in text:
    text += '''\n\ndef test_stage_c_performance_parser_fails_closed_on_base_matrix_drift(tmp_path):\n    data = json.loads(PERFORMANCE.read_text(encoding="utf-8"))\n    base_rows = [row for row in data["rows"] if row["scenario"] == "base"]\n    assert len(base_rows) == 54\n    base_rows[0]["case"] = base_rows[1]["case"]\n    broken = tmp_path / "broken_base_matrix.json"\n    broken.write_text(json.dumps(data), encoding="utf-8")\n    with pytest.raises(ValueError, match="base matrix drifted"):\n        parse_panel_stage_c_performance(broken, "remote-p100-pr126-20260810")\n\n\ndef test_stage_c_performance_parser_fails_closed_on_high_t_backend_matrix_drift(tmp_path):\n    data = json.loads(PERFORMANCE.read_text(encoding="utf-8"))\n    high_t = [row for row in data["rows"] if row["scenario"] == "high_t_qs"]\n    assert len(high_t) == 4\n    high_t[0]["backend"] = "torch"\n    broken = tmp_path / "broken_high_t_matrix.json"\n    broken.write_text(json.dumps(data), encoding="utf-8")\n    with pytest.raises(ValueError, match="high-T QS matrix drifted"):\n        parse_panel_stage_c_performance(broken, "remote-p100-pr126-20260810")\n'''
test_path.write_text(text, encoding="utf-8")
