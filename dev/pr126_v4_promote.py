#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_SHA = "a99726e19c535dfcd0a94711bbc8be6aac437584"
RAW_COMMIT = "ccc46da6c5f2dee025c7715e39215db69b2872b8"
CORRECTNESS_SHA256 = "2d929bccf1c7a0ade385c495bd6a3144cd607dec413c80857e455263c9f1f017"
CORRECTNESS_BLOB = "ca40d98e48e7747c080f7bf1868cf355ada048a5"
PERFORMANCE_SHA256 = "2238002d491fe9397890af1d5e87162458f0a98b293ecf41f6b8831e5a9152b6"
PERFORMANCE_BLOB = "e1b61e05ea93425947d1e6a7b35d38227d22c358"
VALIDATION_SOURCE_ID = "panel-stage-c-identifiability-validation-pr126-20260812-2d929bccf1c7"
PERFORMANCE_SOURCE_ID = "panel-stage-c-identifiability-performance-pr126-20260812-2238002d491f"
VALIDATION_COMPARISON_ID = "panel-stage-c-pr126-identifiability-validation-20260812"
PERFORMANCE_COMPARISON_ID = "panel-stage-c-pr126-identifiability-performance-20260812"
ENV_ID = "remote-p100-pr126-identifiability-20260812"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, found {count}")
    return text.replace(old, new, 1)


def build_v4_parser() -> None:
    source_path = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_df.py"
    target_path = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c_identifiability.py"
    if target_path.exists():
        raise RuntimeError("v4 parser already exists; refusing to overwrite")
    text = source_path.read_text()
    text = text.replace("Canonical v3 parsers", "Canonical v4 parsers")
    text = text.replace("post-rank-df", "post-identifiability")
    text = text.replace("``f1546476``", "``a99726e1``")
    text = replace_once(
        text,
        '_MEASUREMENT_SHA = "f154647665788df2570439a1cc154a43f509aa45"',
        f'_MEASUREMENT_SHA = "{MEASUREMENT_SHA}"',
        "measurement sha",
    )
    text = text.replace("rank_df", "identifiability")
    text = text.replace("rank-df", "identifiability")
    text = replace_once(text, '_PARSER_VERSION = "3.0"', '_PARSER_VERSION = "4.0"', "parser version")
    text = replace_once(
        text,
        '_VALIDATION_PARSER = "parse_panel_stage_c_identifiability_physical_validation_v3"',
        '_VALIDATION_PARSER = "parse_panel_stage_c_identifiability_physical_validation_v4"',
        "validation parser identity",
    )
    text = replace_once(
        text,
        '_PERFORMANCE_PARSER = "parse_panel_stage_c_identifiability_performance_v3"',
        '_PERFORMANCE_PARSER = "parse_panel_stage_c_identifiability_performance_v4"',
        "performance parser identity",
    )
    text = replace_once(
        text,
        'if int(data.get("schema_version", -1)) != 1:',
        'if int(data.get("schema_version", -1)) != 2:',
        "correctness schema",
    )
    text = replace_once(
        text,
        'requires schema_version=1',
        'requires schema_version=2',
        "correctness schema message",
    )
    text = replace_once(
        text,
        'if int(data.get("schema_version", -1)) != 2:',
        'if int(data.get("schema_version", -1)) != 3:',
        "performance schema",
    )
    text = replace_once(
        text,
        'performance requires schema_version=2',
        'performance requires schema_version=3',
        "performance schema message",
    )

    marker = "\n_BASE_CASES = {"
    insertion = '''\n_EXPECTED_IDENTIFIABILITY_CASES = _EXPECTED_RANK_DEFICIENT_CASES | {\n    "panel_rank_boundary_dk"\n}\n_PREDICTION_BACKEND_CASES = {\n    "panel_entity_hc0",\n    "random_effects_explicit_constant_hc0",\n}\n'''
    text = replace_once(text, marker, insertion + marker, "identifiability constants")

    old_boundary = '''        boundary_contract_ok = (\n            boundary.get("design_rank") == 2\n            and boundary.get("design_columns") == 3\n            and boundary.get("rank_deficient_extension") is True\n        )'''
    new_boundary = '''        boundary_contract_ok = (\n            boundary.get("design_rank") == 2\n            and boundary.get("design_columns") == 3\n            and boundary.get("rank_deficient_extension") is True\n            and boundary.get("coefficient_inference_applicable") is False\n            and isinstance(boundary.get("coefficient_inference_reason"), str)\n            and "rank deficient" in boundary.get("coefficient_inference_reason", "").lower()\n        )'''
    text = replace_once(text, old_boundary, new_boundary, "rank-boundary identifiability")
    text = replace_once(
        text,
        'for rank_case_name in sorted(_EXPECTED_RANK_DEFICIENT_CASES):',
        'for rank_case_name in sorted(_EXPECTED_IDENTIFIABILITY_CASES):',
        "rank case loop",
    )
    text = replace_once(
        text,
        '''                and 0 < fit_rank < parameter_count\n            )''',
        '''                and 0 < fit_rank < parameter_count\n                and rank_case.get("coefficient_inference_applicable") is False\n                and isinstance(rank_case.get("coefficient_inference_reason"), str)\n                and "rank deficient" in rank_case.get("coefficient_inference_reason", "").lower()\n            )''',
        "rank inference contract",
    )

    case_anchor = '''            case = cases[case_name]\n            diff_ok = _finite_diff_map(case.get("max_abs_differences"))'''
    case_insert = '''            case = cases[case_name]\n            identified_case = case_name in _EXPECTED_IDENTIFIABILITY_CASES\n            inference_contract_ok = (\n                case.get("coefficient_inference_applicable") is (not identified_case)\n                and (\n                    (\n                        identified_case\n                        and isinstance(case.get("coefficient_inference_reason"), str)\n                        and "rank deficient"\n                        in case.get("coefficient_inference_reason", "").lower()\n                    )\n                    or (\n                        not identified_case\n                        and case.get("coefficient_inference_reason") is None\n                    )\n                )\n            )\n            if not inference_contract_ok:\n                raise ValueError(\n                    f"{backend}: {case_name} coefficient-inference applicability drifted"\n                )\n            prediction_backend_ok = (\n                case_name not in _PREDICTION_BACKEND_CASES\n                or case.get("prediction_backend") == backend\n            )\n            if not prediction_backend_ok:\n                raise ValueError(\n                    f"{backend}: {case_name} prediction backend provenance drifted"\n                )\n            diff_ok = _finite_diff_map(case.get("max_abs_differences"))'''
    text = replace_once(text, case_anchor, case_insert, "case identifiability checks")
    text = replace_once(
        text,
        '''                _bool_check("recorded_numpy_difference_finite", diff_ok),\n            ]''',
        '''                _bool_check("recorded_numpy_difference_finite", diff_ok),\n                _bool_check("coefficient_inference_applicability_contract", inference_contract_ok),\n            ]\n            if case_name in _PREDICTION_BACKEND_CASES:\n                checks.append(\n                    _bool_check("prediction_backend_matches_requested", prediction_backend_ok)\n                )''',
        "case validation checks",
    )
    text = text.replace(
        'if case_name in _EXPECTED_RANK_DEFICIENT_CASES:',
        'if case_name in _EXPECTED_IDENTIFIABILITY_CASES:',
    )
    text = replace_once(
        text,
        '''                        "parameter_count": case.get("parameter_count"),\n                    },''',
        '''                        "parameter_count": case.get("parameter_count"),\n                        "coefficient_inference_applicable": case.get(\n                            "coefficient_inference_applicable"\n                        ),\n                        "coefficient_inference_reason": case.get(\n                            "coefficient_inference_reason"\n                        ),\n                        "prediction_backend": case.get("prediction_backend"),\n                    },''',
        "case parameters",
    )

    text = replace_once(
        text,
        '''    if data.get("high_t_scale") != "10000x2x200":\n        raise ValueError("PR126 identifiability Stage-C high-T scale drifted")''',
        '''    if data.get("high_t_scale") != "10000x2x200":\n        raise ValueError("PR126 identifiability Stage-C high-T scale drifted")\n    if data.get("two_way_unbalanced_scale") != "10000x2x20":\n        raise ValueError("PR126 identifiability Stage-C two-way scale drifted")''',
        "two-way scale",
    )
    text = replace_once(text, 'if len(rows) != 58:', 'if len(rows) != 60:', "performance row count")
    text = replace_once(
        text,
        'requires 58 rows, got {len(rows)}',
        'requires 60 rows, got {len(rows)}',
        "performance row message",
    )

    high_t_anchor = '''    if len(actual_high_t) != len(expected_high_t) or set(actual_high_t) != expected_high_t:\n        raise ValueError("PR126 identifiability Stage-C performance high-T QS matrix drifted")\n\n    output: list[dict] = []'''
    high_t_insert = '''    if len(actual_high_t) != len(expected_high_t) or set(actual_high_t) != expected_high_t:\n        raise ValueError("PR126 identifiability Stage-C performance high-T QS matrix drifted")\n\n    two_way = [row for row in rows if row.get("scenario") == "two_way_unbalanced"]\n    expected_two_way = {\n        (backend, "panel_two_way_nonrobust", 10000, 2, 20)\n        for backend in ("cupy", "torch")\n    }\n    actual_two_way = [\n        (\n            row.get("backend"), row.get("case"), int(row.get("n_samples", 0)),\n            int(row.get("n_features", 0)), int(row.get("n_times", 0)),\n        )\n        for row in two_way\n    ]\n    if len(actual_two_way) != len(expected_two_way) or set(actual_two_way) != expected_two_way:\n        raise ValueError(\n            "PR126 identifiability Stage-C performance two-way matrix drifted"\n        )\n\n    output: list[dict] = []'''
    text = replace_once(text, high_t_anchor, high_t_insert, "two-way performance matrix")
    text = replace_once(
        text,
        'if scenario not in {"base", "high_t_qs"}:',
        'if scenario not in {"base", "high_t_qs", "two_way_unbalanced"}:',
        "scenario set",
    )
    target_path.write_text(text)


def update_parser_exports() -> None:
    path = ROOT / "dev/benchmarks/frontend_data/parsers/__init__.py"
    text = path.read_text()
    anchor = '''from .panel_stage_c_rank_df import (\n    parse_panel_stage_c_rank_df_physical_validation,\n    parse_panel_stage_c_rank_df_performance,\n)\n'''
    addition = anchor + '''from .panel_stage_c_identifiability import (\n    parse_panel_stage_c_identifiability_physical_validation,\n    parse_panel_stage_c_identifiability_performance,\n)\n'''
    text = replace_once(text, anchor, addition, "parser import")
    anchor_list = '''    "parse_panel_stage_c_rank_df_physical_validation",\n    "parse_panel_stage_c_rank_df_performance",\n]'''
    replacement = '''    "parse_panel_stage_c_rank_df_physical_validation",\n    "parse_panel_stage_c_rank_df_performance",\n    "parse_panel_stage_c_identifiability_physical_validation",\n    "parse_panel_stage_c_identifiability_performance",\n]'''
    path.write_text(replace_once(text, anchor_list, replacement, "parser __all__"))


def update_registry() -> None:
    path = ROOT / "dev/benchmarks/frontend_data/registry.py"
    text = path.read_text()
    import_anchor = ''')\n\nMINIMUM_DASHBOARD_SOURCE_DATE = date(2026, 6, 1)'''
    import_replacement = ''')\nfrom .parsers import (\n    parse_panel_stage_c_identifiability_physical_validation,\n    parse_panel_stage_c_identifiability_performance,\n)\n\nMINIMUM_DASHBOARD_SOURCE_DATE = date(2026, 6, 1)'''
    text = replace_once(text, import_anchor, import_replacement, "registry import")
    map_anchor = '''}\n\n\ndef validate_manifest_source_dates'''
    map_replacement = '''}\nPARSER_FUNCTIONS.update(\n    {\n        "panel_stage_c_identifiability_physical_validation": (\n            parse_panel_stage_c_identifiability_physical_validation\n        ),\n        "panel_stage_c_identifiability_performance": (\n            parse_panel_stage_c_identifiability_performance\n        ),\n    }\n)\n\n\ndef validate_manifest_source_dates'''
    path.write_text(replace_once(text, map_anchor, map_replacement, "registry map"))


def update_manifest() -> None:
    path = ROOT / "dev/benchmarks/frontend_sources.json"
    manifest = json.loads(path.read_text())
    if ENV_ID in manifest["environments"]:
        raise RuntimeError("v4 environment already registered")
    manifest["environments"][ENV_ID] = {
        "label": "Tesla P100 PR #126 Panel Stage C identifiability — 2026-08-12",
        "gpu": "Tesla P100-SXM2-16GB",
        "cpu": "x86_64",
    }
    manifest["comparisons"][VALIDATION_COMPARISON_ID] = {
        "label": "Panel Stage C identifiability physical validation — PR #126 — 2026-08-12",
        "env_id": ENV_ID,
    }
    manifest["comparisons"][PERFORMANCE_COMPARISON_ID] = {
        "label": "Panel Stage C identifiability synchronized performance — PR #126 — 2026-08-12",
        "env_id": ENV_ID,
    }
    sources = manifest["sources"]
    if any(s.get("source_id") in {VALIDATION_SOURCE_ID, PERFORMANCE_SOURCE_ID} for s in sources):
        raise RuntimeError("v4 source already registered")
    old_validation = next(
        s for s in sources
        if s.get("source_id") == "panel-stage-c-rank-df-validation-pr126-20260812-0b4eb5810ad0"
    )
    old_performance = next(
        s for s in sources
        if s.get("source_id") == "panel-stage-c-rank-df-performance-pr126-20260812-09337cc62c94"
    )
    validation = copy.deepcopy(old_validation)
    validation.update(
        {
            "source_id": VALIDATION_SOURCE_ID,
            "comparison_id": VALIDATION_COMPARISON_ID,
            "path": "results/pr126_p100/panel_stage_c_gpu_validation_a99726e1.json",
            "sha256": CORRECTNESS_SHA256,
            "parser": "panel_stage_c_identifiability_physical_validation",
            "parser_version": "4.0",
            "env_id": ENV_ID,
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                "PR #126 exact-clean a99726e1 P100 correctness evidence committed immutably in "
                f"{RAW_COMMIT}; Git blob {CORRECTNESS_BLOB}; SHA-256 {CORRECTNESS_SHA256}. "
                "CuPy and Torch each pass 35 estimator integrations plus 12 public covariance "
                "primitives (47/47). Every fitted estimator persists the requested/executed backend "
                "identity. All nine rank-deficient acceptance cases, including the DK rank boundary, "
                "record fit_rank < parameter_count, coefficient_inference_applicable=false, and an "
                "explicit rank-deficiency reason. Representative PanelOLS and RandomEffects predictions "
                "persist the requested prediction backend. No numerical CPU fallback is accepted."
            ),
        }
    )
    performance = copy.deepcopy(old_performance)
    performance.update(
        {
            "source_id": PERFORMANCE_SOURCE_ID,
            "comparison_id": PERFORMANCE_COMPARISON_ID,
            "path": "results/pr126_p100/panel_stage_c_performance_a99726e1.json",
            "sha256": PERFORMANCE_SHA256,
            "parser": "panel_stage_c_identifiability_performance",
            "parser_version": "4.0",
            "env_id": ENV_ID,
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                "PR #126 exact-clean a99726e1 P100 synchronized timing evidence committed immutably in "
                f"{RAW_COMMIT}; Git blob {PERFORMANCE_BLOB}; SHA-256 {PERFORMANCE_SHA256}. "
                "All 60 synchronized end-to-end fit rows pass: 54 base rows, four bounded "
                "N=10000, k=2, T=200 QS all-lag rows, and two N=10000, k=2, T=20 unbalanced "
                "two-way-FE rows. Every row has three finite positive samples and an exact stored median. "
                "Schema v3 persists the requested backend label, not a second row-local executed_backend "
                "field; the exact immutable runner fails closed before returning elapsed time unless the "
                "fitted model persists _backend_name equal to that requested backend. The paired required "
                "correctness source independently persists fit execution and prediction backend provenance. "
                "No executed_backend field is synthesized and no CPU-speedup claim is made."
            ),
        }
    )
    sources.extend([validation, performance])
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def update_coverage() -> None:
    path = ROOT / "dev/benchmarks/benchmark_coverage_matrix.json"
    data = json.loads(path.read_text())
    panel = next(c for c in data["capabilities"] if c["capability_id"] == "panel-estimation")
    for source_id in (VALIDATION_SOURCE_ID, PERFORMANCE_SOURCE_ID):
        if source_id not in panel["source_ids"]:
            panel["source_ids"].append(source_id)
    for dim in ("coefficient_inference_applicability", "prediction_backend", "two_way_fe_timing"):
        if dim not in panel["representative_dimensions"]:
            panel["representative_dimensions"].append(dim)
    panel["disposition"] = (
        "June timing rows cover aligned PanelOLS and RandomEffects; PR #122 provides canonical Stage-B "
        "diagnostic/physical validation. PR #126 current Stage-C acceptance is the exact-clean a99726e1 "
        "P100 identifiability source: CuPy and Torch each pass 35 estimator integrations plus 12 direct "
        "public primitives (47/47), including nine rank-deficient acceptance cases with ordinary "
        "coordinate inference explicitly unavailable and representative PanelOLS/RandomEffects prediction "
        "backend provenance. The paired current performance source contributes 60 synchronized rows: "
        "54 base rows, four bounded N=10,000, k=2, T=200 QS rows, and two N=10,000, k=2, T=20 unbalanced "
        "two-way-FE rows. Historical v1/v2/v3 Stage-C sources remain immutable audit evidence and are not "
        "overwritten. No Stage-C speedup claim is made."
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def update_review_record() -> None:
    path = ROOT / "dev/reviews/pr126_round4_autofix_review_2026-08-12.md"
    text = path.read_text()
    text = text.replace(
        "`LOCAL_REVIEW_CLEAN / HOSTED_FINAL_PENDING / NOT MERGE-READY`",
        "`PHYSICAL_GPU_ACCEPTED / CANONICAL_PROMOTION_PENDING / NOT MERGE-READY`",
    )
    text = text.replace(
        "requested backend must equal persisted executed fit backend",
        "each timing row persists the requested backend label, and the exact runner must fail closed unless the fitted model persists `_backend_name` equal to that requested backend",
    )
    section = f'''\n\n## Fresh physical v4 audit — accepted\n\nExact measurement source: `{MEASUREMENT_SHA}`.  Raw artifact commit: `{RAW_COMMIT}`.\n\n- correctness schema v2: CuPy **47/47**, Torch **47/47**; SHA-256 `{CORRECTNESS_SHA256}`; Git blob `{CORRECTNESS_BLOB}`;\n- performance schema v3: **60/60** synchronized rows; SHA-256 `{PERFORMANCE_SHA256}`; Git blob `{PERFORMANCE_BLOB}`;\n- all nine rank-deficient estimator acceptance cases record `fit_rank < parameter_count`, `coefficient_inference_applicable=false`, and an explicit reason;\n- `panel_entity_hc0` and `random_effects_explicit_constant_hc0` persist the requested prediction backend;\n- all timing rows have three finite positive samples and exact persisted medians.\n\n[MEDIUM][ARTIFACT][fixed] The pre-measurement review wording overstated schema-v3 timing provenance as a row-local persisted `executed_backend`. The artifact truth is narrower and still fail-closed: each row persists `backend`, while the exact immutable runner checks fitted `model._backend_name == backend` before returning elapsed time. The v4 source documents this runner-level proof and does **not** synthesize an absent field. No physical rerun is required because neither the runner nor numerical behavior changed after measurement.\n\nThe v1/v2/v3 parser/source identities remain frozen. Fresh evidence is promoted only through new v4 identifiability parser/source identities.\n'''
    if "## Fresh physical v4 audit — accepted" not in text:
        text += section
    path.write_text(text)

    plan_path = ROOT / "dev/plans/panel_p1_stage_c_covariance_plan.md"
    plan = plan_path.read_text()
    plan = plan.replace(
        "requested backend must equal persisted executed fit backend",
        "each timing row persists the requested backend label, while the exact runner fails closed unless the fitted model persists the same executed backend before elapsed time is returned",
    )
    plan_path.write_text(plan)


def write_tests() -> None:
    path = ROOT / "dev/tests/test_panel_stage_c_identifiability_frontend_source.py"
    if path.exists():
        raise RuntimeError("v4 parser test already exists")
    path.write_text('''from __future__ import annotations\n\nimport copy\nimport json\nfrom pathlib import Path\n\nimport pytest\n\nfrom dev.benchmarks.frontend_data.parsers.panel_stage_c_identifiability import (\n    parse_panel_stage_c_identifiability_performance,\n    parse_panel_stage_c_identifiability_physical_validation,\n)\n\nROOT = Path(__file__).resolve().parents[2]\nVALIDATION = ROOT / "results/pr126_p100/panel_stage_c_gpu_validation_a99726e1.json"\nPERFORMANCE = ROOT / "results/pr126_p100/panel_stage_c_performance_a99726e1.json"\nENV = "remote-p100-pr126-identifiability-20260812"\n\n\ndef _write(tmp_path, name, payload):\n    path = tmp_path / name\n    path.write_text(json.dumps(payload))\n    return path\n\n\ndef test_v4_validation_parser_enforces_identifiability_and_prediction_backend():\n    runs, models, warnings = parse_panel_stage_c_identifiability_physical_validation(VALIDATION, ENV)\n    assert len(runs) == 94\n    assert warnings == []\n    assert models\n    assert all(run["source"]["parser_version"] == "4.0" for run in runs)\n    rank_runs = [\n        run for run in runs\n        if run["parameters"].get("coefficient_inference_applicable") is False\n    ]\n    assert len(rank_runs) == 18\n    assert all("rank deficient" in run["parameters"]["coefficient_inference_reason"].lower() for run in rank_runs)\n    prediction_runs = [run for run in runs if run["parameters"].get("prediction_backend") is not None]\n    assert len(prediction_runs) == 4\n    assert all(run["parameters"]["prediction_backend"] == run["backend"] for run in prediction_runs)\n\n\ndef test_v4_validation_parser_rejects_inference_or_prediction_provenance_drift(tmp_path):\n    payload = json.loads(VALIDATION.read_text())\n    broken = copy.deepcopy(payload)\n    broken["backends"]["cupy"]["cases"]["panel_rank_boundary_dk"]["coefficient_inference_applicable"] = True\n    with pytest.raises(ValueError, match="coefficient-inference|rank-boundary|identified-rank"):\n        parse_panel_stage_c_identifiability_physical_validation(\n            _write(tmp_path, "bad-inference.json", broken), ENV\n        )\n\n    broken = copy.deepcopy(payload)\n    broken["backends"]["torch"]["cases"]["panel_entity_hc0"]["prediction_backend"] = "numpy"\n    with pytest.raises(ValueError, match="prediction backend"):\n        parse_panel_stage_c_identifiability_physical_validation(\n            _write(tmp_path, "bad-predict.json", broken), ENV\n        )\n\n\ndef test_v4_performance_parser_enforces_60_row_matrix_and_two_way_case():\n    runs, models, warnings = parse_panel_stage_c_identifiability_performance(PERFORMANCE, ENV)\n    assert len(runs) == 60\n    assert warnings == []\n    assert models\n    two_way = [run for run in runs if run["parameters"]["scenario"] == "two_way_unbalanced"]\n    assert len(two_way) == 2\n    assert {run["backend"] for run in two_way} == {"cupy", "torch"}\n    assert all(run["source"]["parser_version"] == "4.0" for run in runs)\n\n\ndef test_v4_performance_parser_rejects_matrix_or_median_drift(tmp_path):\n    payload = json.loads(PERFORMANCE.read_text())\n    broken = copy.deepcopy(payload)\n    broken["rows"] = broken["rows"][:-1]\n    with pytest.raises(ValueError, match="60 rows"):\n        parse_panel_stage_c_identifiability_performance(\n            _write(tmp_path, "bad-matrix.json", broken), ENV\n        )\n\n    broken = copy.deepcopy(payload)\n    broken["rows"][0]["median_seconds"] *= 2.0\n    with pytest.raises(ValueError, match="median"):\n        parse_panel_stage_c_identifiability_performance(\n            _write(tmp_path, "bad-median.json", broken), ENV\n        )\n''')


def main() -> None:
    build_v4_parser()
    update_parser_exports()
    update_registry()
    update_manifest()
    update_coverage()
    update_review_record()
    write_tests()
    print("PR126 v4 promotion source changes staged")
    print(VALIDATION_SOURCE_ID)
    print(PERFORMANCE_SOURCE_ID)


if __name__ == "__main__":
    main()
