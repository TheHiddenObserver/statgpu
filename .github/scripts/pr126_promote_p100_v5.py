from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_SHA = "5f0cea9216321361842bc3c438219084a4cf5538"
ARTIFACT_COMMIT = "c578f9af48c4322a6deec126f0e1440bb3519852"
SOURCE_DATE = "2026-08-13"
ENV_ID = "remote-p100-pr126-final-20260813"
CORRECTNESS = ROOT / "results/pr126_p100_fresh/panel_stage_c_correctness_p100.json"
PERFORMANCE = ROOT / "results/pr126_p100_fresh/panel_stage_c_performance_p100.json"
RAW_DIR = ROOT / "results/pr126_p100_fresh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return bool(value) and all(finite_tree(v) for v in value)
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def audit_raw_evidence() -> tuple[str, str, str, str]:
    parent = subprocess.check_output(
        ["git", "rev-parse", f"{ARTIFACT_COMMIT}^"], cwd=ROOT, text=True
    ).strip()
    if parent != MEASUREMENT_SHA:
        raise SystemExit(f"artifact commit parent drifted: {parent}")

    correctness = json.loads(CORRECTNESS.read_text(encoding="utf-8"))
    if correctness.get("schema_version") != 2:
        raise SystemExit("correctness schema drifted")
    if correctness.get("git_sha") != MEASUREMENT_SHA:
        raise SystemExit("correctness measurement SHA drifted")
    if correctness.get("working_tree_clean") is not True:
        raise SystemExit("correctness source is not clean-tree evidence")
    if correctness.get("status") != "success":
        raise SystemExit("correctness source is not successful")
    env = correctness.get("environment", {})
    if env.get("gpu") != "Tesla P100-SXM2-16GB":
        raise SystemExit("correctness GPU identity drifted")
    packages = env.get("packages", {})
    if packages.get("cupy") != "13.6.0" or packages.get("torch") != "2.0.0":
        raise SystemExit("correctness CuPy/Torch provenance drifted")
    if correctness.get("case_count_per_backend") != 35:
        raise SystemExit("correctness estimator count drifted")
    if correctness.get("public_primitive_count_per_backend") != 12:
        raise SystemExit("correctness primitive count drifted")
    backends = correctness.get("backends", {})
    if set(backends) != {"cupy", "torch"}:
        raise SystemExit("correctness backend set drifted")
    expected_prediction_contracts = {
        "two_way_disconnected",
        "two_way_connected_partial_labels",
    }
    for backend in ("cupy", "torch"):
        payload = backends[backend]
        if payload.get("status") != "success" or payload.get("requested_backend") != backend:
            raise SystemExit(f"{backend}: correctness backend status drifted")
        if len(payload.get("cases", {})) != 35 or len(payload.get("public_primitives", {})) != 12:
            raise SystemExit(f"{backend}: correctness matrix size drifted")
        for name, case in payload["cases"].items():
            if case.get("status") != "success" or case.get("executed_backend") != backend:
                raise SystemExit(f"{backend}/{name}: estimator provenance drifted")
        for name, primitive in payload["public_primitives"].items():
            if primitive.get("status") != "success" or primitive.get("executed_backend") != backend:
                raise SystemExit(f"{backend}/{name}: primitive provenance drifted")
        contracts = payload.get("prediction_contracts", {})
        if set(contracts) != expected_prediction_contracts:
            raise SystemExit(f"{backend}: prediction-contract identity drifted")
        for name, contract in contracts.items():
            guards = contract.get("guards")
            if (
                contract.get("status") != "success"
                or contract.get("executed_backend") != backend
                or contract.get("prediction_backend") != backend
                or not isinstance(guards, dict)
                or not guards
                or not all(value is True for value in guards.values())
                or not finite_tree(contract.get("max_abs_differences", {}))
            ):
                raise SystemExit(f"{backend}/{name}: prediction contract drifted")
        level = payload.get("level_constant_contract", {})
        if (
            level.get("status") != "success"
            or level.get("executed_backend") != backend
            or level.get("prediction_backend") != backend
            or level.get("constant_index") != 0
            or float(level.get("constant_value", float("nan"))) != 1.0
            or not finite_tree(level.get("max_abs_differences_vs_numpy", {}))
        ):
            raise SystemExit(f"{backend}: level-constant contract drifted")

    performance = json.loads(PERFORMANCE.read_text(encoding="utf-8"))
    if performance.get("schema_version") != 3:
        raise SystemExit("performance schema drifted")
    if performance.get("git_sha") != MEASUREMENT_SHA:
        raise SystemExit("performance measurement SHA drifted")
    if performance.get("working_tree_clean") is not True:
        raise SystemExit("performance source is not clean-tree evidence")
    if performance.get("benchmark") != "panel_stage_c_covariance_fit_overhead":
        raise SystemExit("performance benchmark identity drifted")
    if performance.get("timing_scope") != "synchronized end-to-end estimator fit":
        raise SystemExit("performance timing scope drifted")
    if performance.get("high_t_scale") != "10000x2x200":
        raise SystemExit("performance high-T scale drifted")
    if performance.get("two_way_unbalanced_scale") != "10000x2x20":
        raise SystemExit("performance two-way scale drifted")
    penv = performance.get("environment", {})
    if penv.get("gpu_by_backend") != {
        "cupy": "Tesla P100-SXM2-16GB",
        "torch": "Tesla P100-SXM2-16GB",
    }:
        raise SystemExit("performance GPU provenance drifted")
    ppackages = penv.get("packages", {})
    if ppackages.get("cupy") != "13.6.0" or ppackages.get("torch") != "2.0.0":
        raise SystemExit("performance CuPy/Torch provenance drifted")
    rows = performance.get("rows", [])
    if len(rows) != 60:
        raise SystemExit(f"performance row count drifted: {len(rows)}")
    scenario_counts = {"base": 0, "high_t_qs": 0, "two_way_unbalanced": 0}
    backend_counts = {"cupy": 0, "torch": 0}
    for row in rows:
        backend = row.get("backend")
        scenario = row.get("scenario")
        if backend not in backend_counts or scenario not in scenario_counts:
            raise SystemExit("performance backend/scenario drifted")
        backend_counts[backend] += 1
        scenario_counts[scenario] += 1
        samples = row.get("samples_seconds")
        if row.get("repeats") != 3 or not isinstance(samples, list) or len(samples) != 3:
            raise SystemExit("performance raw-sample contract drifted")
        numeric = [float(v) for v in samples]
        if any(not math.isfinite(v) or v <= 0 for v in numeric):
            raise SystemExit("performance sample is not finite/positive")
        if float(row.get("median_seconds")) != sorted(numeric)[1]:
            raise SystemExit("performance median does not match raw samples")
    if backend_counts != {"cupy": 30, "torch": 30}:
        raise SystemExit(f"performance backend matrix drifted: {backend_counts}")
    if scenario_counts != {"base": 54, "high_t_qs": 4, "two_way_unbalanced": 2}:
        raise SystemExit(f"performance scenario matrix drifted: {scenario_counts}")

    corr_sha = sha256(CORRECTNESS)
    perf_sha = sha256(PERFORMANCE)
    corr_blob = subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{CORRECTNESS.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        text=True,
    ).strip()
    perf_blob = subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{PERFORMANCE.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        text=True,
    ).strip()
    return corr_sha, perf_sha, corr_blob, perf_blob


def create_v5_parser() -> None:
    v4_path = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c_identifiability.py"
    v5_path = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c_final.py"
    if v5_path.exists():
        raise SystemExit("v5 parser already exists")
    parser = v4_path.read_text(encoding="utf-8")
    replacements = [
        (
            "Canonical v4 parsers for PR #126 post-identifiability P100 evidence.",
            "Canonical v5 parsers for PR #126 final fresh-P100 acceptance evidence.",
        ),
        ("fresh ``a99726e1`` evidence", "fresh ``5f0cea92`` evidence"),
        ('_SOURCE_DATE = "2026-08-12"', '_SOURCE_DATE = "2026-08-13"'),
        (
            '_MEASUREMENT_SHA = "a99726e19c535dfcd0a94711bbc8be6aac437584"',
            '_MEASUREMENT_SHA = "5f0cea9216321361842bc3c438219084a4cf5538"',
        ),
        (
            '_VALIDATION_PARSER = "parse_panel_stage_c_identifiability_physical_validation_v4"',
            '_VALIDATION_PARSER = "parse_panel_stage_c_final_physical_validation_v5"',
        ),
        (
            '_PERFORMANCE_PARSER = "parse_panel_stage_c_identifiability_performance_v4"',
            '_PERFORMANCE_PARSER = "parse_panel_stage_c_final_performance_v5"',
        ),
        ('_PARSER_VERSION = "4.0"', '_PARSER_VERSION = "5.0"'),
        (
            "def parse_panel_stage_c_identifiability_physical_validation(",
            "def parse_panel_stage_c_final_physical_validation(",
        ),
        (
            "def parse_panel_stage_c_identifiability_performance(",
            "def parse_panel_stage_c_final_performance(",
        ),
        ("PR126 identifiability Stage-C", "PR126 final Stage-C"),
        ("pr126-identifiability-validation", "pr126-final-validation"),
        ("pr126-identifiability-performance", "pr126-final-performance"),
        ("stage-c-identifiability-primitive", "stage-c-final-primitive"),
        ("stage-c-identifiability-public-primitive", "stage-c-final-public-primitive"),
        ("stage-c-identifiability-performance", "stage-c-final-performance"),
        ('"stage-c-identifiability"', '"stage-c-final"'),
        ('f"identifiability-public-{primitive_name.replace', 'f"final-public-{primitive_name.replace'),
        ('f"identifiability-{case_name.replace', 'f"final-{case_name.replace'),
    ]
    for old, new in replacements:
        if old not in parser:
            raise SystemExit(f"v5 parser template anchor missing: {old!r}")
        parser = parser.replace(old, new)

    old_set = '''_PREDICTION_BACKEND_CASES = {
    "panel_entity_hc0",
    "random_effects_explicit_constant_hc0",
}'''
    new_set = '''_PREDICTION_BACKEND_CASES = {
    "panel_entity_hc0",
    "panel_two_way_hc3",
    "random_effects_explicit_constant_hc0",
}'''
    parser = replace_once(parser, old_set, new_set, "v5 prediction backend set")

    anchor = '''        backend_ok = result.get("status") == "success" and result.get("requested_backend") == backend

        boundary = cases["panel_rank_boundary_dk"].get("covariance_metadata", {})'''
    injected = '''        backend_ok = result.get("status") == "success" and result.get("requested_backend") == backend

        prediction_contracts = result.get("prediction_contracts", {})
        expected_prediction_contracts = {
            "two_way_disconnected",
            "two_way_connected_partial_labels",
        }
        if set(prediction_contracts) != expected_prediction_contracts:
            raise ValueError(f"{backend}: PR126 final prediction-contract identity drifted")
        for prediction_name, contract in prediction_contracts.items():
            guards = contract.get("guards")
            differences = contract.get("max_abs_differences", {})
            if (
                contract.get("status") != "success"
                or contract.get("executed_backend") != backend
                or contract.get("prediction_backend") != backend
                or not isinstance(guards, dict)
                or not guards
                or not all(value is True for value in guards.values())
                or not _finite_diff_map(differences)
            ):
                raise ValueError(
                    f"{backend}: {prediction_name} final prediction contract drifted"
                )

        level_constant = result.get("level_constant_contract", {})
        if (
            level_constant.get("status") != "success"
            or level_constant.get("executed_backend") != backend
            or level_constant.get("prediction_backend") != backend
            or level_constant.get("constant_index") != 0
            or level_constant.get("constant_value") != 1.0
            or not _finite_diff_map(
                level_constant.get("max_abs_differences_vs_numpy", {})
            )
        ):
            raise ValueError(f"{backend}: PR126 final level-constant contract drifted")
        fresh_contract_ok = True

        boundary = cases["panel_rank_boundary_dk"].get("covariance_metadata", {})'''
    parser = replace_once(parser, anchor, injected, "v5 contract insertion")
    check_anchor = '                _bool_check("source_status_success", source_ok),\n'
    if parser.count(check_anchor) < 2:
        raise SystemExit("v5 validation check anchors missing")
    parser = parser.replace(
        check_anchor,
        check_anchor
        + '                _bool_check("fresh_prediction_and_level_contracts", fresh_contract_ok),\n',
        2,
    )
    v5_path.write_text(parser, encoding="utf-8")


def register_parser() -> None:
    init_path = ROOT / "dev/benchmarks/frontend_data/parsers/__init__.py"
    text = init_path.read_text(encoding="utf-8")
    anchor = '''from .panel_stage_c_identifiability import (
    parse_panel_stage_c_identifiability_physical_validation,
    parse_panel_stage_c_identifiability_performance,
)
'''
    addition = anchor + '''from .panel_stage_c_final import (
    parse_panel_stage_c_final_physical_validation,
    parse_panel_stage_c_final_performance,
)
'''
    text = replace_once(text, anchor, addition, "parser __init__ import")
    all_anchor = '    "parse_panel_stage_c_identifiability_performance",\n'
    text = replace_once(
        text,
        all_anchor,
        all_anchor
        + '    "parse_panel_stage_c_final_physical_validation",\n'
        + '    "parse_panel_stage_c_final_performance",\n',
        "parser __all__",
    )
    init_path.write_text(text, encoding="utf-8")

    registry_path = ROOT / "dev/benchmarks/frontend_data/registry.py"
    registry = registry_path.read_text(encoding="utf-8")
    import_anchor = '''from .parsers import (
    parse_panel_stage_c_identifiability_physical_validation,
    parse_panel_stage_c_identifiability_performance,
)
'''
    import_add = import_anchor + '''from .parsers import (
    parse_panel_stage_c_final_physical_validation,
    parse_panel_stage_c_final_performance,
)
'''
    registry = replace_once(registry, import_anchor, import_add, "registry import")
    map_anchor = '''PARSER_FUNCTIONS.update(
    {
        "panel_stage_c_identifiability_physical_validation": (
            parse_panel_stage_c_identifiability_physical_validation
        ),
        "panel_stage_c_identifiability_performance": (
            parse_panel_stage_c_identifiability_performance
        ),
    }
)
'''
    map_add = map_anchor + '''PARSER_FUNCTIONS.update(
    {
        "panel_stage_c_final_physical_validation": (
            parse_panel_stage_c_final_physical_validation
        ),
        "panel_stage_c_final_performance": parse_panel_stage_c_final_performance,
    }
)
'''
    registry = replace_once(registry, map_anchor, map_add, "registry map")
    registry_path.write_text(registry, encoding="utf-8")


def register_sources(corr_sha: str, perf_sha: str, corr_blob: str, perf_blob: str) -> tuple[str, str]:
    manifest_path = ROOT / "dev/benchmarks/frontend_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if ENV_ID in manifest["environments"]:
        raise SystemExit("v5 environment identity already exists")
    manifest["environments"][ENV_ID] = {
        "label": "Tesla P100 PR #126 final fresh acceptance — 2026-08-13",
        "gpu": "Tesla P100-SXM2-16GB",
        "cpu": "x86_64",
    }
    validation_comparison = "panel-stage-c-pr126-final-validation-20260813"
    performance_comparison = "panel-stage-c-pr126-final-performance-20260813"
    manifest["comparisons"][validation_comparison] = {
        "label": "Panel Stage C final physical acceptance — PR #126 — 2026-08-13",
        "env_id": ENV_ID,
    }
    manifest["comparisons"][performance_comparison] = {
        "label": "Panel Stage C final synchronized performance — PR #126 — 2026-08-13",
        "env_id": ENV_ID,
    }
    validation_source_id = f"panel-stage-c-final-validation-pr126-20260813-{corr_sha[:12]}"
    performance_source_id = f"panel-stage-c-final-performance-pr126-20260813-{perf_sha[:12]}"
    existing_ids = {src["source_id"] for src in manifest["sources"]}
    if validation_source_id in existing_ids or performance_source_id in existing_ids:
        raise SystemExit("v5 source identity already exists")
    manifest["sources"].extend(
        [
            {
                "source_id": validation_source_id,
                "comparison_id": validation_comparison,
                "path": CORRECTNESS.relative_to(ROOT).as_posix(),
                "sha256": corr_sha,
                "parser": "panel_stage_c_final_physical_validation",
                "parser_version": "5.0",
                "env_id": ENV_ID,
                "required": True,
                "allowed_issue_codes": [],
                "source_date": SOURCE_DATE,
                "measurement_git_sha": MEASUREMENT_SHA,
                "raw_git_sha": MEASUREMENT_SHA,
                "provenance_note": (
                    f"PR #126 final exact-clean fresh P100 correctness/backend-provenance evidence. "
                    f"Measured source {MEASUREMENT_SHA}; raw artifact commit {ARTIFACT_COMMIT}; "
                    f"Git blob {corr_blob}; SHA-256 {corr_sha}. CuPy 13.6.0 and Torch 2.0.0 "
                    "each pass 35 estimator integrations plus 12 public covariance primitives "
                    "(47/47 per backend), connected/disconnected two-way prediction guards, "
                    "level-constant inference/prediction parity, and requested/executed backend "
                    "provenance. Historical v1-v4 Stage-C sources remain immutable audit evidence."
                ),
            },
            {
                "source_id": performance_source_id,
                "comparison_id": performance_comparison,
                "path": PERFORMANCE.relative_to(ROOT).as_posix(),
                "sha256": perf_sha,
                "parser": "panel_stage_c_final_performance",
                "parser_version": "5.0",
                "env_id": ENV_ID,
                "required": True,
                "allowed_issue_codes": [],
                "source_date": SOURCE_DATE,
                "measurement_git_sha": MEASUREMENT_SHA,
                "raw_git_sha": MEASUREMENT_SHA,
                "provenance_note": (
                    f"PR #126 final exact-clean fresh P100 synchronized timing evidence. "
                    f"Measured source {MEASUREMENT_SHA}; raw artifact commit {ARTIFACT_COMMIT}; "
                    f"Git blob {perf_blob}; SHA-256 {perf_sha}. The immutable source contains "
                    "60 synchronized rows: 54 base, four bounded N=10000 k=2 T=200 QS, and "
                    "two N=10000 k=2 T=20 unbalanced two-way-FE rows, with three finite positive "
                    "samples per row and exact medians. No CPU speedup claim is encoded."
                ),
            },
        ]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return validation_source_id, performance_source_id


def update_coverage(validation_source_id: str, performance_source_id: str) -> None:
    path = ROOT / "dev/benchmarks/benchmark_coverage_matrix.json"
    coverage = json.loads(path.read_text(encoding="utf-8"))
    panel = next(row for row in coverage["capabilities"] if row["capability_id"] == "panel-estimation")
    panel["source_ids"].extend([validation_source_id, performance_source_id])
    panel["disposition"] = (
        "June timing rows and PR #122 Stage-B diagnostics remain canonical historical/current "
        "coverage. PR #126 final Stage-C physical acceptance is now backed by the immutable fresh "
        f"P100 v5 pair measured on {MEASUREMENT_SHA}: CuPy and Torch each pass 35 estimator "
        "integrations plus 12 public primitives (47/47), including connected and disconnected "
        "two-way prediction identifiability guards, level-constant parity, and backend provenance. "
        "Synchronized performance contains 60 rows (54 base, four high-T QS, two unbalanced "
        "two-way-FE). Historical Stage-C v1-v4 sources remain immutable audit evidence and are not "
        "overwritten. No Stage-C speedup claim is made."
    )
    path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")


def update_ci_and_tests() -> None:
    workflow_path = ROOT / ".github/workflows/benchmark-frontend.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    trigger = "      - 'results/pr126_p100/**'\n"
    if workflow.count(trigger) != 2:
        raise SystemExit("benchmark frontend trigger anchor count drifted")
    workflow = workflow.replace(trigger, trigger + "      - 'results/pr126_p100_fresh/**'\n")
    workflow_path.write_text(workflow, encoding="utf-8")

    test_path = ROOT / "dev/tests/test_panel_stage_c_frontend_source.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''

# PR126 final fresh-P100 v5 canonical source
from dev.benchmarks.frontend_data.parsers.panel_stage_c_final import (
    parse_panel_stage_c_final_performance,
    parse_panel_stage_c_final_physical_validation,
)

FINAL_CORRECTNESS = ROOT / "results/pr126_p100_fresh/panel_stage_c_correctness_p100.json"
FINAL_PERFORMANCE = ROOT / "results/pr126_p100_fresh/panel_stage_c_performance_p100.json"
FINAL_ENV_ID = "remote-p100-pr126-final-20260813"


def test_final_stage_c_v5_validation_emits_47_checks_per_backend():
    runs, models, warnings = parse_panel_stage_c_final_physical_validation(
        FINAL_CORRECTNESS, FINAL_ENV_ID
    )
    assert warnings == []
    assert len(runs) == 94
    assert {run["backend"] for run in runs} == {"cupy", "torch"}
    assert sum(run["model_id"] == "PanelCovariancePrimitive" for run in runs) == 24
    assert all(run["metrics"]["validation"]["status"] == "pass" for run in runs)
    assert all(
        run["parameters"]["measurement_git_sha"]
        == "5f0cea9216321361842bc3c438219084a4cf5538"
        for run in runs
    )
    assert {model["model_id"] for model in models} == {
        "PooledOLS", "PanelOLS", "RandomEffects", "BetweenOLS",
        "FirstDifferenceOLS", "PanelCovariancePrimitive",
    }


def test_final_stage_c_v5_performance_emits_60_synchronized_rows():
    runs, models, warnings = parse_panel_stage_c_final_performance(
        FINAL_PERFORMANCE, FINAL_ENV_ID
    )
    assert warnings == []
    assert len(runs) == 60
    assert len([run for run in runs if run["parameters"]["scenario"] == "base"]) == 54
    assert len([run for run in runs if run["parameters"]["scenario"] == "high_t_qs"]) == 4
    assert len([run for run in runs if run["parameters"]["scenario"] == "two_way_unbalanced"]) == 2
    assert all(run["metrics"]["timing"]["fit_time_ms"] > 0 for run in runs)
    assert all("speedup" not in run["metrics"] for run in runs)
    assert {model["model_id"] for model in models} == {
        "PooledOLS", "PanelOLS", "RandomEffects"
    }


def test_final_stage_c_v5_fails_closed_on_prediction_guard_drift(tmp_path):
    data = json.loads(FINAL_CORRECTNESS.read_text(encoding="utf-8"))
    data["backends"]["cupy"]["prediction_contracts"]["two_way_disconnected"]["guards"][
        "cross_component"
    ] = False
    broken = tmp_path / "broken_final_prediction.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction contract drifted"):
        parse_panel_stage_c_final_physical_validation(broken, FINAL_ENV_ID)


def test_final_stage_c_v5_fails_closed_on_level_constant_backend_drift(tmp_path):
    data = json.loads(FINAL_CORRECTNESS.read_text(encoding="utf-8"))
    data["backends"]["torch"]["level_constant_contract"]["prediction_backend"] = "numpy"
    broken = tmp_path / "broken_final_level_constant.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="level-constant contract drifted"):
        parse_panel_stage_c_final_physical_validation(broken, FINAL_ENV_ID)
'''
    test_path.write_text(tests, encoding="utf-8")


def update_changelog_and_checksums() -> None:
    path = ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    old = "- Added maintained Python/R external-definition checks plus exact physical CuPy/Torch correctness and synchronized performance validation."
    new = (
        "- Added maintained Python/R external-definition checks plus exact physical CuPy/Torch correctness and synchronized performance validation; "
        "final fresh Tesla P100 acceptance on `5f0cea9216321361842bc3c438219084a4cf5538` passes 47/47 checks per backend and the synchronized 60-row performance matrix under immutable v5 source identities."
    )
    text = replace_once(text, old, new, "CHANGELOG PR126 physical acceptance")
    path.write_text(text, encoding="utf-8")

    raw_names = [
        "environment.txt",
        "panel_stage_c_correctness_p100.json",
        "panel_stage_c_correctness_p100.log",
        "panel_stage_c_performance_p100.json",
        "panel_stage_c_performance_p100.log",
        "validation_summary.txt",
    ]
    sums = [f"{sha256(RAW_DIR / name)}  {name}" for name in raw_names]
    (RAW_DIR / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def main() -> None:
    corr_sha, perf_sha, corr_blob, perf_blob = audit_raw_evidence()
    create_v5_parser()
    register_parser()
    validation_id, performance_id = register_sources(corr_sha, perf_sha, corr_blob, perf_blob)
    update_coverage(validation_id, performance_id)
    update_ci_and_tests()
    update_changelog_and_checksums()
    print("V5_VALIDATION_SOURCE_ID=" + validation_id)
    print("V5_PERFORMANCE_SOURCE_ID=" + performance_id)
    print("CORRECTNESS_SHA256=" + corr_sha)
    print("PERFORMANCE_SHA256=" + perf_sha)


if __name__ == "__main__":
    main()
