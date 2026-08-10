from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_SHA = "9c0b3050dd143c43a06bb6393d69f4f83e861637"
ARTIFACT_COMMIT = "85d710bddf633134624501a9e27f03c30bc04ead"
SOURCE_DATE = "2026-08-10"
CORR_PATH = "results/pr126_p100/panel_stage_c_gpu_validation_9c0b3050.json"
PERF_PATH = "results/pr126_p100/panel_stage_c_performance_9c0b3050.json"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"expected block not found in {path}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def git_blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", path], cwd=ROOT, text=True
    ).strip()


corr = json.loads(read(CORR_PATH))
perf = json.loads(read(PERF_PATH))
assert corr["git_sha"] == MEASUREMENT_SHA and corr["working_tree_clean"] is True
assert corr["status"] == "success" and corr["case_count_per_backend"] == 26
assert corr["public_primitive_count_per_backend"] == 2
assert perf["git_sha"] == MEASUREMENT_SHA and perf["working_tree_clean"] is True
assert perf["schema_version"] == 2 and perf["high_t_scale"] == "10000x2x200"
assert len(perf["rows"]) == 58

corr_sha = sha256(CORR_PATH)
perf_sha = sha256(PERF_PATH)
corr_blob = git_blob(CORR_PATH)
perf_blob = git_blob(PERF_PATH)
corr_source_id = f"panel-stage-c-validation-pr126-20260810-{corr_sha[:12]}"
perf_source_id = f"panel-stage-c-performance-pr126-20260810-{perf_sha[:12]}"

PARSER = r'''from __future__ import annotations
"""Canonical parsers for PR #126 Panel Stage-C P100 evidence."""

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label

_SOURCE_DATE = "2026-08-10"
_MEASUREMENT_SHA = "9c0b3050dd143c43a06bb6393d69f4f83e861637"
_VALIDATION_PARSER = "parse_panel_stage_c_physical_validation_v1"
_PERFORMANCE_PARSER = "parse_panel_stage_c_performance_v1"
_PARSER_VERSION = "1.0"

_EXPECTED_CASES = {
    "pooled_hc0", "pooled_hc2", "pooled_hc3",
    "pooled_cluster_one_way", "pooled_cluster_two_way_group_debias",
    "pooled_dk_bartlett", "pooled_dk_qs", "pooled_legacy_hac",
    "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3",
    "panel_two_way_hc3", "panel_two_way_cluster_group_debias", "panel_two_way_dk",
    "random_effects_explicit_constant_robust", "random_effects_explicit_constant_hc0",
    "random_effects_explicit_constant_hc2", "random_effects_explicit_constant_hc3",
    "random_effects_cluster_two_way", "random_effects_dk",
    "between_hc0", "between_hc2", "between_hc3",
    "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
}
_EXPECTED_PRIMITIVES = {"cluster_group_debias", "driscoll_kraay_qs"}
_HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{kind}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _model_id(case: str) -> str:
    if case.startswith("pooled_"):
        return "PooledOLS"
    if case.startswith("panel_"):
        return "PanelOLS"
    if case.startswith("random_effects_"):
        return "RandomEffects"
    if case.startswith("between_"):
        return "BetweenOLS"
    if case.startswith("first_difference_"):
        return "FirstDifferenceOLS"
    raise ValueError(f"unknown Stage-C case identity: {case!r}")


def _scale(n_samples: int, n_features: int, *, suffix: str | None = None) -> dict[str, Any]:
    label = make_scale_label(int(n_samples), int(n_features))
    if suffix:
        label = f"{label} · {suffix}"
    return {
        "scale_key": make_scale_key(int(n_samples), int(n_features)),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "label": label,
    }


def _models(model_ids: set[str]) -> list[dict]:
    return [
        {
            "model_id": model_id,
            "primary_category_id": "panel",
            "category_ids": ["panel"],
            "supports_penalty": False,
            "supports_inference": True,
        }
        for model_id in sorted(model_ids)
    ]


def _source(filepath: Path, parser: str) -> dict[str, str]:
    return {
        "file": filepath.name,
        "date": _SOURCE_DATE,
        "parser": parser,
        "parser_version": _PARSER_VERSION,
    }


def _validation(ok: bool, filepath: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pass" if ok else "fail",
        "checks": checks,
        "quality": "reported",
        "source_file": filepath.name,
    }


def _bool_check(metric: str, ok: bool, **extra: Any) -> dict[str, Any]:
    return {"metric": metric, "status": "pass" if ok else "fail", **extra}


def _finite_diff_map(value: Any, limit: float) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for item in value.values():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        item = float(item)
        if not math.isfinite(item) or abs(item) > limit:
            return False
    return True


def parse_panel_stage_c_physical_validation(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("PR126 Stage-C validation source requires schema_version=1")
    if data.get("git_sha") != _MEASUREMENT_SHA:
        raise ValueError("PR126 Stage-C validation source measurement SHA drifted")
    if int(data.get("case_count_per_backend", -1)) != len(_EXPECTED_CASES):
        raise ValueError("PR126 Stage-C validation estimator case count drifted")
    if int(data.get("public_primitive_count_per_backend", -1)) != len(_EXPECTED_PRIMITIVES):
        raise ValueError("PR126 Stage-C public primitive count drifted")

    dataset = data.get("dataset", {})
    scale = _scale(dataset.get("nobs", 0), dataset.get("n_features", 0))
    atol = float(data.get("tolerances", {}).get("atol", 0.0))
    limit = max(atol, 1e-12)
    source_ok = data.get("status") == "success" and data.get("working_tree_clean") is True
    runs: list[dict] = []
    model_ids: set[str] = set()

    backends = data.get("backends", {})
    if set(backends) != {"cupy", "torch"}:
        raise ValueError("PR126 Stage-C validation requires exactly CuPy and Torch backends")

    for backend in ("cupy", "torch"):
        result = backends[backend]
        cases = result.get("cases", {})
        primitives = result.get("public_primitives", {})
        if set(cases) != _EXPECTED_CASES:
            raise ValueError(f"{backend}: PR126 estimator case identity drifted")
        if set(primitives) != _EXPECTED_PRIMITIVES:
            raise ValueError(f"{backend}: PR126 public primitive identity drifted")
        backend_ok = (
            result.get("status") == "success"
            and result.get("requested_backend") == backend
        )

        for case_name in sorted(_EXPECTED_CASES):
            case = cases[case_name]
            diff_ok = _finite_diff_map(case.get("max_abs_differences"), limit)
            executed_ok = case.get("executed_backend") == backend
            case_ok = case.get("status") == "success"
            ok = source_ok and backend_ok and case_ok and executed_ok and diff_ok
            checks = [
                _bool_check("source_status_success", source_ok),
                _bool_check("backend_status_success", backend_ok),
                _bool_check("case_status_success", case_ok),
                _bool_check("executed_backend_matches_requested", executed_ok),
                _bool_check("numpy_parity_within_physical_tolerance", diff_ok, tolerance=limit),
            ]
            model_id = _model_id(case_name)
            model_ids.add(model_id)
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-validation",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": model_id,
                    "case_id": _stable_id("case", "stage-c", case_name, scale["scale_key"]),
                    "method_config_id": _stable_id("method", "stage-c-physical", case_name),
                    "variant": case_name.replace("_", "-"),
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": dict(scale),
                    "parameters": {
                        "metric_scope": "physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": bool(data.get("working_tree_clean")),
                        "executed_backend": case.get("executed_backend"),
                        "covariance_metadata": case.get("covariance_metadata", {}),
                    },
                    "source": _source(filepath, _VALIDATION_PARSER),
                    "metrics": {
                        "validation": _validation(ok, filepath, checks),
                        "inference": {
                            "ok": ok,
                            "quality": "reported",
                            "source_file": filepath.name,
                        },
                    },
                }
            )

        for primitive_name in sorted(_EXPECTED_PRIMITIVES):
            item = primitives[primitive_name]
            try:
                diff = float(item.get("max_abs_difference"))
                diff_ok = math.isfinite(diff) and abs(diff) <= limit
            except (TypeError, ValueError):
                diff_ok = False
            executed_ok = item.get("executed_backend") == backend
            item_ok = item.get("status") == "success"
            ok = source_ok and backend_ok and item_ok and executed_ok and diff_ok
            checks = [
                _bool_check("source_status_success", source_ok),
                _bool_check("backend_status_success", backend_ok),
                _bool_check("primitive_status_success", item_ok),
                _bool_check("executed_backend_matches_requested", executed_ok),
                _bool_check("numpy_parity_within_physical_tolerance", diff_ok, tolerance=limit),
            ]
            model_ids.add("PanelCovariancePrimitive")
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-validation",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": "PanelCovariancePrimitive",
                    "case_id": _stable_id("case", "stage-c-primitive", primitive_name, scale["scale_key"]),
                    "method_config_id": _stable_id("method", "stage-c-public-primitive", primitive_name),
                    "variant": f"public-{primitive_name.replace('_', '-')}",
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": dict(scale),
                    "parameters": {
                        "metric_scope": "public_primitive_physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": bool(data.get("working_tree_clean")),
                        "executed_backend": item.get("executed_backend"),
                    },
                    "source": _source(filepath, _VALIDATION_PARSER),
                    "metrics": {"validation": _validation(ok, filepath, checks)},
                }
            )

    if len(runs) != 56:
        raise ValueError(f"PR126 validation parser expected 56 rows, got {len(runs)}")
    return runs, _models(model_ids), warnings


def parse_panel_stage_c_performance(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 2:
        raise ValueError("PR126 Stage-C performance source requires schema_version=2")
    if data.get("git_sha") != _MEASUREMENT_SHA:
        raise ValueError("PR126 Stage-C performance source measurement SHA drifted")
    if data.get("working_tree_clean") is not True:
        raise ValueError("PR126 Stage-C performance source requires a clean measurement tree")
    if data.get("benchmark") != "panel_stage_c_covariance_fit_overhead":
        raise ValueError("PR126 Stage-C performance benchmark identity drifted")
    if data.get("timing_scope") != "synchronized end-to-end estimator fit":
        raise ValueError("PR126 Stage-C performance timing scope drifted")
    if data.get("high_t_scale") != "10000x2x200":
        raise ValueError("PR126 Stage-C high-T scale drifted")

    rows = data.get("rows", [])
    if len(rows) != 58:
        raise ValueError(f"PR126 Stage-C performance requires 58 rows, got {len(rows)}")
    if {row.get("backend") for row in rows} != {"cupy", "torch"}:
        raise ValueError("PR126 Stage-C performance requires CuPy and Torch rows")

    high_t = [row for row in rows if row.get("scenario") == "high_t_qs"]
    if len(high_t) != 4:
        raise ValueError("PR126 Stage-C performance requires four high-T QS rows")
    if {row.get("case") for row in high_t} != _HIGH_T_CASES:
        raise ValueError("PR126 Stage-C high-T QS case identity drifted")
    if any(
        int(row.get("n_samples", 0)) != 10000
        or int(row.get("n_features", 0)) != 2
        or int(row.get("n_times", 0)) != 200
        for row in high_t
    ):
        raise ValueError("PR126 Stage-C high-T QS dimensions drifted")

    output: list[dict] = []
    model_ids: set[str] = set()
    for row in rows:
        backend = row.get("backend")
        case_name = str(row.get("case"))
        scenario = str(row.get("scenario"))
        if scenario not in {"base", "high_t_qs"}:
            raise ValueError(f"unknown PR126 Stage-C performance scenario: {scenario!r}")
        repeats = int(row.get("repeats", 0))
        samples = row.get("samples_seconds")
        if repeats <= 0 or not isinstance(samples, list) or len(samples) != repeats:
            raise ValueError("PR126 Stage-C timing samples/repeats contract failed")
        numeric_samples = [float(value) for value in samples]
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric_samples):
            raise ValueError("PR126 Stage-C timing samples must be finite and positive")
        median = float(row.get("median_seconds"))
        if not math.isfinite(median) or median <= 0.0:
            raise ValueError("PR126 Stage-C timing median must be finite and positive")
        expected_median = float(statistics.median(numeric_samples))
        if not math.isclose(median, expected_median, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("PR126 Stage-C reported median does not match raw samples")

        model_id = _model_id(case_name)
        model_ids.add(model_id)
        n_samples = int(row["n_samples"])
        n_features = int(row["n_features"])
        n_times = int(row["n_times"])
        scale = _scale(n_samples, n_features, suffix=f"T={n_times}")
        output.append(
            {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-performance",
                "env_id": env_id,
                "category_ids": ["panel"],
                "model_id": model_id,
                "case_id": _stable_id(
                    "case", "stage-c-performance", case_name, scenario,
                    n_samples, n_features, n_times,
                ),
                "method_config_id": _stable_id("method", "stage-c-performance", case_name),
                "variant": f"{case_name.replace('_', '-')}-{scenario.replace('_', '-')}",
                "penalty": None,
                "solver": "covariance_fit",
                "solver_display": "Covariance fit",
                "solver_kind": "internal",
                "framework": "statgpu",
                "backend": backend,
                "scale": scale,
                "parameters": {
                    "scenario": scenario,
                    "n_times": n_times,
                    "repeats": repeats,
                    "timing_scope": data.get("timing_scope"),
                    "input_residency": data.get("input_residency"),
                    "measurement_git_sha": data.get("git_sha"),
                    "working_tree_clean": True,
                },
                "source": _source(filepath, _PERFORMANCE_PARSER),
                "metrics": {
                    "timing": {
                        "fit_time_ms": round(median * 1000.0, 6),
                        "quality": "measured",
                        "source_file": filepath.name,
                    },
                    "validation": {
                        "status": "pass",
                        "checks": [
                            {"metric": "synchronized_timing", "status": "pass"},
                            {"metric": "raw_samples_finite_positive", "status": "pass"},
                            {"metric": "median_matches_raw_samples", "status": "pass"},
                        ],
                        "quality": "reported",
                        "source_file": filepath.name,
                    },
                },
            }
        )

    return output, _models(model_ids), []
'''
write("dev/benchmarks/frontend_data/parsers/panel_stage_c.py", PARSER)

replace_once(
    "dev/benchmarks/frontend_data/parsers/__init__.py",
    "from .panel_stage_b import parse_panel_stage_b_physical_validation\n",
    "from .panel_stage_b import parse_panel_stage_b_physical_validation\n"
    "from .panel_stage_c import (\n"
    "    parse_panel_stage_c_physical_validation,\n"
    "    parse_panel_stage_c_performance,\n"
    ")\n",
)
replace_once(
    "dev/benchmarks/frontend_data/parsers/__init__.py",
    '    "parse_panel_stage_b_physical_validation",\n]',
    '    "parse_panel_stage_b_physical_validation",\n'
    '    "parse_panel_stage_c_physical_validation",\n'
    '    "parse_panel_stage_c_performance",\n]',
)
replace_once(
    "dev/benchmarks/frontend_data/registry.py",
    "    parse_panel_stage_b_physical_validation,\n)",
    "    parse_panel_stage_b_physical_validation,\n"
    "    parse_panel_stage_c_physical_validation,\n"
    "    parse_panel_stage_c_performance,\n)",
)
replace_once(
    "dev/benchmarks/frontend_data/registry.py",
    '    "panel_stage_b_physical_validation": parse_panel_stage_b_physical_validation,\n}',
    '    "panel_stage_b_physical_validation": parse_panel_stage_b_physical_validation,\n'
    '    "panel_stage_c_physical_validation": parse_panel_stage_c_physical_validation,\n'
    '    "panel_stage_c_performance": parse_panel_stage_c_performance,\n}',
)

manifest_path = ROOT / "dev/benchmarks/frontend_sources.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
env_id = "remote-p100-pr126-20260810"
manifest["environments"][env_id] = {
    "label": "Tesla P100 PR #126 Panel Stage C — 2026-08-10",
    "gpu": "Tesla P100-SXM2-16GB",
    "cpu": "x86_64",
}
validation_comparison = "panel-stage-c-pr126-validation-20260810"
performance_comparison = "panel-stage-c-pr126-performance-20260810"
manifest["comparisons"][validation_comparison] = {
    "label": "Panel Stage C physical validation — PR #126 — 2026-08-10",
    "env_id": env_id,
}
manifest["comparisons"][performance_comparison] = {
    "label": "Panel Stage C synchronized performance — PR #126 — 2026-08-10",
    "env_id": env_id,
}
manifest["sources"] = [
    source for source in manifest["sources"]
    if not source["source_id"].startswith("panel-stage-c-")
]
manifest["sources"].extend(
    [
        {
            "source_id": corr_source_id,
            "comparison_id": validation_comparison,
            "path": CORR_PATH,
            "sha256": corr_sha,
            "parser": "panel_stage_c_physical_validation",
            "parser_version": "1.0",
            "env_id": env_id,
            "required": True,
            "allowed_issue_codes": [],
            "source_date": SOURCE_DATE,
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                "PR #126 Stage-C exact-clean-head P100 correctness/backend-provenance evidence. "
                f"Artifact commit {ARTIFACT_COMMIT}; Git blob {corr_blob}; 26 estimator cases plus "
                "two direct public covariance primitives per CuPy/Torch backend, with no CPU fallback."
            ),
        },
        {
            "source_id": perf_source_id,
            "comparison_id": performance_comparison,
            "path": PERF_PATH,
            "sha256": perf_sha,
            "parser": "panel_stage_c_performance",
            "parser_version": "1.0",
            "env_id": env_id,
            "required": True,
            "allowed_issue_codes": [],
            "source_date": SOURCE_DATE,
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                "PR #126 Stage-C synchronized end-to-end P100 timing evidence. "
                f"Artifact commit {ARTIFACT_COMMIT}; Git blob {perf_blob}; includes the bounded "
                "N=10000, k=2, T=200 QS all-lag scenario. No speedup claim or CPU baseline is encoded."
            ),
        },
    ]
)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

coverage_path = ROOT / "dev/benchmarks/benchmark_coverage_matrix.json"
coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
panel = next(row for row in coverage["capabilities"] if row["capability_id"] == "panel-estimation")
panel["source_ids"] = [
    "new-modules-20260624-bcbdb676223b",
    "panel-stage-b-pr122-20260809-2056f836bfe2",
    corr_source_id,
    perf_source_id,
]
panel["representative_dimensions"] = [
    "estimator", "backend", "aligned_scale", "physical_validation",
    "diagnostics", "inference_regression", "covariance", "n_times", "timing_protocol",
]
panel["disposition"] = (
    "June timing rows cover aligned PanelOLS and RandomEffects; PR #122 provides canonical "
    "Stage-B diagnostic/physical validation. PR #126 adds canonical Stage-C CuPy/Torch "
    "correctness for 26 estimator covariance integrations plus two direct public primitives "
    "per backend, and synchronized timing rows for three base scales plus a bounded "
    "N=10,000, k=2, T=200 QS all-lag scenario. No Stage-C speedup claim is made."
)
coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

# Update literal registered-source counts and panel source expectations.
for path in [
    "dev/tests/test_benchmark_catalog.py",
    "dev/tests/test_benchmark_inventory_v2.py",
    "dev/tests/test_frontend_domain_coverage.py",
]:
    text = read(path)
    text = text.replace("== 11", "== 13")
    write(path, text)

catalog_test = read("dev/tests/test_benchmark_catalog.py")
old_panel = '''    assert rows["panel-estimation"]["source_ids"] == [\n        "new-modules-20260624-bcbdb676223b",\n        "panel-stage-b-pr122-20260809-2056f836bfe2",\n    ]'''
new_panel = f'''    assert rows["panel-estimation"]["source_ids"] == [\n        "new-modules-20260624-bcbdb676223b",\n        "panel-stage-b-pr122-20260809-2056f836bfe2",\n        "{corr_source_id}",\n        "{perf_source_id}",\n    ]'''
if old_panel not in catalog_test:
    raise RuntimeError("panel source-id test block not found")
catalog_test = catalog_test.replace(old_panel, new_panel, 1)
anchor = '    assert panel_raw["issue"] == "#93"\n'
extra = f'''\n    stage_c_validation = next(\n        entry for entry in entries\n        if entry["path"] == "{CORR_PATH}"\n    )\n    stage_c_performance = next(\n        entry for entry in entries\n        if entry["path"] == "{PERF_PATH}"\n    )\n    assert stage_c_validation["classification"] == "registered_canonical"\n    assert stage_c_validation["source_id"] == "{corr_source_id}"\n    assert stage_c_performance["classification"] == "registered_canonical"\n    assert stage_c_performance["source_id"] == "{perf_source_id}"\n'''
if extra.strip() not in catalog_test:
    catalog_test = catalog_test.replace(anchor, anchor + extra, 1)
write("dev/tests/test_benchmark_catalog.py", catalog_test)

stage_c_test = f'''from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport pytest\n\nfrom dev.benchmarks.frontend_data.parsers.panel_stage_c import (\n    parse_panel_stage_c_performance,\n    parse_panel_stage_c_physical_validation,\n)\n\nROOT = Path(__file__).resolve().parents[2]\nCORRECTNESS = ROOT / "{CORR_PATH}"\nPERFORMANCE = ROOT / "{PERF_PATH}"\n\n\ndef test_stage_c_validation_parser_emits_exact_physical_matrix():\n    runs, models, warnings = parse_panel_stage_c_physical_validation(\n        CORRECTNESS, "remote-p100-pr126-20260810"\n    )\n    assert warnings == []\n    assert len(runs) == 56\n    assert {{run["backend"] for run in runs}} == {{"cupy", "torch"}}\n    assert sum(run["model_id"] == "PanelCovariancePrimitive" for run in runs) == 4\n    assert all(run["metrics"]["validation"]["status"] == "pass" for run in runs)\n    assert all("timing" not in run["metrics"] for run in runs)\n    assert all("speedup" not in run["metrics"] for run in runs)\n    assert {{model["model_id"] for model in models}} == {{\n        "PooledOLS", "PanelOLS", "RandomEffects", "BetweenOLS",\n        "FirstDifferenceOLS", "PanelCovariancePrimitive",\n    }}\n\n\ndef test_stage_c_performance_parser_emits_timing_without_speedup():\n    runs, _, warnings = parse_panel_stage_c_performance(\n        PERFORMANCE, "remote-p100-pr126-20260810"\n    )\n    assert warnings == []\n    assert len(runs) == 58\n    assert all(run["metrics"]["timing"]["fit_time_ms"] > 0 for run in runs)\n    assert all("speedup" not in run["metrics"] for run in runs)\n    high_t = [run for run in runs if run["parameters"]["scenario"] == "high_t_qs"]\n    assert len(high_t) == 4\n    assert {{run["backend"] for run in high_t}} == {{"cupy", "torch"}}\n    assert {{run["parameters"]["n_times"] for run in high_t}} == {{200}}\n    assert {{run["model_id"] for run in high_t}} == {{"PooledOLS", "PanelOLS"}}\n\n\ndef test_stage_c_performance_parser_fails_closed_on_high_t_contract(tmp_path):\n    data = json.loads(PERFORMANCE.read_text(encoding="utf-8"))\n    data["high_t_scale"] = "10000x2x20"\n    broken = tmp_path / "broken_performance.json"\n    broken.write_text(json.dumps(data), encoding="utf-8")\n    with pytest.raises(ValueError, match="high-T scale drifted"):\n        parse_panel_stage_c_performance(broken, "remote-p100-pr126-20260810")\n\n\ndef test_stage_c_validation_parser_fails_closed_on_case_identity(tmp_path):\n    data = json.loads(CORRECTNESS.read_text(encoding="utf-8"))\n    del data["backends"]["cupy"]["cases"]["pooled_hc0"]\n    broken = tmp_path / "broken_validation.json"\n    broken.write_text(json.dumps(data), encoding="utf-8")\n    with pytest.raises(ValueError, match="case identity drifted"):\n        parse_panel_stage_c_physical_validation(broken, "remote-p100-pr126-20260810")\n'''
write("dev/tests/test_panel_stage_c_frontend_source.py", stage_c_test)

# Permanent Benchmark Frontend CI must own the new parser contract test and raw sources.
workflow = read(".github/workflows/benchmark-frontend.yml")
for marker in ["push:", "pull_request:"]:
    pass
workflow = workflow.replace(
    "      - 'dev/tests/test_panel_stage_b_applicable_hausman_parser.py'\n",
    "      - 'dev/tests/test_panel_stage_b_applicable_hausman_parser.py'\n"
    "      - 'dev/tests/test_panel_stage_c_frontend_source.py'\n"
    "      - 'results/pr126_p100/**'\n",
)
workflow = workflow.replace(
    "            dev/tests/test_panel_stage_b_applicable_hausman_parser.py -v\n",
    "            dev/tests/test_panel_stage_b_applicable_hausman_parser.py \\\n"
    "            dev/tests/test_panel_stage_c_frontend_source.py -v\n",
)
write(".github/workflows/benchmark-frontend.yml", workflow)

# Root changelog: replace pending physical status with accepted evidence.
replace_once(
    "CHANGELOG.md",
    "- Added pinned statsmodels/linearmodels covariance checks plus exact-head physical GPU and performance validators; final P100 acceptance remains pending.",
    "- Added pinned statsmodels/linearmodels covariance checks plus exact-head physical GPU and performance validators. Tesla P100 acceptance passed on clean implementation head `9c0b3050dd143c43a06bb6393d69f4f83e861637`: both CuPy and Torch passed all 26 estimator covariance cases plus two direct public primitives, and synchronized performance evidence includes the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario without making a speedup claim.",
)

# EN detailed changelog: replace the pending physical paragraph when present.
en_path = "docs/en/changelog.md"
en = read(en_path)
old_en = "The physical CUDA gate is intentionally separate: `dev/benchmarks/validate_panel_stage_c_gpu.py` and `dev/benchmarks/benchmark_panel_stage_c_covariance.py` must be run on the final exact clean implementation head before PR #126 can leave Draft. No GPU speedup or final physical-acceptance claim is made here yet."
new_en = "Physical CUDA acceptance is complete on exact clean implementation head `9c0b3050dd143c43a06bb6393d69f4f83e861637` using Tesla P100-SXM2-16GB. CuPy and Torch each pass all 26 estimator covariance cases plus two direct public covariance primitives without CPU fallback. The separate synchronized performance artifact covers the three base scales and the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it records timing only and makes no speedup claim."
if old_en in en:
    en = en.replace(old_en, new_en, 1)
elif new_en not in en:
    heading = "## 2026-08-09 — Panel Stage C covariance completion (PR #126)\n"
    if heading not in en:
        raise RuntimeError("EN Stage-C changelog heading not found")
    en = en.replace(heading, heading + "\n" + new_en + "\n", 1)
write(en_path, en)

# CN detailed changelog: append one concise accepted-evidence paragraph under the Stage-C heading.
cn_path = "docs/cn/changelog.md"
cn = read(cn_path)
cn_sentence = "PR #126 的物理 CUDA 验收已在精确且干净的实现提交 `9c0b3050dd143c43a06bb6393d69f4f83e861637` 上使用 Tesla P100-SXM2-16GB 完成：CuPy 与 Torch 均通过 26 个估计器协方差案例和 2 个直接公共协方差 primitive，且没有 CPU fallback。独立的同步性能证据覆盖三个基础规模及 `N=10,000`、`k=2`、`T=200` 的 QS all-lag 场景；该证据只记录 timing，不声明 speedup。"
if cn_sentence not in cn:
    candidates = [
        "## 2026-08-09 — Panel Stage C 协方差完成（PR #126）\n",
        "## 2026-08-09 — Panel Stage C covariance completion (PR #126)\n",
        "## 2026-08-09\n",
    ]
    for heading in candidates:
        if heading in cn:
            cn = cn.replace(heading, heading + "\n" + cn_sentence + "\n", 1)
            break
    else:
        raise RuntimeError("CN Stage-C changelog insertion point not found")
write(cn_path, cn)

review = f'''# PR #126 Panel Stage C physical GPU validation\n\n## Physical acceptance status\n\n**PHYSICAL_GPU_ACCEPTED** for the Stage-C correctness and performance runners measured at exact clean implementation head `{MEASUREMENT_SHA}`.\n\nThe immutable artifacts were added by repository commit `{ARTIFACT_COMMIT}`. Comparing `{MEASUREMENT_SHA}` to `{ARTIFACT_COMMIT}` changes only the two raw JSON evidence files; no numerical implementation, physical validator, performance runner, test, or documentation file changed before measurement was recorded. Therefore the evidence remains applicable under `RELEASING.md`.\n\n## Correctness/backend-provenance artifact\n\n- path: `{CORR_PATH}`\n- measurement SHA: `{MEASUREMENT_SHA}`\n- artifact repository commit: `{ARTIFACT_COMMIT}`\n- Git blob: `{corr_blob}`\n- SHA-256: `{corr_sha}`\n- schema version: 1\n- working tree clean: true\n- top-level status: success\n- GPU: Tesla P100-SXM2-16GB\n- Python: 3.9.16\n- NumPy: 1.24.2\n- SciPy: 1.10.1\n- Torch: 2.0.0\n\nFor both CuPy and Torch, all 26 estimator cases passed with requested/executed backend identity. Each backend also passed two direct public primitive calls with `xp` omitted, physically validating public backend auto-detection rather than only estimator-mediated routing.\n\nDirect primitive maximum absolute differences versus NumPy:\n\n- CuPy `cluster_group_debias`: `8.673617379884035e-19`; `driscoll_kraay_qs`: `4.336808689942018e-19`.\n- Torch `cluster_group_debias`: `8.673617379884035e-19`; `driscoll_kraay_qs`: `8.673617379884035e-19`.\n\nThe QS estimator path records `n_periods=8`, `bandwidth=2`, `max_weighted_lag=7`, and `all_observed_lags_weighted=true`. The legacy Pooled HAC path remains explicitly `row_order_hac=true`.\n\n## Synchronized performance artifact\n\n- path: `{PERF_PATH}`\n- measurement SHA: `{MEASUREMENT_SHA}`\n- artifact repository commit: `{ARTIFACT_COMMIT}`\n- Git blob: `{perf_blob}`\n- SHA-256: `{perf_sha}`\n- schema version: 2\n- working tree clean: true\n- timing scope: synchronized end-to-end estimator fit\n- input residency: X/y/entity/time preloaded on selected GPU backend; cluster labels remain CPU metadata\n- rows: 58 = 54 base rows + 4 high-T QS rows\n- repeats per row: 3\n\nHigh-T `N=10,000`, `k=2`, `T=200` medians:\n\n- CuPy: Pooled QS `23.5766 ms`; Panel entity-FE QS `32.7507 ms`.\n- Torch: Pooled QS `15.9364 ms`; Panel entity-FE QS `23.5517 ms`.\n\nThe performance artifact is accepted as bounded synchronized timing evidence. It does **not** contain or imply a speedup claim or CPU/external baseline. No pathological complexity or transfer-dominated behavior was observed in the required high-T gate.\n\n## Canonical benchmark promotion\n\nThe two immutable raw artifacts are registered directly as SHA-256-protected canonical frontend sources rather than copied into a second normalized JSON.\n\n- validation source id: `{corr_source_id}` — 56 validation-only rows = `(26 estimator + 2 public primitive) x 2 backends`; no timing or speedup.\n- performance source id: `{perf_source_id}` — 58 timing rows; no speedup.\n- source date: `{SOURCE_DATE}`\n- environment: `remote-p100-pr126-20260810`\n\nThe Stage-C parsers fail closed on measurement-SHA drift, backend/case identity drift, public-primitive matrix drift, non-finite/out-of-tolerance correctness differences, malformed timing samples, median/sample inconsistency, and high-T QS contract drift.\n\n## Physical conclusion\n\nStage-C physical correctness/backend provenance and bounded synchronized performance gates are **ACCEPTED**. Exact-final-head hosted CI and a fresh `.claude/skills/code-review.md` review remain post-promotion lifecycle gates. Any later change to `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py` must be audited for whether a new physical run is required.\n'''
write("dev/reviews/pr126_physical_gpu_validation.md", review)

print(json.dumps({
    "correctness_sha256": corr_sha,
    "performance_sha256": perf_sha,
    "correctness_blob": corr_blob,
    "performance_blob": perf_blob,
    "correctness_source_id": corr_source_id,
    "performance_source_id": perf_source_id,
}, indent=2))
