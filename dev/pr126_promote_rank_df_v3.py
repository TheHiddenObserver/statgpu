from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_SHA = "f154647665788df2570439a1cc154a43f509aa45"
RAW_EVIDENCE_COMMIT = "2cda842d24f38a1ba95b949215b5779556168a2a"
ENV_ID = "remote-p100-pr126-20260812"
VALIDATION_PATH = Path("results/pr126_p100/panel_stage_c_gpu_validation_f1546476.json")
PERFORMANCE_PATH = Path("results/pr126_p100/panel_stage_c_performance_f1546476.json")
V2_VALIDATION_ID = "panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f"
V2_PERFORMANCE_ID = "panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55"
V2_VALIDATION_COMPARISON = "panel-stage-c-pr126-rank-policy-validation-20260811"
V2_PERFORMANCE_COMPARISON = "panel-stage-c-pr126-rank-policy-performance-20260811"
V3_VALIDATION_COMPARISON = "panel-stage-c-pr126-rank-df-validation-20260812"
V3_PERFORMANCE_COMPARISON = "panel-stage-c-pr126-rank-df-performance-20260812"
RANK_DEF_CASES = (
    "panel_entity_rank_deficient_nonrobust",
    "panel_entity_rank_deficient_robust",
    "between_rank_deficient_nonrobust",
    "between_rank_deficient_robust",
    "first_difference_rank_deficient_nonrobust",
    "first_difference_rank_deficient_robust",
    "random_effects_rank_deficient_nonrobust",
    "random_effects_rank_deficient_robust",
)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one marker, found {count}")
    return text.replace(old, new, 1)


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)], cwd=ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def validate_raw_sources() -> tuple[str, str, str, str]:
    validation_file = ROOT / VALIDATION_PATH
    performance_file = ROOT / PERFORMANCE_PATH
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    performance = json.loads(performance_file.read_text(encoding="utf-8"))

    if validation.get("git_sha") != MEASUREMENT_SHA:
        raise RuntimeError("fresh correctness measurement SHA drifted")
    if validation.get("working_tree_clean") is not True or validation.get("status") != "success":
        raise RuntimeError("fresh correctness source must be exact-clean and successful")
    if validation.get("environment", {}).get("gpu") != "Tesla P100-SXM2-16GB":
        raise RuntimeError("fresh correctness GPU provenance drifted")
    packages = validation.get("environment", {}).get("packages", {})
    if packages.get("cupy") != "13.6.0" or packages.get("torch") != "2.0.0":
        raise RuntimeError("fresh correctness package provenance drifted")
    if validation.get("case_count_per_backend") != 35:
        raise RuntimeError("fresh correctness estimator count must be 35/backend")
    if validation.get("public_primitive_count_per_backend") != 12:
        raise RuntimeError("fresh correctness primitive count must be 12/backend")

    backends = validation.get("backends", {})
    if set(backends) != {"cupy", "torch"}:
        raise RuntimeError("fresh correctness must contain exactly CuPy and Torch")
    for backend in ("cupy", "torch"):
        payload = backends[backend]
        if payload.get("status") != "success" or payload.get("requested_backend") != backend:
            raise RuntimeError(f"{backend}: backend status/provenance failed")
        cases = payload.get("cases", {})
        primitives = payload.get("public_primitives", {})
        if len(cases) != 35 or len(primitives) != 12:
            raise RuntimeError(f"{backend}: physical matrix shape drifted")
        for name, item in cases.items():
            if item.get("status") != "success" or item.get("executed_backend") != backend:
                raise RuntimeError(f"{backend}/{name}: estimator backend provenance failed")
        for name, item in primitives.items():
            if item.get("status") != "success" or item.get("executed_backend") != backend:
                raise RuntimeError(f"{backend}/{name}: primitive backend provenance failed")
        for name in RANK_DEF_CASES:
            item = cases.get(name)
            if item is None:
                raise RuntimeError(f"{backend}: missing rank-deficient case {name}")
            fit_rank = item.get("fit_rank")
            parameter_count = item.get("parameter_count")
            if (
                isinstance(fit_rank, bool)
                or not isinstance(fit_rank, int)
                or isinstance(parameter_count, bool)
                or not isinstance(parameter_count, int)
                or fit_rank <= 0
                or parameter_count <= fit_rank
            ):
                raise RuntimeError(
                    f"{backend}/{name}: expected 0 < fit_rank < parameter_count, "
                    f"got {fit_rank!r}, {parameter_count!r}"
                )

    if performance.get("git_sha") != MEASUREMENT_SHA:
        raise RuntimeError("fresh performance measurement SHA drifted")
    if performance.get("working_tree_clean") is not True:
        raise RuntimeError("fresh performance source must be exact-clean")
    if performance.get("benchmark") != "panel_stage_c_covariance_fit_overhead":
        raise RuntimeError("fresh performance benchmark identity drifted")
    if performance.get("timing_scope") != "synchronized end-to-end estimator fit":
        raise RuntimeError("fresh performance timing scope drifted")
    if performance.get("environment", {}).get("gpu_by_backend") != {
        "cupy": "Tesla P100-SXM2-16GB",
        "torch": "Tesla P100-SXM2-16GB",
    }:
        raise RuntimeError("fresh performance GPU provenance drifted")
    perf_packages = performance.get("environment", {}).get("packages", {})
    if perf_packages.get("cupy") != "13.6.0" or perf_packages.get("torch") != "2.0.0":
        raise RuntimeError("fresh performance package provenance drifted")
    rows = performance.get("rows", [])
    if len(rows) != 58:
        raise RuntimeError(f"fresh performance expected 58 rows, got {len(rows)}")
    if any("speedup" in row for row in rows):
        raise RuntimeError("fresh performance must not encode CPU speedup claims")

    validation_hash = sha256(VALIDATION_PATH)
    performance_hash = sha256(PERFORMANCE_PATH)
    return validation_hash, performance_hash, git_blob(VALIDATION_PATH), git_blob(PERFORMANCE_PATH)


def build_v3_parser() -> None:
    src = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_policy.py"
    dst = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_df.py"
    if dst.exists():
        raise RuntimeError(f"refusing to overwrite existing {dst}")
    text = src.read_text(encoding="utf-8")
    text = text.replace("Canonical v2 parsers", "Canonical v3 parsers")
    text = text.replace("post-rank-policy", "post-rank-df")
    text = text.replace("rank-policy", "rank-df")
    text = text.replace("rank_policy", "rank_df")
    text = text.replace("3dc7df19176f8fb881a8d37e9d75b4f75e71b058", MEASUREMENT_SHA)
    text = text.replace("2026-08-11", "2026-08-12")
    text = replace_once(text, '_PARSER_VERSION = "2.0"', '_PARSER_VERSION = "3.0"', label="parser version")
    text = replace_once(
        text,
        '_EXPECTED_CUPY_VERSION = "13.6.0"',
        '_EXPECTED_CUPY_VERSION = "13.6.0"\n_EXPECTED_TORCH_VERSION = "2.0.0"',
        label="package constants",
    )
    extra_cases = "\n".join(f'    "{name}",' for name in RANK_DEF_CASES)
    text = replace_once(
        text,
        '    "panel_rank_boundary_dk",\n}',
        '    "panel_rank_boundary_dk",\n' + extra_cases + '\n}',
        label="estimator case set",
    )
    rank_set = "_EXPECTED_RANK_DEFICIENT_CASES = {\n" + extra_cases + "\n}\n"
    text = replace_once(
        text,
        "_BASE_CASES = {",
        rank_set + "_BASE_CASES = {",
        label="rank-deficient set",
    )
    text = replace_once(
        text,
        '    if environment.get("packages", {}).get("cupy") != _EXPECTED_CUPY_VERSION:\n        raise ValueError("PR126 rank-df Stage-C CuPy provenance drifted")',
        '    packages = environment.get("packages", {})\n'
        '    if packages.get("cupy") != _EXPECTED_CUPY_VERSION:\n'
        '        raise ValueError("PR126 rank-df Stage-C CuPy provenance drifted")\n'
        '    if packages.get("torch") != _EXPECTED_TORCH_VERSION:\n'
        '        raise ValueError("PR126 rank-df Stage-C Torch provenance drifted")',
        label="correctness package validation",
    )
    contract = '''        rank_deficient_contract_ok: dict[str, bool] = {}\n        for rank_case_name in sorted(_EXPECTED_RANK_DEFICIENT_CASES):\n            rank_case = cases[rank_case_name]\n            fit_rank = rank_case.get("fit_rank")\n            parameter_count = rank_case.get("parameter_count")\n            contract_ok = (\n                isinstance(fit_rank, int)\n                and not isinstance(fit_rank, bool)\n                and isinstance(parameter_count, int)\n                and not isinstance(parameter_count, bool)\n                and 0 < fit_rank < parameter_count\n            )\n            if not contract_ok:\n                raise ValueError(\n                    f"{backend}: {rank_case_name} identified-rank contract drifted"\n                )\n            rank_deficient_contract_ok[rank_case_name] = True\n\n'''
    text = replace_once(
        text,
        "        for case_name in sorted(_EXPECTED_CASES):\n",
        contract + "        for case_name in sorted(_EXPECTED_CASES):\n",
        label="rank-deficient validation block",
    )
    text = replace_once(
        text,
        '            if case_name == "panel_rank_boundary_dk":\n                checks.append(_bool_check("rank_boundary_identified_subspace", boundary_contract_ok))\n',
        '            if case_name == "panel_rank_boundary_dk":\n'
        '                checks.append(_bool_check("rank_boundary_identified_subspace", boundary_contract_ok))\n'
        '            if case_name in _EXPECTED_RANK_DEFICIENT_CASES:\n'
        '                checks.append(\n'
        '                    _bool_check(\n'
        '                        "identified_rank_less_than_parameter_count",\n'
        '                        rank_deficient_contract_ok[case_name],\n'
        '                    )\n'
        '                )\n',
        label="rank-deficient emitted check",
    )
    text = replace_once(
        text,
        '                        "covariance_metadata": case.get("covariance_metadata", {}),\n',
        '                        "covariance_metadata": case.get("covariance_metadata", {}),\n'
        '                        "fit_rank": case.get("fit_rank"),\n'
        '                        "parameter_count": case.get("parameter_count"),\n',
        label="rank-deficient emitted parameters",
    )
    text = replace_once(
        text,
        '    gpu_by_backend = data.get("environment", {}).get("gpu_by_backend", {})\n    if gpu_by_backend != {"cupy": _EXPECTED_GPU, "torch": _EXPECTED_GPU}:\n        raise ValueError("PR126 rank-df Stage-C performance GPU provenance drifted")\n',
        '    performance_environment = data.get("environment", {})\n'
        '    gpu_by_backend = performance_environment.get("gpu_by_backend", {})\n'
        '    if gpu_by_backend != {"cupy": _EXPECTED_GPU, "torch": _EXPECTED_GPU}:\n'
        '        raise ValueError("PR126 rank-df Stage-C performance GPU provenance drifted")\n'
        '    performance_packages = performance_environment.get("packages", {})\n'
        '    if performance_packages.get("cupy") != _EXPECTED_CUPY_VERSION:\n'
        '        raise ValueError("PR126 rank-df Stage-C performance CuPy provenance drifted")\n'
        '    if performance_packages.get("torch") != _EXPECTED_TORCH_VERSION:\n'
        '        raise ValueError("PR126 rank-df Stage-C performance Torch provenance drifted")\n',
        label="performance package validation",
    )
    dst.write_text(text, encoding="utf-8")


def update_parser_exports_and_registry() -> None:
    init_path = ROOT / "dev/benchmarks/frontend_data/parsers/__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    import_marker = '''from .panel_stage_c_rank_policy import (\n    parse_panel_stage_c_rank_policy_physical_validation,\n    parse_panel_stage_c_rank_policy_performance,\n)\n'''
    import_add = import_marker + '''from .panel_stage_c_rank_df import (\n    parse_panel_stage_c_rank_df_physical_validation,\n    parse_panel_stage_c_rank_df_performance,\n)\n'''
    init_text = replace_once(init_text, import_marker, import_add, label="parser __init__ import")
    init_text = replace_once(
        init_text,
        '    "parse_panel_stage_c_rank_policy_performance",\n]',
        '    "parse_panel_stage_c_rank_policy_performance",\n'
        '    "parse_panel_stage_c_rank_df_physical_validation",\n'
        '    "parse_panel_stage_c_rank_df_performance",\n'
        ']'
        ,
        label="parser __all__",
    )
    init_path.write_text(init_text, encoding="utf-8")

    registry_path = ROOT / "dev/benchmarks/frontend_data/registry.py"
    registry = registry_path.read_text(encoding="utf-8")
    registry = replace_once(
        registry,
        '    parse_panel_stage_c_rank_policy_performance,\n)',
        '    parse_panel_stage_c_rank_policy_performance,\n'
        '    parse_panel_stage_c_rank_df_physical_validation,\n'
        '    parse_panel_stage_c_rank_df_performance,\n'
        ')',
        label="registry import",
    )
    registry = replace_once(
        registry,
        '    "panel_stage_c_rank_policy_performance": parse_panel_stage_c_rank_policy_performance,\n}',
        '    "panel_stage_c_rank_policy_performance": parse_panel_stage_c_rank_policy_performance,\n'
        '    "panel_stage_c_rank_df_physical_validation": (\n'
        '        parse_panel_stage_c_rank_df_physical_validation\n'
        '    ),\n'
        '    "panel_stage_c_rank_df_performance": parse_panel_stage_c_rank_df_performance,\n'
        '}',
        label="registry function mapping",
    )
    registry_path.write_text(registry, encoding="utf-8")


def update_tests() -> None:
    path = ROOT / "dev/tests/test_panel_stage_c_frontend_source.py"
    text = path.read_text(encoding="utf-8")
    import_marker = '''from dev.benchmarks.frontend_data.parsers.panel_stage_c_rank_policy import (\n    parse_panel_stage_c_rank_policy_performance,\n    parse_panel_stage_c_rank_policy_physical_validation,\n)\n'''
    import_add = import_marker + '''from dev.benchmarks.frontend_data.parsers.panel_stage_c_rank_df import (\n    parse_panel_stage_c_rank_df_performance,\n    parse_panel_stage_c_rank_df_physical_validation,\n)\n'''
    text = replace_once(text, import_marker, import_add, label="test parser import")
    const_marker = '''RANK_POLICY_PERFORMANCE = (\n    ROOT / "results/pr126_p100/panel_stage_c_performance_3dc7df19.json"\n)\nENV_ID = "remote-p100-pr126-20260811"\n'''
    const_add = const_marker + '''RANK_DF_CORRECTNESS = (\n    ROOT / "results/pr126_p100/panel_stage_c_gpu_validation_f1546476.json"\n)\nRANK_DF_PERFORMANCE = (\n    ROOT / "results/pr126_p100/panel_stage_c_performance_f1546476.json"\n)\nRANK_DF_ENV_ID = "remote-p100-pr126-20260812"\n'''
    text = replace_once(text, const_marker, const_add, label="test source constants")
    tests = r'''


def test_rank_df_validation_parser_emits_47_checks_per_backend():
    runs, models, warnings = parse_panel_stage_c_rank_df_physical_validation(
        RANK_DF_CORRECTNESS, RANK_DF_ENV_ID
    )
    assert warnings == []
    assert len(runs) == 94
    assert {run["backend"] for run in runs} == {"cupy", "torch"}
    assert sum(run["model_id"] == "PanelCovariancePrimitive" for run in runs) == 24
    assert all(run["metrics"]["validation"]["status"] == "pass" for run in runs)
    assert all(
        run["parameters"]["measurement_git_sha"]
        == "f154647665788df2570439a1cc154a43f509aa45"
        for run in runs
    )
    rank_deficient = [
        run
        for run in runs
        if "rank-deficient" in run["variant"]
        and run["model_id"] != "PanelCovariancePrimitive"
    ]
    assert len(rank_deficient) == 16
    assert {run["backend"] for run in rank_deficient} == {"cupy", "torch"}
    assert all(
        0 < run["parameters"]["fit_rank"] < run["parameters"]["parameter_count"]
        for run in rank_deficient
    )
    assert {model["model_id"] for model in models} == {
        "PooledOLS", "PanelOLS", "RandomEffects", "BetweenOLS",
        "FirstDifferenceOLS", "PanelCovariancePrimitive",
    }


def test_rank_df_validation_parser_fails_closed_on_identified_rank_contract(tmp_path):
    data = json.loads(RANK_DF_CORRECTNESS.read_text(encoding="utf-8"))
    case = data["backends"]["cupy"]["cases"]["between_rank_deficient_nonrobust"]
    case["fit_rank"] = case["parameter_count"]
    broken = tmp_path / "broken_rank_df_validation.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="identified-rank contract drifted"):
        parse_panel_stage_c_rank_df_physical_validation(broken, RANK_DF_ENV_ID)


def test_rank_df_performance_parser_emits_58_synchronized_rows():
    runs, models, warnings = parse_panel_stage_c_rank_df_performance(
        RANK_DF_PERFORMANCE, RANK_DF_ENV_ID
    )
    assert warnings == []
    assert len(runs) == 58
    assert all(run["metrics"]["timing"]["fit_time_ms"] > 0 for run in runs)
    assert all("speedup" not in run["metrics"] for run in runs)
    assert all(
        run["parameters"]["measurement_git_sha"]
        == "f154647665788df2570439a1cc154a43f509aa45"
        for run in runs
    )
    assert {model["model_id"] for model in models} == {
        "PooledOLS", "PanelOLS", "RandomEffects"
    }


def test_rank_df_performance_parser_fails_closed_on_median_drift(tmp_path):
    data = json.loads(RANK_DF_PERFORMANCE.read_text(encoding="utf-8"))
    data["rows"][0]["median_seconds"] *= 1.5
    broken = tmp_path / "broken_rank_df_performance.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="reported median does not match raw samples"):
        parse_panel_stage_c_rank_df_performance(broken, RANK_DF_ENV_ID)
'''
    if "test_rank_df_validation_parser_emits_47_checks_per_backend" in text:
        raise RuntimeError("v3 tests already exist")
    path.write_text(text.rstrip() + tests + "\n", encoding="utf-8")


def update_manifest(validation_hash: str, performance_hash: str, validation_blob: str, performance_blob: str) -> None:
    path = ROOT / "dev/benchmarks/frontend_sources.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    environments = manifest["environments"]
    comparisons = manifest["comparisons"]
    sources = manifest["sources"]
    if ENV_ID in environments or V3_VALIDATION_COMPARISON in comparisons or V3_PERFORMANCE_COMPARISON in comparisons:
        raise RuntimeError("fresh v3 environment/comparison identity already exists")

    environments[ENV_ID] = {
        "label": "Tesla P100 PR #126 Panel Stage C rank-df — 2026-08-12",
        "gpu": "Tesla P100-SXM2-16GB",
        "cpu": "x86_64",
    }
    comparisons[V3_VALIDATION_COMPARISON] = copy.deepcopy(comparisons[V2_VALIDATION_COMPARISON])
    comparisons[V3_VALIDATION_COMPARISON]["label"] = (
        "Panel Stage C rank-df physical validation — PR #126 — 2026-08-12"
    )
    comparisons[V3_VALIDATION_COMPARISON]["env_id"] = ENV_ID
    comparisons[V3_PERFORMANCE_COMPARISON] = copy.deepcopy(comparisons[V2_PERFORMANCE_COMPARISON])
    comparisons[V3_PERFORMANCE_COMPARISON]["label"] = (
        "Panel Stage C rank-df synchronized performance — PR #126 — 2026-08-12"
    )
    comparisons[V3_PERFORMANCE_COMPARISON]["env_id"] = ENV_ID

    by_id = {source["source_id"]: source for source in sources}
    old_validation = by_id[V2_VALIDATION_ID]
    old_performance = by_id[V2_PERFORMANCE_ID]
    new_validation = copy.deepcopy(old_validation)
    new_validation.update(
        {
            "source_id": f"panel-stage-c-rank-df-validation-pr126-20260812-{validation_hash[:12]}",
            "comparison_id": V3_VALIDATION_COMPARISON,
            "path": str(VALIDATION_PATH),
            "sha256": validation_hash,
            "parser": "panel_stage_c_rank_df_physical_validation",
            "parser_version": "3.0",
            "env_id": ENV_ID,
            "source_date": "2026-08-12",
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                "PR #126 post-rank-df exact-clean P100 correctness/backend-provenance evidence. "
                f"Raw artifact commit {RAW_EVIDENCE_COMMIT}; Git blob {validation_blob}; "
                f"SHA-256 {validation_hash}. CuPy 13.6.0 and Torch 2.0.0 each pass "
                "35 estimator integrations plus 12 direct public covariance primitives "
                "(47/47 per backend). The eight exact-collinearity nonrobust/HC1 estimator "
                "cases record fit_rank < parameter_count; requested/executed backend identity "
                "is required for every estimator and primitive. Historical Stage-C v1/v2 "
                "sources remain immutable and are not overwritten."
            ),
        }
    )
    new_performance = copy.deepcopy(old_performance)
    new_performance.update(
        {
            "source_id": f"panel-stage-c-rank-df-performance-pr126-20260812-{performance_hash[:12]}",
            "comparison_id": V3_PERFORMANCE_COMPARISON,
            "path": str(PERFORMANCE_PATH),
            "sha256": performance_hash,
            "parser": "panel_stage_c_rank_df_performance",
            "parser_version": "3.0",
            "env_id": ENV_ID,
            "source_date": "2026-08-12",
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                "PR #126 post-rank-df synchronized end-to-end P100 timing evidence. "
                f"Raw artifact commit {RAW_EVIDENCE_COMMIT}; Git blob {performance_blob}; "
                f"SHA-256 {performance_hash}. The immutable source contains 58 synchronized "
                "rows with three positive finite raw samples per row and exact stored medians, "
                "including bounded high-T quadratic-spectral cases on CuPy and Torch. "
                "No CPU speedup claim is encoded. Historical Stage-C v1/v2 sources remain "
                "immutable and are not overwritten."
            ),
        }
    )
    source_ids = {source["source_id"] for source in sources}
    if new_validation["source_id"] in source_ids or new_performance["source_id"] in source_ids:
        raise RuntimeError("fresh v3 immutable source id already exists")
    sources.extend([new_validation, new_performance])
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    validation_hash, performance_hash, validation_blob, performance_blob = validate_raw_sources()
    build_v3_parser()
    update_parser_exports_and_registry()
    update_tests()
    update_manifest(validation_hash, performance_hash, validation_blob, performance_blob)
    print("validation_sha256", validation_hash)
    print("performance_sha256", performance_hash)
    print("validation_blob", validation_blob)
    print("performance_blob", performance_blob)


if __name__ == "__main__":
    main()
