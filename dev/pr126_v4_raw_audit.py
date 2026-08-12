#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

MEASUREMENT_SHA = "a99726e19c535dfcd0a94711bbc8be6aac437584"
RAW_COMMIT = "ccc46da6c5f2dee025c7715e39215db69b2872b8"
CORRECTNESS = Path("results/pr126_p100/panel_stage_c_gpu_validation_a99726e1.json")
PERFORMANCE = Path("results/pr126_p100/panel_stage_c_performance_a99726e1.json")
GPU = "Tesla P100-SXM2-16GB"

EXPECTED_CASES = {
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
    "panel_rank_boundary_dk",
    "panel_entity_rank_deficient_nonrobust", "panel_entity_rank_deficient_robust",
    "between_rank_deficient_nonrobust", "between_rank_deficient_robust",
    "first_difference_rank_deficient_nonrobust", "first_difference_rank_deficient_robust",
    "random_effects_rank_deficient_nonrobust", "random_effects_rank_deficient_robust",
}
EXPECTED_PRIMITIVES = {
    "cluster_group_debias", "driscoll_kraay_qs",
    "ill_conditioned_hc0", "ill_conditioned_hc2", "ill_conditioned_hc3", "ill_conditioned_dk",
    "rank_boundary_nonrobust", "rank_boundary_hc0", "rank_boundary_hc2",
    "rank_boundary_hc3", "rank_boundary_cluster", "rank_boundary_dk",
}
RANK_DEFICIENT = {
    "panel_rank_boundary_dk",
    "panel_entity_rank_deficient_nonrobust", "panel_entity_rank_deficient_robust",
    "between_rank_deficient_nonrobust", "between_rank_deficient_robust",
    "first_difference_rank_deficient_nonrobust", "first_difference_rank_deficient_robust",
    "random_effects_rank_deficient_nonrobust", "random_effects_rank_deficient_robust",
}
PREDICT_CASES = {"panel_entity_hc0", "random_effects_explicit_constant_hc0"}
BASE_CASES = {
    "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",
    "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",
    "random_effects_nonrobust", "random_effects_hc3",
}
BASE_SCALES = {(10000, 2, 20), (100000, 2, 20), (100000, 10, 20)}
HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def finite_nonnegative_tree(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite_nonnegative_tree(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            finite_nonnegative_tree(item, f"{label}[{idx}]")
        return
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)) and float(value) >= 0.0, f"{label} is not finite/nonnegative: {value!r}")


def package_contract(environment: dict, *, gpu_key: str) -> None:
    if gpu_key == "gpu":
        require(environment.get("gpu") == GPU, f"GPU identity drifted: {environment.get('gpu')!r}")
    else:
        require(environment.get("gpu_by_backend") == {"cupy": GPU, "torch": GPU}, "per-backend GPU identity drifted")
    packages = environment.get("packages", {})
    require(packages.get("cupy") == "13.6.0", f"CuPy provenance drifted: {packages.get('cupy')!r}")
    require(packages.get("torch") == "2.0.0", f"Torch provenance drifted: {packages.get('torch')!r}")


def audit_correctness() -> dict:
    data = json.loads(CORRECTNESS.read_text())
    require(data.get("schema_version") == 2, "correctness schema must be v2")
    require(data.get("git_sha") == MEASUREMENT_SHA, "correctness measurement SHA drifted")
    require(data.get("working_tree_clean") is True, "correctness measurement tree was dirty")
    require(data.get("status") == "success", "correctness top-level status failed")
    require(data.get("case_count_per_backend") == 35, "correctness estimator count drifted")
    require(data.get("public_primitive_count_per_backend") == 12, "correctness primitive count drifted")
    package_contract(data.get("environment", {}), gpu_key="gpu")
    backends = data.get("backends", {})
    require(set(backends) == {"cupy", "torch"}, "correctness must contain exactly CuPy and Torch")
    total = 0
    for backend in ("cupy", "torch"):
        result = backends[backend]
        require(result.get("status") == "success", f"{backend}: backend status failed")
        require(result.get("requested_backend") == backend, f"{backend}: requested backend drifted")
        cases = result.get("cases", {})
        primitives = result.get("public_primitives", {})
        require(set(cases) == EXPECTED_CASES, f"{backend}: estimator case set drifted")
        require(set(primitives) == EXPECTED_PRIMITIVES, f"{backend}: primitive set drifted")
        for name, case in cases.items():
            total += 1
            require(case.get("status") == "success", f"{backend}/{name}: failed")
            require(case.get("executed_backend") == backend, f"{backend}/{name}: executed backend mismatch")
            finite_nonnegative_tree(case.get("max_abs_differences", {}), f"{backend}/{name}/diff")
            rank = case.get("fit_rank")
            params = case.get("parameter_count")
            require(isinstance(rank, int) and not isinstance(rank, bool) and rank > 0, f"{backend}/{name}: invalid fit_rank")
            require(isinstance(params, int) and not isinstance(params, bool) and params > 0, f"{backend}/{name}: invalid parameter_count")
            if name in RANK_DEFICIENT:
                require(rank < params, f"{backend}/{name}: not rank deficient")
                require(case.get("coefficient_inference_applicable") is False, f"{backend}/{name}: coordinate inference unexpectedly applicable")
                reason = case.get("coefficient_inference_reason")
                require(isinstance(reason, str) and "rank deficient" in reason.lower(), f"{backend}/{name}: missing rank-deficiency reason")
            else:
                require(case.get("coefficient_inference_applicable") is True, f"{backend}/{name}: full-rank inference unexpectedly unavailable")
                require(case.get("coefficient_inference_reason") is None, f"{backend}/{name}: unexpected inference reason")
            if name in PREDICT_CASES:
                require(case.get("prediction_backend") == backend, f"{backend}/{name}: prediction backend mismatch")
            else:
                require(case.get("prediction_backend") is None, f"{backend}/{name}: unexpected prediction backend")
        for name, primitive in primitives.items():
            total += 1
            require(primitive.get("status") == "success", f"{backend}/primitive/{name}: failed")
            require(primitive.get("executed_backend") == backend, f"{backend}/primitive/{name}: backend mismatch")
            diff = primitive.get("max_abs_difference")
            require(isinstance(diff, (int, float)) and not isinstance(diff, bool) and math.isfinite(float(diff)) and float(diff) >= 0.0, f"{backend}/primitive/{name}: invalid diff")
    require(total == 94, f"correctness total must be 94, got {total}")
    return data


def row_key(row: dict) -> tuple:
    return (
        row.get("backend"), row.get("case"), row.get("scenario"),
        int(row.get("n_samples", 0)), int(row.get("n_features", 0)), int(row.get("n_times", 0)),
    )


def audit_performance() -> dict:
    data = json.loads(PERFORMANCE.read_text())
    require(data.get("schema_version") == 3, "performance schema must be v3")
    require(data.get("git_sha") == MEASUREMENT_SHA, "performance measurement SHA drifted")
    require(data.get("working_tree_clean") is True, "performance measurement tree was dirty")
    require(data.get("benchmark") == "panel_stage_c_covariance_fit_overhead", "performance benchmark identity drifted")
    require(data.get("timing_scope") == "synchronized end-to-end estimator fit", "performance timing scope drifted")
    require(data.get("repeats") == 3, "performance repeats drifted")
    require(data.get("high_t_scale") == "10000x2x200", "high-T scale drifted")
    require(data.get("two_way_unbalanced_scale") == "10000x2x20", "two-way scale drifted")
    package_contract(data.get("environment", {}), gpu_key="gpu_by_backend")
    rows = data.get("rows", [])
    require(len(rows) == 60, f"performance must contain 60 rows, got {len(rows)}")
    expected = {
        (backend, case, "base", n, k, t)
        for backend in ("cupy", "torch")
        for case in BASE_CASES
        for (n, k, t) in BASE_SCALES
    }
    expected |= {
        (backend, case, "high_t_qs", 10000, 2, 200)
        for backend in ("cupy", "torch") for case in HIGH_T_CASES
    }
    expected |= {
        (backend, "panel_two_way_nonrobust", "two_way_unbalanced", 10000, 2, 20)
        for backend in ("cupy", "torch")
    }
    actual = [row_key(row) for row in rows]
    require(len(actual) == len(set(actual)), "performance contains duplicate matrix rows")
    require(set(actual) == expected, f"performance exact matrix drifted; missing={sorted(expected-set(actual))}, extra={sorted(set(actual)-expected)}")
    for idx, row in enumerate(rows):
        backend = row.get("backend")
        require(backend in {"cupy", "torch"}, f"row {idx}: invalid backend")
        require(row.get("repeats") == 3, f"row {idx}: repeats drifted")
        samples = row.get("samples_seconds")
        require(isinstance(samples, list) and len(samples) == 3, f"row {idx}: expected three raw samples")
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0.0 for v in samples), f"row {idx}: invalid timing samples")
        median = row.get("median_seconds")
        require(isinstance(median, (int, float)) and math.isfinite(float(median)) and float(median) > 0.0, f"row {idx}: invalid median")
        require(float(median) == float(statistics.median(samples)), f"row {idx}: stored median is not exact sample median")
    # Keep the artifact descriptive only: it must not persist a speedup/baseline claim.
    forbidden = {"speedup", "cpu_baseline", "baseline_seconds", "speedup_vs_cpu"}
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                require(key not in forbidden, f"performance artifact contains forbidden claim field {key!r}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(data)
    return data


def identities(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    blob = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    return sha256, blob


def main() -> None:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{MEASUREMENT_SHA}..{RAW_COMMIT}"], text=True
    ).splitlines()
    require(set(changed) == {str(CORRECTNESS), str(PERFORMANCE)}, f"raw commit contains unexpected changes: {changed}")
    audit_correctness()
    audit_performance()
    c_sha, c_blob = identities(CORRECTNESS)
    p_sha, p_blob = identities(PERFORMANCE)
    print("RAW_AUDIT_SUCCESS")
    print(f"correctness_sha256={c_sha}")
    print(f"correctness_blob={c_blob}")
    print(f"performance_sha256={p_sha}")
    print(f"performance_blob={p_blob}")
    print("correctness_rows=94")
    print("performance_rows=60")


if __name__ == "__main__":
    main()
