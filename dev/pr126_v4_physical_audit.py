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
ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS = ROOT / "results/pr126_p100/panel_stage_c_gpu_validation_a99726e1.json"
PERFORMANCE = ROOT / "results/pr126_p100/panel_stage_c_performance_a99726e1.json"
RUNNER = ROOT / "dev/benchmarks/benchmark_panel_stage_c_covariance.py"

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
    "cluster_group_debias", "driscoll_kraay_qs", "ill_conditioned_hc0",
    "ill_conditioned_hc2", "ill_conditioned_hc3", "ill_conditioned_dk",
    "rank_boundary_nonrobust", "rank_boundary_hc0", "rank_boundary_hc2",
    "rank_boundary_hc3", "rank_boundary_cluster", "rank_boundary_dk",
}
RANK_DEF_CASES = {
    "panel_entity_rank_deficient_nonrobust", "between_rank_deficient_nonrobust",
    "first_difference_rank_deficient_nonrobust", "random_effects_rank_deficient_nonrobust",
    "panel_entity_rank_deficient_robust", "between_rank_deficient_robust",
    "first_difference_rank_deficient_robust", "random_effects_rank_deficient_robust",
    "panel_rank_boundary_dk",
}
PREDICTION_CASES = {"panel_entity_hc0", "random_effects_explicit_constant_hc0"}
BASE_CASES = {
    "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",
    "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",
    "random_effects_nonrobust", "random_effects_hc3",
}
BASE_SCALES = {(10000, 2, 20), (100000, 2, 20), (100000, 10, 20)}
HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def check_finite_nonnegative(value, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise AssertionError(f"{label}: expected finite non-negative number, got {value!r}")


def audit_correctness() -> None:
    data = json.loads(CORRECTNESS.read_text())
    assert data["schema_version"] == 2
    assert data["git_sha"] == MEASUREMENT_SHA
    assert data["working_tree_clean"] is True
    assert data["status"] == "success"
    env = data["environment"]
    assert "P100" in env["gpu"]
    assert env["packages"]["cupy"] == "13.6.0"
    assert env["packages"]["torch"] == "2.0.0"
    assert set(data["backends"]) == {"cupy", "torch"}
    for backend in ("cupy", "torch"):
        payload = data["backends"][backend]
        assert payload["status"] == "success"
        assert payload["requested_backend"] == backend
        cases = payload["cases"]
        primitives = payload["public_primitives"]
        assert set(cases) == EXPECTED_CASES, (backend, set(cases) ^ EXPECTED_CASES)
        assert set(primitives) == EXPECTED_PRIMITIVES
        for name, row in cases.items():
            assert row["status"] == "success"
            assert row["executed_backend"] == backend
            for key, value in row["max_abs_differences"].items():
                check_finite_nonnegative(value, f"{backend}.{name}.{key}")
            rank = int(row["fit_rank"])
            count = int(row["parameter_count"])
            applicable = row["coefficient_inference_applicable"]
            reason = row["coefficient_inference_reason"]
            if name in RANK_DEF_CASES:
                assert 0 < rank < count, (backend, name, rank, count)
                assert applicable is False, (backend, name, applicable)
                assert isinstance(reason, str) and "rank deficient" in reason.lower()
            else:
                assert rank == count, (backend, name, rank, count)
                assert applicable is True, (backend, name, applicable)
                assert reason is None
            if name in PREDICTION_CASES:
                assert row["prediction_backend"] == backend, (backend, name, row["prediction_backend"])
        for name, row in primitives.items():
            assert row["status"] == "success"
            assert row["executed_backend"] == backend
            check_finite_nonnegative(row["max_abs_difference"], f"{backend}.primitive.{name}")
        rank_meta = cases["panel_rank_boundary_dk"]["covariance_metadata"]
        assert rank_meta["design_rank"] == 2
        assert rank_meta["design_columns"] == 3
        assert rank_meta["rank_deficient_extension"] is True
        assert rank_meta["coefficient_inference_applicable"] is False
    assert data["case_count_per_backend"] == len(EXPECTED_CASES) == 35
    assert data["public_primitive_count_per_backend"] == len(EXPECTED_PRIMITIVES) == 12


def audit_performance() -> None:
    data = json.loads(PERFORMANCE.read_text())
    assert data["schema_version"] == 3
    assert data["git_sha"] == MEASUREMENT_SHA
    assert data["working_tree_clean"] is True
    assert data["benchmark"] == "panel_stage_c_covariance_fit_overhead"
    assert data["timing_scope"] == "synchronized end-to-end estimator fit"
    assert data["environment"]["packages"]["cupy"] == "13.6.0"
    assert data["environment"]["packages"]["torch"] == "2.0.0"
    gpu_names = data["environment"]["gpu_by_backend"]
    assert set(gpu_names) == {"cupy", "torch"}
    assert all("P100" in gpu_names[b] for b in gpu_names)
    assert data["high_t_scale"] == "10000x2x200"
    assert data["two_way_unbalanced_scale"] == "10000x2x20"
    rows = data["rows"]
    assert len(rows) == 60
    seen = set()
    base = high_t = tw = 0
    for row in rows:
        backend = row["backend"]
        assert backend in {"cupy", "torch"}
        assert int(row["repeats"]) == 3
        samples = [float(v) for v in row["samples_seconds"]]
        assert len(samples) == 3
        assert all(math.isfinite(v) and v > 0 for v in samples)
        median = float(row["median_seconds"])
        assert math.isfinite(median) and median > 0
        assert median == statistics.median(samples), (row, statistics.median(samples))
        key = (backend, row["case"], row["scenario"], int(row["n_samples"]), int(row["n_features"]), int(row["n_times"]))
        assert key not in seen, key
        seen.add(key)
        scenario = row["scenario"]
        dims = (int(row["n_samples"]), int(row["n_features"]), int(row["n_times"]))
        if scenario == "base":
            assert row["case"] in BASE_CASES and dims in BASE_SCALES
            base += 1
        elif scenario == "high_t_qs":
            assert row["case"] in HIGH_T_CASES and dims == (10000, 2, 200)
            high_t += 1
        elif scenario == "two_way_unbalanced":
            assert row["case"] == "panel_two_way_nonrobust" and dims == (10000, 2, 20)
            tw += 1
        else:
            raise AssertionError(f"unexpected scenario: {scenario}")
    assert (base, high_t, tw) == (54, 4, 2), (base, high_t, tw)

    # Schema v3 does not persist a second row-local executed_backend field.
    # Instead the exact immutable runner fails closed before returning elapsed
    # time unless the fitted estimator persisted _backend_name == requested
    # backend. Audit that source contract verbatim rather than synthesizing data.
    text = RUNNER.read_text()
    required = [
        'executed = getattr(model, "_backend_name", None)',
        'if executed is None:',
        'if executed != backend:',
        'return elapsed',
    ]
    for needle in required:
        assert needle in text, needle
    assert text.index('if executed != backend:') < text.index('return elapsed')


def main() -> None:
    audit_correctness()
    audit_performance()
    print("PR126_V4_PHYSICAL_AUDIT=PASS")
    print(f"measurement_sha={MEASUREMENT_SHA}")
    print(f"raw_commit={RAW_COMMIT}")
    print(f"correctness_sha256={sha256(CORRECTNESS)}")
    print(f"correctness_blob={blob(CORRECTNESS)}")
    print(f"performance_sha256={sha256(PERFORMANCE)}")
    print(f"performance_blob={blob(PERFORMANCE)}")
    print("correctness=47/47 per backend")
    print("performance=60/60 rows")
    print("performance_backend_provenance=runner_fail_closed_not_row_local_field")


if __name__ == "__main__":
    main()
