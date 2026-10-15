"""Executable Option-A acceptance contract for Issue #160 physical evidence."""

from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "reviews"
    / "issue160_float32_lbfgs_matrix.json"
)
SOURCE_SHA = "0cf1ec085dc6a1cb0942a5497a3d0b0a8e02f8a3"
MODES = ("unweighted", "weighted", "weighted_scaled")

# Reviewed float32 Option-A contract. These are acceptance bounds for the
# maintained native-float32 path; they do not redefine the strict float64
# contract or make NumPy float32 an exact parameter oracle.
F32_OBJECTIVE_ABS = 2.0e-6
F32_PARAMS_ABS = 2.0e-3
F32_GRADIENT_NORM = 2.0e-4
F32_WEIGHT_RESCALE_PARAMS_ABS = 1.0e-3
F32_WEIGHT_RESCALE_GRADIENT_ABS = 2.0e-4

# The physical float64 matrix agrees much more tightly than this. Keep a
# conservative strict check here without coupling Issue #160 to a new public
# float64 solver tolerance.
F64_CROSS_BACKEND_ABS = 1.0e-10


def _load_artifact():
    with ARTIFACT.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_issue160_physical_artifact_provenance_is_immutable_and_complete():
    payload = _load_artifact()
    assert payload["schema"] == 1
    assert payload["status"] == "diagnostic_complete"
    assert payload["physical_cuda_complete"] is True
    assert payload["source"] == {"sha": SOURCE_SHA, "clean": True}
    assert payload["source_after_execution"] == {"sha": SOURCE_SHA, "clean": True}
    assert set(payload["fixture"]["seeds"]) == {151025, 16001, 16002, 16003}
    assert payload["fixture"]["modes"] == list(MODES)


def test_issue160_float32_native_paths_satisfy_option_a_contract():
    payload = _load_artifact()
    cases = payload["cases"]["float32"]

    for seed, seed_cases in cases.items():
        assert set(seed_cases) == {"numpy", "torch", "torch_cuda", "cupy"}, seed
        for backend, backend_cases in seed_cases.items():
            scale = backend_cases["analytic_weight_rescale_errors"]
            assert scale["objective_abs"] <= F32_OBJECTIVE_ABS, (seed, backend, scale)
            assert scale["params_max_abs"] <= F32_WEIGHT_RESCALE_PARAMS_ABS, (
                seed,
                backend,
                scale,
            )
            assert scale["gradient_norm_abs"] <= F32_WEIGHT_RESCALE_GRADIENT_ABS, (
                seed,
                backend,
                scale,
            )

            for mode in MODES:
                case = backend_cases[mode]
                assert case["trace_matches_production_max_abs"] == 0.0, (
                    seed,
                    backend,
                    mode,
                )

                final = case["final"]
                assert final["gradient_norm"] <= F32_GRADIENT_NORM, (
                    seed,
                    backend,
                    mode,
                    final,
                )

                same64 = case["errors_vs_same_backend_float64"]
                assert same64["objective_abs"] <= F32_OBJECTIVE_ABS, (
                    seed,
                    backend,
                    mode,
                    same64,
                )
                assert same64["params_max_abs"] <= F32_PARAMS_ABS, (
                    seed,
                    backend,
                    mode,
                    same64,
                )

                vs_numpy = case["errors_vs_numpy"]
                assert vs_numpy["objective_abs"] <= F32_OBJECTIVE_ABS, (
                    seed,
                    backend,
                    mode,
                    vs_numpy,
                )
                assert vs_numpy["params_max_abs"] <= F32_PARAMS_ABS, (
                    seed,
                    backend,
                    mode,
                    vs_numpy,
                )

                bridge = case["ordinary_estimator_bridge"]
                if bridge["available"]:
                    assert bridge["params_max_abs"] == 0.0, (
                        seed,
                        backend,
                        mode,
                        bridge,
                    )
                    assert bridge["n_iter"] == case["production_n_iter"], (
                        seed,
                        backend,
                        mode,
                        bridge,
                        case["production_n_iter"],
                    )
                    assert bridge["selected_solver"] == "lbfgs"


def test_issue160_float64_matrix_remains_strictly_cross_backend_aligned():
    payload = _load_artifact()
    cases = payload["cases"]["float64"]

    for seed, seed_cases in cases.items():
        for backend, backend_cases in seed_cases.items():
            scale = backend_cases["analytic_weight_rescale_errors"]
            assert scale["objective_abs"] <= F64_CROSS_BACKEND_ABS, (
                seed,
                backend,
                scale,
            )
            assert scale["params_max_abs"] <= F64_CROSS_BACKEND_ABS, (
                seed,
                backend,
                scale,
            )

            for mode in MODES:
                case = backend_cases[mode]
                assert case["trace_matches_production_max_abs"] == 0.0
                vs_numpy = case["errors_vs_numpy"]
                assert vs_numpy["objective_abs"] <= F64_CROSS_BACKEND_ABS, (
                    seed,
                    backend,
                    mode,
                    vs_numpy,
                )
                assert vs_numpy["params_max_abs"] <= F64_CROSS_BACKEND_ABS, (
                    seed,
                    backend,
                    mode,
                    vs_numpy,
                )
