#!/usr/bin/env python3
"""PR151 physical CUDA schema v7: corrected analytic-weight inference gate.

Schema v7 preserves the accepted schema-v5 chain unchanged and reuses the
schema-v6 analytic-weight inference definition unchanged. It replaces only the
failed schema-v6 raw-sum-overflow fixture, whose implementation had accidentally
changed both the weight dtype *and* the design/response dtype.

The intended contract is narrower and was already stated by schema v6: finite
float32 analytic weights whose raw float32 sum overflows must still define the
same normalized objective and inference problem. That contract does not add a
new requirement that float32-design CuPy L-BFGS must match NumPy within the
float64-oriented coefficient parity threshold. Therefore the corrected gate
keeps X/y at the maintained float64 reference dtype and varies only the weights
between ordinary-scale float32 and globally scaled float32 overflow values.

The failed schema-v6 run remains historical diagnostic evidence and is not
rewritten. Schema v7 must be run on the exact clean current source before
PR151 can be promoted to merge-ready.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from dev.benchmarks import validate_pr151_final_gpu_v5 as v5
from dev.benchmarks import validate_pr151_final_gpu_v6 as v6


SCHEMA_VERSION = 7
SOLVER_TOL = v6.SOLVER_TOL
ATOL_COEF = v6.ATOL_COEF
ATOL_INTERCEPT = v6.ATOL_INTERCEPT
ATOL_WEIGHT_RESCALE = v6.ATOL_WEIGHT_RESCALE
ATOL_INFERENCE = v6.ATOL_INFERENCE
_WEIGHT_SCALE = v6._WEIGHT_SCALE
_FLOAT32_OVERFLOW_SCALE = v6._FLOAT32_OVERFLOW_SCALE
_SOLVERS = v6._SOLVERS


def _float32_weight_raw_sum_overflow_inference_gate(
    cp, torch, device_id, torch_device
):
    """Check float32-weight overflow without changing the design precision."""
    X_np, y_np = v5._logistic_data(seed=151025, n=96, p=3)
    if X_np.dtype != np.float64 or y_np.dtype != np.float64:
        raise AssertionError(
            "schema-v7 overflow fixture must keep X/y at float64 so the gate "
            "isolates float32 analytic-weight normalization/inference"
        )

    raw = np.linspace(0.75, 1.05, X_np.shape[0], dtype=np.float32)
    overflow_weights = raw * np.float32(_FLOAT32_OVERFLOW_SCALE)
    base_weights = overflow_weights / np.float32(_FLOAT32_OVERFLOW_SCALE)

    if base_weights.dtype != np.float32 or overflow_weights.dtype != np.float32:
        raise AssertionError("schema-v7 overflow fixture weights must remain float32")
    if not np.all(np.isfinite(overflow_weights)):
        raise AssertionError("float32 overflow-inference fixture has non-finite entries")
    with np.errstate(over="ignore"):
        raw_sum = np.sum(overflow_weights, dtype=np.float32)
    if np.isfinite(raw_sum):
        raise AssertionError("float32 overflow-inference fixture raw sum did not overflow")

    expected_cuda = f"cuda:{device_id}"
    with cp.cuda.Device(device_id):
        cupy_overflow = cp.asarray(overflow_weights, dtype=cp.float32)
        cupy_raw_sum_overflow = bool(cp.isinf(cp.sum(cupy_overflow)).item())
    with torch.cuda.device(torch_device):
        torch_overflow = torch.as_tensor(
            overflow_weights,
            dtype=torch.float32,
            device=torch_device,
        )
        torch_raw_sum_overflow = bool(torch.isinf(torch.sum(torch_overflow)).item())

    if not cupy_raw_sum_overflow:
        raise AssertionError(
            "schema-v7 CuPy float32 overflow fixture raw sum did not overflow"
        )
    if not torch_raw_sum_overflow:
        raise AssertionError(
            "schema-v7 Torch float32 overflow fixture raw sum did not overflow"
        )

    results = v6._consumer_matrix(
        X_np=X_np,
        y_np=y_np,
        base_weights=base_weights,
        scaled_weights=overflow_weights,
        cp=cp,
        torch=torch,
        device_id=device_id,
        torch_device=torch_device,
        label="float32_weight_raw_sum_overflow_inference",
    )
    return {
        "design_dtype": str(X_np.dtype),
        "response_dtype": str(y_np.dtype),
        "weight_dtype": str(overflow_weights.dtype),
        "raw_float32_sum_overflow": True,
        "backend_raw_float32_sum_overflow": {
            "cupy": cupy_raw_sum_overflow,
            "torch": torch_raw_sum_overflow,
        },
        "expected_cuda_device": expected_cuda,
        "overflow_scale": _FLOAT32_OVERFLOW_SCALE,
        "routes": results,
    }


def run(output: Path):
    # Re-run the accepted v5 -> v4 -> v3 chain on this exact clean source.
    with tempfile.TemporaryDirectory(prefix="pr151-v7-") as tmpdir:
        v5_path = Path(tmpdir) / "schema_v5.json"
        v5.run(v5_path)
        legacy = json.loads(v5_path.read_text(encoding="utf-8"))

    if legacy.get("status") != "success" or not legacy.get("source_clean"):
        raise AssertionError("embedded schema-v5 acceptance did not succeed cleanly")

    source_sha = str(legacy["source_sha"])
    cp, torch, device_id, torch_device = v5.v4.v3._require_gpu_backends()

    # Reuse the schema-v6 analytic-weight inference matrix unchanged. Do not
    # call v6.run(): the failed v6 overflow fixture intentionally remains
    # immutable historical diagnostic evidence.
    analytic_weight_gate = v6._analytic_weight_inference_gate(
        cp, torch, device_id, torch_device
    )
    overflow_gate = _float32_weight_raw_sum_overflow_inference_gate(
        cp, torch, device_id, torch_device
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": True,
        "environment": legacy["environment"],
        "frozen_tolerances": {
            "coef_max_abs": ATOL_COEF,
            "intercept_abs": ATOL_INTERCEPT,
            "weight_rescale_max_abs": ATOL_WEIGHT_RESCALE,
            "inference_max_abs": ATOL_INFERENCE,
            "solver_tol": SOLVER_TOL,
            "inference_weight_scale": _WEIGHT_SCALE,
            "float32_overflow_scale": _FLOAT32_OVERFLOW_SCALE,
        },
        "legacy_schema_v5": legacy,
        "schema_v6_failure_disposition": {
            "status": "superseded_failed_validator",
            "reason": (
                "schema v6 accidentally cast X/y to float32 in the raw-sum-"
                "overflow inference fixture, adding an unrelated float32-design "
                "CuPy L-BFGS cross-backend parity requirement"
            ),
            "analytic_weight_nonrobust_inference_gate_reused_unchanged": True,
            "failed_float32_design_parity_is_not_pr151_acceptance_scope": True,
        },
        "review_closure": {
            "analytic_weight_nonrobust_inference_scale_invariance": (
                analytic_weight_gate
            ),
            "float32_weight_raw_sum_overflow_inference": overflow_gate,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "ordinary_backend_solver_rows": len(_SOLVERS) * 2,
        "penalized_backend_solver_rows": len(_SOLVERS) * 2,
        "float32_weight_overflow_backend_solver_rows": len(_SOLVERS) * 2 * 2,
        "output": str(output),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/reviews/pr151_final_gpu_v7.json"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
