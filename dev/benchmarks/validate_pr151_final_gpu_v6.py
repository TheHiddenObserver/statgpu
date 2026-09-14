#!/usr/bin/env python3
"""PR151 physical CUDA schema v6: analytic-weight inference closure gate.

Schema v6 preserves schema v5 unchanged and runs it first on the same exact
clean source.  It then covers the final review finding that the public
M-estimation inference pipeline must treat ``sample_weight`` as relative
analytic weights, not frequency counts.  Positive global rescaling of one
weight vector must therefore leave both fitted parameters and nonrobust
covariance inference unchanged.

The new gate covers ordinary and penalized logistic GLM consumers on CuPy and
Torch for explicit Newton and L-BFGS, with NumPy as the reference backend.  It
also exercises finite float32 analytic weights whose raw float32 sum overflows,
so inference must use the same stable normalization principle as fitting.
Earlier schema-v3/v4/v5 artifacts remain immutable historical exact-source
evidence and are intentionally not rewritten.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from dev.benchmarks import validate_pr151_final_gpu_v5 as v5
from statgpu.backends import _to_numpy
from statgpu.linear_model import GeneralizedLinearModel
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


SCHEMA_VERSION = 6
SOLVER_TOL = v5.SOLVER_TOL
ATOL_COEF = v5.ATOL_COEF
ATOL_INTERCEPT = v5.ATOL_INTERCEPT
ATOL_WEIGHT_RESCALE = v5.ATOL_WEIGHT_RESCALE
ATOL_INFERENCE = 2.0e-5
_WEIGHT_SCALE = 7.25
_FLOAT32_OVERFLOW_SCALE = 3.0e38
_SOLVERS = v5._SOLVERS


def _snapshot(model):
    return {
        "coef": np.asarray(_to_numpy(model.coef_), dtype=np.float64),
        "intercept": float(model.intercept_),
        "bse": np.asarray(_to_numpy(model._bse), dtype=np.float64),
        "pvalues": np.asarray(_to_numpy(model._pvalues), dtype=np.float64),
        "conf_int": np.asarray(_to_numpy(model._conf_int), dtype=np.float64),
        "selected_solver": str(model._selected_solver),
        "backend": str(model._selected_backend_name),
        "device": str(model._selected_backend_device),
        "inference_method": str(model._inference_result.method),
        "cov_type": str(model._inference_result.cov_type),
    }


def _json_snapshot(snap):
    return {
        "coef": snap["coef"].tolist(),
        "intercept": snap["intercept"],
        "bse": snap["bse"].tolist(),
        "pvalues": snap["pvalues"].tolist(),
        "conf_int": snap["conf_int"].tolist(),
        "selected_solver": snap["selected_solver"],
        "backend": snap["backend"],
        "device": snap["device"],
        "inference_method": snap["inference_method"],
        "cov_type": snap["cov_type"],
    }


def _errors(ref, other):
    return {
        "coef": float(np.max(np.abs(ref["coef"] - other["coef"]))),
        "intercept": abs(ref["intercept"] - other["intercept"]),
        "bse": float(np.max(np.abs(ref["bse"] - other["bse"]))),
        "pvalues": float(np.max(np.abs(ref["pvalues"] - other["pvalues"]))),
        "conf_int": float(np.max(np.abs(ref["conf_int"] - other["conf_int"]))),
    }


def _assert_provenance(name, snap, *, solver, backend, device):
    expected = {
        "selected_solver": solver,
        "backend": backend,
        "device": device,
        "inference_method": "m_estimation",
        "cov_type": "nonrobust",
    }
    for key, value in expected.items():
        if snap.get(key) != value:
            raise AssertionError(
                f"{name}: {key}={snap.get(key)!r}, expected {value!r}"
            )


def _assert_numpy_parity(name, ref, other):
    errors = _errors(ref, other)
    if errors["coef"] > ATOL_COEF or errors["intercept"] > ATOL_INTERCEPT:
        raise AssertionError(f"{name}: parameter parity error {errors}")
    for key in ("bse", "pvalues", "conf_int"):
        if errors[key] > ATOL_INFERENCE:
            raise AssertionError(f"{name}: inference parity error {errors}")
    return errors


def _assert_scale_invariance(name, base, scaled):
    errors = _errors(base, scaled)
    if max(errors.values()) > ATOL_WEIGHT_RESCALE:
        raise AssertionError(f"{name}: global analytic-weight rescaling drift {errors}")
    return errors


def _fit_ordinary(solver, X, y, weights, *, device):
    with v5.v4.v3._solver_warning_gate():
        model = GeneralizedLinearModel(
            family="binomial",
            fit_intercept=True,
            solver=solver,
            device=device,
            max_iter=1000,
            tol=SOLVER_TOL,
            compute_inference=True,
            cov_type="nonrobust",
        ).fit(X, y, sample_weight=weights)
    return _snapshot(model)


def _fit_penalized(solver, X, y, weights, *, device):
    with v5.v4.v3._solver_warning_gate():
        model = PenalizedGeneralizedLinearModel(
            loss="logistic",
            penalty="l2",
            alpha=0.025,
            fit_intercept=True,
            solver=solver,
            device=device,
            max_iter=1000,
            tol=SOLVER_TOL,
            compute_inference=True,
            inference_method="auto",
            cov_type="nonrobust",
        ).fit(X, y, sample_weight=weights)
    return _snapshot(model)


def _container_arrays(backend, X, y, weights, cp, torch, torch_device):
    if backend == "cupy":
        return cp.asarray(X), cp.asarray(y), cp.asarray(weights)

    torch_dtype = torch.float32 if X.dtype == np.float32 else torch.float64
    weight_dtype = (
        torch.float32 if weights.dtype == np.float32 else torch.float64
    )
    return (
        torch.as_tensor(X, dtype=torch_dtype, device=torch_device),
        torch.as_tensor(y, dtype=torch_dtype, device=torch_device),
        torch.as_tensor(weights, dtype=weight_dtype, device=torch_device),
    )


def _consumer_matrix(
    *,
    X_np,
    y_np,
    base_weights,
    scaled_weights,
    cp,
    torch,
    device_id,
    torch_device,
    label,
):
    expected_cuda = f"cuda:{device_id}"
    results = {"ordinary": {}, "penalized": {}}

    for consumer, fit_fn in (
        ("ordinary", _fit_ordinary),
        ("penalized", _fit_penalized),
    ):
        for solver in _SOLVERS:
            ref = fit_fn(solver, X_np, y_np, base_weights, device="cpu")
            ref_scaled = fit_fn(
                solver, X_np, y_np, scaled_weights, device="cpu"
            )
            ref_scale_errors = _assert_scale_invariance(
                f"{label}/{consumer}/{solver}/numpy",
                ref,
                ref_scaled,
            )

            results[consumer][solver] = {
                "numpy": {
                    "base": _json_snapshot(ref),
                    "scaled": _json_snapshot(ref_scaled),
                    "errors_scaled_vs_base": ref_scale_errors,
                }
            }

            for backend, device in (("cupy", "cuda"), ("torch", "torch")):
                Xb, yb, wb = _container_arrays(
                    backend,
                    X_np,
                    y_np,
                    base_weights,
                    cp,
                    torch,
                    torch_device,
                )
                _, _, wb_scaled = _container_arrays(
                    backend,
                    X_np,
                    y_np,
                    scaled_weights,
                    cp,
                    torch,
                    torch_device,
                )
                if backend == "cupy":
                    context = cp.cuda.Device(device_id)
                else:
                    context = torch.cuda.device(torch_device)

                with context:
                    base = fit_fn(solver, Xb, yb, wb, device=device)
                    scaled = fit_fn(
                        solver, Xb, yb, wb_scaled, device=device
                    )

                _assert_provenance(
                    f"{label}/{consumer}/{solver}/{backend}",
                    base,
                    solver=solver,
                    backend=backend,
                    device=expected_cuda,
                )
                parity_errors = _assert_numpy_parity(
                    f"{label}/{consumer}/{solver}/{backend}/base",
                    ref,
                    base,
                )
                scale_errors = _assert_scale_invariance(
                    f"{label}/{consumer}/{solver}/{backend}/scale",
                    base,
                    scaled,
                )
                results[consumer][solver][backend] = {
                    "base": _json_snapshot(base),
                    "scaled": _json_snapshot(scaled),
                    "errors_vs_numpy": parity_errors,
                    "errors_scaled_vs_base": scale_errors,
                }

    return results


def _analytic_weight_inference_gate(cp, torch, device_id, torch_device):
    X_np, y_np = v5._logistic_data(seed=151024, n=144, p=3)
    weights_np = np.linspace(0.4, 1.9, X_np.shape[0], dtype=np.float64)
    return _consumer_matrix(
        X_np=X_np,
        y_np=y_np,
        base_weights=weights_np,
        scaled_weights=_WEIGHT_SCALE * weights_np,
        cp=cp,
        torch=torch,
        device_id=device_id,
        torch_device=torch_device,
        label="analytic_weight_inference",
    )


def _float32_raw_sum_overflow_inference_gate(
    cp, torch, device_id, torch_device
):
    X_np, y_np = v5._logistic_data(seed=151025, n=96, p=3)
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)
    raw = np.linspace(0.75, 1.05, X_np.shape[0], dtype=np.float32)
    overflow_weights = raw * np.float32(_FLOAT32_OVERFLOW_SCALE)
    base_weights = overflow_weights / np.float32(_FLOAT32_OVERFLOW_SCALE)

    if not np.all(np.isfinite(overflow_weights)):
        raise AssertionError("float32 overflow-inference fixture has non-finite entries")
    with np.errstate(over="ignore"):
        raw_sum = np.sum(overflow_weights, dtype=np.float32)
    if np.isfinite(raw_sum):
        raise AssertionError("float32 overflow-inference fixture raw sum did not overflow")

    results = _consumer_matrix(
        X_np=X_np,
        y_np=y_np,
        base_weights=base_weights,
        scaled_weights=overflow_weights,
        cp=cp,
        torch=torch,
        device_id=device_id,
        torch_device=torch_device,
        label="float32_raw_sum_overflow_inference",
    )
    return {
        "raw_float32_sum_overflow": True,
        "overflow_scale": _FLOAT32_OVERFLOW_SCALE,
        "routes": results,
    }


def run(output: Path):
    with tempfile.TemporaryDirectory(prefix="pr151-v6-") as tmpdir:
        v5_path = Path(tmpdir) / "schema_v5.json"
        v5.run(v5_path)
        legacy = json.loads(v5_path.read_text(encoding="utf-8"))

    if legacy.get("status") != "success" or not legacy.get("source_clean"):
        raise AssertionError("embedded schema-v5 acceptance did not succeed cleanly")

    source_sha = str(legacy["source_sha"])
    cp, torch, device_id, torch_device = v5.v4.v3._require_gpu_backends()

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
        "review_closure": {
            "analytic_weight_nonrobust_inference_scale_invariance": (
                _analytic_weight_inference_gate(
                    cp, torch, device_id, torch_device
                )
            ),
            "float32_raw_sum_overflow_inference": (
                _float32_raw_sum_overflow_inference_gate(
                    cp, torch, device_id, torch_device
                )
            ),
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
        "float32_overflow_backend_solver_rows": len(_SOLVERS) * 2 * 2,
        "output": str(output),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/reviews/pr151_final_gpu_v6.json"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
