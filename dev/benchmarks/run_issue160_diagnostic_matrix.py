#!/usr/bin/env python3
"""Issue #160 full diagnostic matrix built on the traced L-BFGS runner.

The base diagnostic module proves trace fidelity to production and the public
ordinary GLM route. This matrix adds the comparison dimensions required by the
issue without changing production code:

* unweighted;
* the exact historical float32 weighted fixture;
* the same analytic weights multiplied by one positive common factor;
* float32 and float64;
* NumPy, Torch CPU, plus CuPy/Torch CUDA when physically available.

The output is diagnostic evidence. It deliberately records differences rather
than defining the final float32 cross-backend tolerance before the physical
CUDA data exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dev.benchmarks import diagnose_issue160_float32_lbfgs as diag


WEIGHT_SCALE = 7.25


def _unweighted_case(backend_name, X_np, y_np, *, use_cuda):
    dummy = np.ones(X_np.shape[0], dtype=X_np.dtype)
    X, y, _dummy_w, ctx = diag._backend_arrays(
        backend_name, X_np, y_np, dummy, use_cuda=use_cuda
    )
    with ctx:
        X_work = diag._augment_intercept(X, backend_name)
        loss = diag.LogisticLoss()
        traced, traced_iter, trace = diag.traced_lbfgs(
            loss,
            None,
            X_work,
            y,
            sample_weight=None,
        )
        production, production_iter = diag.lbfgs_solver(
            loss,
            None,
            X_work,
            y,
            max_iter=diag.MAX_ITER,
            tol=diag.TOL,
            history_size=diag.HISTORY_SIZE,
            sample_weight=None,
        )
        traced_np = np.asarray(diag._to_numpy(traced), dtype=np.float64)
        production_np = np.asarray(diag._to_numpy(production), dtype=np.float64)
        reproduction_error = float(np.max(np.abs(traced_np - production_np)))
        if production_iter != traced_iter or reproduction_error > 1.0e-12:
            raise AssertionError(
                f"unweighted trace drifted from production on {backend_name}: "
                f"n_iter={traced_iter}/{production_iter}, "
                f"error={reproduction_error:.3e}"
            )
        metrics = diag._final_metrics(loss, X_work, y, production, None)
        estimator_bridge = diag._public_estimator_bridge(
            backend_name,
            X,
            y,
            None,
            production,
            use_cuda=use_cuda,
        )
    return {
        "production_n_iter": int(production_iter),
        "trace_matches_production_max_abs": reproduction_error,
        "ordinary_estimator_bridge": estimator_bridge,
        "final": metrics,
        "trace": trace,
    }


def _scaled_weights(weights):
    if weights.dtype == np.float32:
        return weights * np.float32(WEIGHT_SCALE)
    return weights * np.float64(WEIGHT_SCALE)


def _errors(left, right):
    left_p = np.asarray(left["final"]["params"], dtype=np.float64)
    right_p = np.asarray(right["final"]["params"], dtype=np.float64)
    return {
        "params_max_abs": float(np.max(np.abs(left_p - right_p))),
        "objective_abs": abs(
            float(left["final"]["objective"])
            - float(right["final"]["objective"])
        ),
        "gradient_norm_abs": abs(
            float(left["final"]["gradient_norm"])
            - float(right["final"]["gradient_norm"])
        ),
        "n_iter_abs": abs(
            int(left["production_n_iter"])
            - int(right["production_n_iter"])
        ),
    }


def _seed_matrix(dtype, seed, backend_specs=None):
    if backend_specs is None:
        backend_specs, _ = diag._available_backends()
    X, y, weights = diag._data(seed, dtype)
    scaled = _scaled_weights(weights)

    result = {}
    for backend_label, use_cuda in backend_specs:
        backend_name = "torch" if backend_label == "torch_cuda" else backend_label
        unweighted = _unweighted_case(
            backend_name, X, y, use_cuda=use_cuda
        )
        weighted = diag._one_case(
            backend_name, X, y, weights, use_cuda=use_cuda
        )
        weighted_scaled = diag._one_case(
            backend_name, X, y, scaled, use_cuda=use_cuda
        )
        result[backend_label] = {
            "unweighted": unweighted,
            "weighted": weighted,
            "weighted_scaled": weighted_scaled,
            "analytic_weight_rescale_errors": _errors(
                weighted, weighted_scaled
            ),
        }

    for mode in ("unweighted", "weighted", "weighted_scaled"):
        ref = result["numpy"][mode]
        for backend_result in result.values():
            backend_result[mode]["errors_vs_numpy"] = _errors(
                ref, backend_result[mode]
            )
    return result


def run(output: Path, *, require_cuda: bool = False):
    source = diag._source_identity()
    if source.get("clean") is not True:
        raise RuntimeError(
            "Issue #160 diagnostic matrix requires an exact clean source tree"
        )

    backend_specs, cuda = diag._available_backends()
    if require_cuda and not (cuda.get("cupy") and cuda.get("torch")):
        raise RuntimeError(
            "--require-cuda needs both CuPy CUDA and Torch CUDA on the physical device"
        )

    payload = {
        "schema": 1,
        "status": "running",
        "source": source,
        "environment": diag._environment(cuda),
        "physical_cuda_complete": bool(cuda.get("cupy") and cuda.get("torch")),
        "weight_scale": WEIGHT_SCALE,
        "fixture": {
            "historical_seed": 151025,
            "seeds": list(diag.SEEDS),
            "float32_weight_construction": (
                "base_weights=(linspace(float32)*float32(3e38))/float32(3e38)"
            ),
            "modes": ["unweighted", "weighted", "weighted_scaled"],
        },
        "cases": {},
    }

    for dtype in (np.float32, np.float64):
        dtype_name = np.dtype(dtype).name
        payload["cases"][dtype_name] = {}
        for seed in diag.SEEDS:
            payload["cases"][dtype_name][str(seed)] = _seed_matrix(
                dtype, seed, backend_specs=backend_specs
            )

    # Same-backend float32 versus float64 comparisons for every mode.
    for seed in diag.SEEDS:
        key = str(seed)
        f32_seed = payload["cases"]["float32"][key]
        f64_seed = payload["cases"]["float64"][key]
        for backend_label in f32_seed:
            for mode in ("unweighted", "weighted", "weighted_scaled"):
                f32_seed[backend_label][mode][
                    "errors_vs_same_backend_float64"
                ] = _errors(
                    f64_seed[backend_label][mode],
                    f32_seed[backend_label][mode],
                )

    source_after = diag._source_identity()
    if source_after.get("clean") is not True:
        raise RuntimeError(
            "Issue #160 source became dirty during diagnostic execution"
        )
    if source_after.get("sha") != source.get("sha"):
        raise RuntimeError(
            "Issue #160 source HEAD changed during diagnostic execution: "
            f"{source.get('sha')} -> {source_after.get('sha')}"
        )
    payload["source_after_execution"] = source_after
    payload["status"] = "diagnostic_complete"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "source": source,
        "source_after_execution": source_after,
        "physical_cuda_complete": payload["physical_cuda_complete"],
        "backends": [label for label, _ in backend_specs],
        "modes": payload["fixture"]["modes"],
        "output": str(output),
    }, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/reviews/issue160_float32_lbfgs_matrix.json"),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless both CuPy CUDA and Torch CUDA are available",
    )
    args = parser.parse_args()
    run(args.output, require_cuda=args.require_cuda)


if __name__ == "__main__":
    main()
