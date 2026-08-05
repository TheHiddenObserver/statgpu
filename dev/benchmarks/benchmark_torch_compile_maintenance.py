"""Physical-GPU benchmark for the maintenance torch.compile policy.

Each compile mode runs in a fresh subprocess so module-level compiled-callable
caches cannot leak across modes. The script writes one machine-readable JSON
artifact and makes no performance-equivalence claim before it is executed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


_PRECISION_RTOL = 1e-6
_PRECISION_ATOL = 1e-8


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _run_child(mode: str, repeats: int) -> dict:
    os.environ["STATGPU_TORCH_COMPILE_MODE"] = mode

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A physical Torch CUDA GPU is required")

    from statgpu.backends import _to_numpy
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.linear_model import ElasticNet, Lasso, PenalizedLinearRegression

    rng = np.random.default_rng(20260805)
    X = rng.normal(size=(1024, 64)).astype(np.float64)
    beta = np.zeros(64, dtype=np.float64)
    beta[:8] = np.array([1.3, -1.0, 0.8, -0.6, 0.5, -0.4, 0.3, -0.2])
    y = X @ beta + 0.05 * rng.normal(size=X.shape[0])

    groups = [list(range(start, start + 8)) for start in range(0, 64, 8)]
    cases = {
        "lasso": lambda: Lasso(
            alpha=0.01, max_iter=100, tol=1e-6, device="torch"
        ),
        "elasticnet": lambda: ElasticNet(
            alpha=0.01, l1_ratio=0.6, max_iter=100, tol=1e-6, device="torch"
        ),
        "scad": lambda: PenalizedLinearRegression(
            penalty="scad",
            alpha=0.03,
            max_iter=60,
            max_lla_iters=3,
            tol=1e-6,
            lla_tol=1e-6,
            device="torch",
            compute_inference=False,
        ),
        "mcp": lambda: PenalizedLinearRegression(
            penalty="mcp",
            alpha=0.03,
            max_iter=60,
            max_lla_iters=3,
            tol=1e-6,
            lla_tol=1e-6,
            device="torch",
            compute_inference=False,
        ),
        "group_scad": lambda: PenalizedLinearRegression(
            penalty="group_scad",
            penalty_kwargs={"groups": groups},
            alpha=0.03,
            max_iter=60,
            max_lla_iters=3,
            tol=1e-6,
            lla_tol=1e-6,
            device="torch",
            compute_inference=False,
        ),
        "group_mcp": lambda: PenalizedLinearRegression(
            penalty="group_mcp",
            penalty_kwargs={"groups": groups},
            alpha=0.03,
            max_iter=60,
            max_lla_iters=3,
            tol=1e-6,
            lla_tol=1e-6,
            device="torch",
            compute_inference=False,
        ),
    }

    case_results = {}
    for name, factory in cases.items():
        get_torch_compile_diagnostics(clear=True)
        timings = []
        prediction = None
        model = None
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model = factory().fit(X, y)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            prediction = np.asarray(_to_numpy(model.predict(X)))
        events = get_torch_compile_diagnostics(clear=True)
        coefficients = np.asarray(_to_numpy(model.coef_))
        finite_prediction = bool(np.isfinite(prediction).all())
        finite_coefficients = bool(np.isfinite(coefficients).all())
        fallback_seen = any("fallback" in event["status"] for event in events)
        if not finite_prediction or not finite_coefficients:
            raise RuntimeError(f"{name}:{mode} produced non-finite output")
        if mode == "default" and fallback_seen:
            raise RuntimeError(f"{name}:{mode} entered fallback")

        case_results[name] = {
            "fit_seconds": timings,
            "finite_prediction": finite_prediction,
            "finite_coefficients": finite_coefficients,
            "prediction": prediction.tolist(),
            "coefficients": coefficients.tolist(),
            "n_iter": _json_value(getattr(model, "n_iter_", None)),
            "converged": _json_value(getattr(model, "converged_", None)),
            "compile_events": events,
            "fallback_seen": fallback_seen,
        }

    return {
        "mode": mode,
        "cases": case_results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }


def _child_main(args) -> None:
    result = _run_child(args.mode, args.repeats)
    Path(args.child_output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parent_main(args) -> None:
    mode_results = {}
    with tempfile.TemporaryDirectory() as directory:
        for mode in ("disable", "default"):
            child_output = Path(directory) / f"{mode}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--mode",
                mode,
                "--repeats",
                str(args.repeats),
                "--child-output",
                str(child_output),
            ]
            subprocess.run(command, check=True)
            mode_results[mode] = json.loads(child_output.read_text(encoding="utf-8"))

    precision = {}
    for case in mode_results["default"]["cases"]:
        eager = mode_results["disable"]["cases"][case]
        compiled = mode_results["default"]["cases"][case]
        eager_prediction = np.asarray(eager["prediction"], dtype=float)
        compiled_prediction = np.asarray(compiled["prediction"], dtype=float)
        eager_coef = np.asarray(eager["coefficients"], dtype=float)
        compiled_coef = np.asarray(compiled["coefficients"], dtype=float)

        np.testing.assert_allclose(
            compiled_prediction,
            eager_prediction,
            rtol=_PRECISION_RTOL,
            atol=_PRECISION_ATOL,
            err_msg=f"{case}: compiled predictions differ from eager reference",
        )
        np.testing.assert_allclose(
            compiled_coef,
            eager_coef,
            rtol=_PRECISION_RTOL,
            atol=_PRECISION_ATOL,
            err_msg=f"{case}: compiled coefficients differ from eager reference",
        )

        precision[case] = {
            "prediction_max_abs_diff": float(
                np.max(np.abs(compiled_prediction - eager_prediction))
            ),
            "coefficient_max_abs_diff": float(
                np.max(np.abs(compiled_coef - eager_coef))
            ),
            "rtol": _PRECISION_RTOL,
            "atol": _PRECISION_ATOL,
            "status": "pass",
        }

    public_mode_results = json.loads(json.dumps(mode_results))
    for result in public_mode_results.values():
        for details in result["cases"].values():
            details.pop("prediction", None)
            details.pop("coefficients", None)

    output = {
        "method": "torch_compile_maintenance",
        "backend_times": {
            "numpy": None,
            "cupy": None,
            "torch": {
                mode: {
                    case: details["fit_seconds"]
                    for case, details in result["cases"].items()
                }
                for mode, result in mode_results.items()
            },
        },
        "external_baseline": {"name": None, "time": None, "version": None},
        "precision_vs_external": {
            "reference": "STATGPU_TORCH_COMPILE_MODE=disable",
            "default_vs_disable": precision,
        },
        "convergence_status": {
            mode: {
                case: {
                    "n_iter": details["n_iter"],
                    "converged": details["converged"],
                }
                for case, details in result["cases"].items()
            }
            for mode, result in mode_results.items()
        },
        "backend_precision": {
            mode: {
                case: {
                    "finite_prediction": details["finite_prediction"],
                    "finite_coefficients": details["finite_coefficients"],
                }
                for case, details in result["cases"].items()
            }
            for mode, result in mode_results.items()
        },
        "compatibility_matrix": public_mode_results,
        "cv_matrix": {},
        "inference_matrix": {
            "status": "not applicable: benchmark uses estimation-only fits"
        },
        "threshold_source": {
            "source": "Issue #45 maintenance workload matrix",
            "repeats": args.repeats,
            "precision_rtol": _PRECISION_RTOL,
            "precision_atol": _PRECISION_ATOL,
        },
        "objective_scaling": "unchanged across compile modes",
        "penalty_scale_mapping": None,
        "cpu_vs_external": None,
        "gpu_vs_cpu": None,
        "crossover_n": None,
        "target_scale_source": "maintenance benchmark n=1024, p=64",
        "optimization_notes": [
            "Compile modes run in fresh subprocesses to isolate callable caches.",
            "Correctness and visible fallback are release gates; timing parity is not assumed.",
        ],
        "validation_tier": "remote-full",
        "schema_status": "ok",
        "timing_scope": {
            "fit": "solver execution including first-use compilation; data generation excluded"
        },
        "reproducibility": {
            "seed": 20260805,
            "modes": ["disable", "default"],
            "environment": mode_results["default"]["environment"],
        },
        "uncovered_reasons": [
            "NumPy/CuPy and external timing baselines are outside this compile-policy benchmark."
        ],
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/torch_compile_maintenance.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=["disable", "default"])
    parser.add_argument("--child-output")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.child:
        if args.mode is None or args.child_output is None:
            parser.error("--child requires --mode and --child-output")
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
