#!/usr/bin/env python3
"""Physical-GPU gate for CoxPHCV staged-screening safety fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv


SOURCE_FILES = (
    "dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py",
    "dev/tests/test_pr80_cox_cv_staged_safety_contract.py",
    "statgpu/survival/__init__.py",
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_cv_penalty_order_contract.py",
    "statgpu/survival/_cox_cv_staged_safety_contract.py",
    "statgpu/survival/_risk_sets.py",
)
GRID = np.array([0.04, 0.8, 0.12, 0.02, 0.4, 0.06, 0.2, 0.1])
EXPECTED_ORDER = np.sort(GRID)[::-1]


def _git(*args):
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample():
    rng = np.random.default_rng(14101)
    X = rng.normal(size=(96, 3))
    beta = np.array([0.5, -0.3, 0.2])
    baseline = rng.exponential(scale=6.0, size=X.shape[0])
    time = 0.2 + baseline * np.exp(-0.2 * (X @ beta))
    time += np.arange(X.shape[0], dtype=np.float64) * 1e-7
    event = (np.arange(X.shape[0]) % 4 != 0).astype(np.float64)
    return X, time, event


def _backend_arrays(name, X, time, event):
    if name == "cupy":
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy CUDA device unavailable")
        raw_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        device_name = (
            raw_name.decode("utf-8", errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        return (
            "cuda",
            cp.asarray(X),
            cp.asarray(time),
            cp.asarray(event),
            device_name,
            cp.__version__,
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA device unavailable")
    device = torch.device("cuda")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device=device),
        torch.as_tensor(time, dtype=torch.float64, device=device),
        torch.as_tensor(event, dtype=torch.float64, device=device),
        torch.cuda.get_device_name(0),
        torch.__version__,
    )


def _score_map(model):
    return {
        f"{penalty:.17g}": float(score)
        for penalty, score in zip(
            np.asarray(model.penalties_, dtype=np.float64),
            np.asarray(model.cv_results_["mean_pl"], dtype=np.float64),
        )
    }


def _run_backend(name, X_np, time_np, event_np):
    device, X, time, event, device_name, version = _backend_arrays(
        name, X_np, time_np, event_np
    )
    cox_cv._COXPH_CV_CACHE.clear()
    model = CoxPHCV(
        penalties=GRID.copy(),
        cv=3,
        random_state=41,
        ties="efron",
        max_iter=300,
        tol=1e-8,
        device=device,
        compute_inference=False,
    ).fit(X, time, event)
    results = model.cv_results_
    fast = np.asarray(results["fast_pass_candidate_mask"], dtype=bool)
    full = np.asarray(results["full_precision_candidate_mask"], dtype=bool)
    screened = np.asarray(results["screened_out_candidate_mask"], dtype=bool)
    evaluation_order = np.asarray(
        results["penalty_evaluation_order"], dtype=np.float64
    )
    passed = all(
        (
            results["two_stage_requested"] is True,
            results["two_stage_enabled"] is False,
            results["successive_halving_requested"] is True,
            results["successive_halving_enabled"] is False,
            results["staged_execution_mode"]
            == "exhaustive_safety_fallback",
            not np.any(fast),
            np.all(full),
            not np.any(screened),
            np.all(np.asarray(results["candidate_complete"], dtype=bool)),
            np.array_equal(evaluation_order, EXPECTED_ORDER),
            np.array_equal(np.asarray(model.penalties_), GRID),
        )
    )
    return {
        "device": device_name,
        "library_version": version,
        "selected_penalty": float(model.penalty_),
        "scores": _score_map(model),
        "coef": np.asarray(_to_numpy(model.coef_), dtype=np.float64).tolist(),
        "evaluation_order": evaluation_order.tolist(),
        "public_order": np.asarray(model.penalties_).tolist(),
        "two_stage_requested": results["two_stage_requested"],
        "two_stage_enabled": results["two_stage_enabled"],
        "successive_halving_requested": results[
            "successive_halving_requested"
        ],
        "successive_halving_enabled": results[
            "successive_halving_enabled"
        ],
        "staged_execution_mode": results["staged_execution_mode"],
        "fast_pass_candidate_mask": fast.tolist(),
        "full_precision_candidate_mask": full.tolist(),
        "screened_out_candidate_mask": screened.tolist(),
        "passed": bool(passed),
    }


def _tree_dirty_excluding_output(output):
    output_path = output.resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    try:
        output_relative = output_path.relative_to(root).as_posix()
    except ValueError:
        output_relative = None
    retained = []
    for line in _git("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"') if len(line) >= 4 else ""
        if output_relative is not None and path == output_relative:
            continue
        retained.append(line)
    return bool(retained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)

    os.environ["STATGPU_COXPHCV_TWO_STAGE"] = "1"
    os.environ["STATGPU_COXPHCV_SUCCESSIVE_HALVING"] = "1"
    os.environ["STATGPU_COXPHCV_HALVING_TOPK"] = "1"

    head = _git("rev-parse", "HEAD")
    dirty_before = bool(_git("status", "--porcelain"))
    missing_sources = [path for path in SOURCE_FILES if not Path(path).is_file()]
    report = {
        "schema_version": 1,
        "validation_tier": "remote-full",
        "source_commit": head,
        "source_clean": not dirty_before,
        "source_sha256": {
            path: _sha256(path)
            for path in SOURCE_FILES
            if Path(path).is_file()
        },
        "command": (
            "python dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py "
            "--output <path>"
        ),
        "backends": {},
        "cross_backend": {},
        "gate_failures": [],
    }
    if dirty_before:
        report["gate_failures"].append("source tree is dirty before runner")
    if missing_sources:
        report["gate_failures"].append(
            "missing source files: " + ", ".join(missing_sources)
        )

    if not report["gate_failures"]:
        X, time, event = _sample()
        for name in ("cupy", "torch"):
            try:
                result = _run_backend(name, X, time, event)
                report["backends"][name] = result
                if not result["passed"]:
                    report["gate_failures"].append(
                        f"{name}: staged safety contract"
                    )
            except Exception as exc:
                report["backends"][name] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["gate_failures"].append(
                    f"{name}: {type(exc).__name__}"
                )

        if all(
            bool((report["backends"].get(name) or {}).get("passed"))
            for name in ("cupy", "torch")
        ):
            cupy = report["backends"]["cupy"]
            torch = report["backends"]["torch"]
            keys = sorted(cupy["scores"])
            score_error = max(
                abs(cupy["scores"][key] - torch["scores"][key])
                for key in keys
            )
            coef_error = float(
                np.max(
                    np.abs(
                        np.asarray(cupy["coef"], dtype=np.float64)
                        - np.asarray(torch["coef"], dtype=np.float64)
                    )
                )
            )
            selected_equal = (
                cupy["selected_penalty"] == torch["selected_penalty"]
            )
            cross_passed = (
                selected_equal and score_error <= 2e-7 and coef_error <= 2e-6
            )
            report["cross_backend"] = {
                "selected_penalty_equal": selected_equal,
                "score_max_abs_error": float(score_error),
                "coef_max_abs_error": coef_error,
                "passed": bool(cross_passed),
            }
            if not cross_passed:
                report["gate_failures"].append(
                    "CuPy/Torch exhaustive fallback parity failed"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dirty_after = _tree_dirty_excluding_output(output)
    report["source_clean_after"] = not dirty_after
    if dirty_after:
        report["gate_failures"].append("source tree is dirty after runner")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
