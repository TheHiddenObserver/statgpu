#!/usr/bin/env python3
"""Physical-GPU gate for order-invariant CoxPHCV custom penalty grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv


SOURCE_FILES = (
    "dev/benchmarks/benchmark_cox_cv_penalty_order_gpu.py",
    "dev/tests/test_pr80_cox_cv_penalty_order_contract.py",
    "statgpu/cross_validation/_grid_validation.py",
    "statgpu/survival/__init__.py",
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_cv_penalty_order_contract.py",
    "statgpu/survival/_cox_fit_adapter.py",
    "statgpu/survival/_risk_sets.py",
)
SORTED_GRID = np.array([0.2, 0.1, 0.05], dtype=np.float64)
UNSORTED_GRID = np.array([0.05, 0.2, 0.1], dtype=np.float64)
THIRD_GRID = np.array([0.1, 0.05, 0.2], dtype=np.float64)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample():
    rng = np.random.default_rng(12101)
    X = rng.normal(size=(96, 3))
    beta = np.array([0.55, -0.35, 0.2])
    baseline = rng.exponential(scale=7.0, size=X.shape[0])
    time = 0.2 + baseline * np.exp(-0.25 * (X @ beta))
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


def _fit(device, X, time, event, grid):
    return CoxPHCV(
        penalties=grid,
        cv=3,
        random_state=37,
        ties="efron",
        max_iter=300,
        tol=1e-8,
        device=device,
        compute_inference=False,
    ).fit(X, time, event)


def _score_map(model):
    penalties = np.asarray(model.penalties_, dtype=np.float64)
    scores = np.asarray(model.cv_results_["mean_pl"], dtype=np.float64)
    complete = np.asarray(
        model.cv_results_["candidate_complete"], dtype=bool
    )
    return {
        f"{penalty:.17g}": {
            "mean_pl": float(score),
            "complete": bool(is_complete),
        }
        for penalty, score, is_complete in zip(penalties, scores, complete)
    }


def _max_score_error(left, right):
    errors = []
    for key in left:
        if key not in right:
            return float("inf")
        left_score = left[key]["mean_pl"]
        right_score = right[key]["mean_pl"]
        if np.isnan(left_score) and np.isnan(right_score):
            error = 0.0
        else:
            error = abs(left_score - right_score)
        errors.append(error)
        if left[key]["complete"] != right[key]["complete"]:
            return float("inf")
    return max(errors, default=0.0)


def _run_backend(name):
    X_np, time_np, event_np = _sample()
    device, X, time, event, device_name, version = _backend_arrays(
        name, X_np, time_np, event_np
    )

    cox_cv._COXPH_CV_CACHE.clear()
    sorted_model = _fit(device, X, time, event, SORTED_GRID.copy())
    sorted_cache_hit = bool(
        sorted_model.cv_results_.get("selection_cache_hit", False)
    )

    cox_cv._COXPH_CV_CACHE.clear()
    unsorted_model = _fit(device, X, time, event, UNSORTED_GRID.copy())
    unsorted_cache_hit = bool(
        unsorted_model.cv_results_.get("selection_cache_hit", False)
    )

    cached_model = _fit(device, X, time, event, THIRD_GRID.copy())
    cached_hit = bool(cached_model.cv_results_.get("selection_cache_hit", False))

    sorted_scores = _score_map(sorted_model)
    unsorted_scores = _score_map(unsorted_model)
    cached_scores = _score_map(cached_model)
    score_error = _max_score_error(sorted_scores, unsorted_scores)
    cache_score_error = _max_score_error(unsorted_scores, cached_scores)
    coef_error = float(
        np.max(
            np.abs(
                np.asarray(_to_numpy(sorted_model.coef_), dtype=np.float64)
                - np.asarray(_to_numpy(unsorted_model.coef_), dtype=np.float64)
            )
        )
    )
    cache_coef_error = float(
        np.max(
            np.abs(
                np.asarray(_to_numpy(unsorted_model.coef_), dtype=np.float64)
                - np.asarray(_to_numpy(cached_model.coef_), dtype=np.float64)
            )
        )
    )

    evaluation_orders = [
        np.asarray(
            model.cv_results_["penalty_evaluation_order"], dtype=np.float64
        )
        for model in (sorted_model, unsorted_model, cached_model)
    ]
    public_orders = [
        np.asarray(model.penalties_, dtype=np.float64)
        for model in (sorted_model, unsorted_model, cached_model)
    ]
    selected = [
        float(model.penalty_)
        for model in (sorted_model, unsorted_model, cached_model)
    ]

    passed = all(
        (
            not sorted_cache_hit,
            not unsorted_cache_hit,
            cached_hit,
            all(np.array_equal(order, SORTED_GRID) for order in evaluation_orders),
            np.array_equal(public_orders[0], SORTED_GRID),
            np.array_equal(public_orders[1], UNSORTED_GRID),
            np.array_equal(public_orders[2], THIRD_GRID),
            np.allclose(selected, selected[0], rtol=0.0, atol=0.0),
            score_error <= 2e-8,
            cache_score_error <= 2e-12,
            coef_error <= 2e-7,
            cache_coef_error <= 2e-12,
        )
    )
    return {
        "device": device_name,
        "library_version": version,
        "selected_penalties": selected,
        "sorted_scores": sorted_scores,
        "unsorted_scores": unsorted_scores,
        "cached_scores": cached_scores,
        "score_max_abs_error": score_error,
        "cache_score_max_abs_error": cache_score_error,
        "coef_max_abs_error": coef_error,
        "cache_coef_max_abs_error": cache_coef_error,
        "selection_cache_hit": [
            sorted_cache_hit,
            unsorted_cache_hit,
            cached_hit,
        ],
        "evaluation_orders": [order.tolist() for order in evaluation_orders],
        "public_orders": [order.tolist() for order in public_orders],
        "passed": bool(passed),
    }


def _tree_dirty_excluding_output(output):
    output_path = output.resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    try:
        output_relative = output_path.relative_to(root).as_posix()
    except ValueError:
        output_relative = None
    lines = _git("status", "--porcelain").splitlines()
    retained = []
    for line in lines:
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
            "python dev/benchmarks/benchmark_cox_cv_penalty_order_gpu.py "
            "--output <path>"
        ),
        "backends": {},
        "gate_failures": [],
    }
    if dirty_before:
        report["gate_failures"].append("source tree is dirty before runner")
    if missing_sources:
        report["gate_failures"].append(
            "missing source files: " + ", ".join(missing_sources)
        )

    if not report["gate_failures"]:
        for name in ("cupy", "torch"):
            try:
                result = _run_backend(name)
                report["backends"][name] = result
                if not result["passed"]:
                    report["gate_failures"].append(
                        f"{name}: penalty-order or cache parity"
                    )
            except Exception as exc:
                report["backends"][name] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["gate_failures"].append(
                    f"{name}: {type(exc).__name__}"
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
