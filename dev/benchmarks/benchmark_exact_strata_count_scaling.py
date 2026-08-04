"""Benchmark a controlled Exact objective as the number of strata grows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import statgpu
import statgpu.survival._risk_sets as risk_sets_module
from dev.benchmarks.benchmark_exact_ties_scaling import device_metadata, synchronize


def make_data(strata_count: int, rows_per_stratum: int, features: int, seed: int):
    """Create one tied failure group per stratum with nested risk sets."""
    rng = np.random.default_rng(seed + 1009 * strata_count)
    n = strata_count * rows_per_stratum
    X = rng.normal(size=(n, features)).astype(np.float64)
    strata = np.repeat(np.arange(strata_count), rows_per_stratum).astype(np.int64)
    stop_pattern = np.repeat(
        np.arange(1, rows_per_stratum // 2 + 1, dtype=np.float64), 2
    )[:rows_per_stratum]
    stop = np.tile(stop_pattern, strata_count)
    event_pattern = np.zeros(rows_per_stratum, dtype=np.int64)
    event_pattern[: min(2, rows_per_stratum)] = 1
    event = np.tile(event_pattern, strata_count)
    start = np.zeros(n, dtype=np.float64)
    beta = np.resize(
        np.array([0.10, -0.20, 0.05, 0.03], dtype=np.float64), features
    )
    return beta, X, stop, event, start, strata


def to_backend(device: str, *values):
    if device == "cuda":
        import cupy as cp

        return [cp.asarray(value) for value in values]
    if device == "torch":
        import torch

        return [
            torch.as_tensor(
                value,
                dtype=(
                    torch.float64
                    if np.asarray(value).dtype.kind == "f"
                    else torch.int64
                ),
                device="cuda",
            )
            for value in values
        ]
    return list(values)


def objective_once(device: str, arrays):
    beta, X, stop, event, start, strata = arrays
    synchronize(device)
    started = time.perf_counter()
    result = risk_sets_module.cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties="exact",
        compute_derivatives=False,
    )
    synchronize(device)
    value = result["log_likelihood"]
    value = float(value.item() if hasattr(value, "item") else value)
    return {"seconds": time.perf_counter() - started, "log_likelihood": value}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strata-counts", type=int, nargs="+", default=[3, 32, 256, 1000]
    )
    parser.add_argument("--rows-per-stratum", type=int, default=8)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "cuda", "torch"],
        default=["cpu", "cuda", "torch"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/benchmark_frontend_sources/"
            "coxph_exact_strata_count_pr80_20260727.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(value <= 0 for value in args.strata_counts):
        raise ValueError("strata counts must be positive")
    if args.rows_per_stratum < 2 or args.features <= 0 or args.repeats <= 0:
        raise ValueError("rows, features, and repeats must be positive")

    script_path = Path(__file__).resolve()
    risk_path = Path(risk_sets_module.__file__).resolve()
    report = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statgpu_version": statgpu.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "command_argv": [
            sys.executable,
            str(script_path.relative_to(REPO_ROOT)),
            *sys.argv[1:],
        ],
        "benchmark_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "risk_sets_sha256": hashlib.sha256(risk_path.read_bytes()).hexdigest(),
        "timing_scope": (
            "cox_counting_process_objective(log-likelihood only), with GPU "
            "synchronization immediately before and after each call"
        ),
        "rows_per_stratum": args.rows_per_stratum,
        "features": args.features,
        "seed": args.seed,
        "repeats": args.repeats,
        "devices": list(args.devices),
        "device_metadata": device_metadata(args.devices),
        "precision_threshold": 1e-9,
        "gate_failures": [],
        "cases": [],
    }

    for strata_count in args.strata_counts:
        raw = make_data(
            strata_count,
            args.rows_per_stratum,
            args.features,
            args.seed,
        )
        case = {
            "strata_count": strata_count,
            "n": strata_count * args.rows_per_stratum,
            "failure_groups": strata_count,
            "backends": {},
        }
        for device in args.devices:
            arrays = to_backend(device, *raw)
            objective_once(device, arrays)
            runs = [objective_once(device, arrays) for _ in range(args.repeats)]
            seconds = [run["seconds"] for run in runs]
            values = [run["log_likelihood"] for run in runs]
            case["backends"][device] = {
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "log_likelihood": values[len(values) // 2],
                "all_finite": bool(np.all(np.isfinite(values))),
            }

        reference = case["backends"].get("cpu")
        if reference is not None:
            for device, result in case["backends"].items():
                difference = abs(
                    result["log_likelihood"] - reference["log_likelihood"]
                )
                result["log_likelihood_abs_vs_cpu"] = difference
                result["speedup_vs_cpu"] = (
                    reference["median_seconds"] / result["median_seconds"]
                )
                if (
                    not result["all_finite"]
                    or difference > report["precision_threshold"]
                ):
                    report["gate_failures"].append(
                        f"strata={strata_count}/{device}: "
                        "non-finite or precision failure"
                    )
        report["cases"].append(case)

    if report["gate_failures"]:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
