"""Benchmark Exact-ties Cox fits as failure-group count grows.

The benchmark times ``CoxPH.fit``, including backend input conversion,
optimization, and inference. GPU measurements synchronize immediately before
and after ``fit``. Results include convergence and NumPy precision evidence,
plus optional external alignment with R ``survival::coxph(ties="exact")``.

Example
-------
python dev/benchmarks/benchmark_exact_ties_scaling.py \
    --sizes 960 1920 --repeats 3 --largest-repeats 1 --include-r
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import statgpu
import statgpu.survival._cox as cox_module
import statgpu.survival._cox_counting as cox_counting_module
import statgpu.survival._risk_sets as risk_sets_module
from statgpu.survival import CoxPH


def make_data(n_samples: int, n_features: int, seed: int):
    """Create deterministic, bounded-size Exact tie groups."""
    rng = np.random.default_rng(seed + n_samples)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    beta = np.resize(np.array([0.32, -0.24, 0.17, -0.11], dtype=np.float64), n_features)
    uniform = np.clip(rng.random(n_samples), 1e-12, 1.0)
    raw_time = -np.log(uniform) / np.exp(np.clip(X @ beta, -12.0, 12.0))
    event = rng.binomial(1, 0.62, size=n_samples).astype(np.int64)
    event[0] = 1

    n_bins = max(2, n_samples // 8)
    edges = np.quantile(
        raw_time, np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)[1:-1]
    )
    stop = (np.searchsorted(edges, raw_time, side="right") + 1).astype(np.float64)
    return X, stop, event, n_bins


def synchronize(device: str) -> None:
    if device == "cuda":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif device == "torch":
        import torch

        torch.cuda.synchronize()


def fit_once(
    device: str,
    X: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    *,
    start: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
):
    model = CoxPH(
        ties="exact",
        device=device,
        compute_inference=True,
        compute_cindex=False,
        tol=1e-8,
        max_iter=50,
    )
    synchronize(device)
    started = time.perf_counter()
    model.fit(X, stop, event, start=start, strata=strata)
    synchronize(device)
    return {
        "status": "complete",
        "seconds": time.perf_counter() - started,
        "coef": np.asarray(model.coef_, dtype=np.float64).tolist(),
        "log_likelihood": float(model._log_likelihood),
        "covariance": np.asarray(model._var_matrix, dtype=np.float64).tolist(),
        "iterations": int(model.n_iter_),
        "converged": bool(model.converged_),
    }


def r_metadata() -> Dict[str, str]:
    """Return the external R and survival package versions."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("--include-r requires Rscript on PATH")
    command = [
        rscript,
        "--vanilla",
        "-e",
        (
            'cat(paste(R.version$major, R.version$minor, sep="."), "\\n"); '
            'cat(as.character(packageVersion("survival")), "\\n")'
        ),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    )
    lines = [line.strip() for line in result.stdout.splitlines()]
    if len(lines) != 2:
        raise RuntimeError(f"unexpected R metadata output: {result.stdout!r}")
    return {"r_version": lines[0], "survival_version": lines[1]}


def fit_r_once(
    X: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    *,
    start: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Fit R survival::coxph with its exact partial likelihood."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("--include-r requires Rscript on PATH")

    names = [f"x{index + 1}" for index in range(X.shape[1])]
    columns = [np.asarray(X, dtype=np.float64)]
    names.extend(["stop", "event"])
    columns.extend(
        [
            np.asarray(stop, dtype=np.float64)[:, None],
            np.asarray(event, dtype=np.int64)[:, None],
        ]
    )
    if start is not None:
        names.append("start")
        columns.append(np.asarray(start, dtype=np.float64)[:, None])
    if strata is not None:
        names.append("stratum")
        columns.append(np.asarray(strata, dtype=np.int64)[:, None])

    with tempfile.TemporaryDirectory(prefix="statgpu-r-exact-") as temp_dir:
        data_path = Path(temp_dir) / "data.csv"
        np.savetxt(
            data_path,
            np.column_stack(columns),
            delimiter=",",
            header=",".join(names),
            comments="",
            fmt="%.17g",
        )
        response = (
            "Surv(start, stop, event)" if start is not None else "Surv(stop, event)"
        )
        predictors = " + ".join(f"x{index + 1}" for index in range(X.shape[1]))
        if strata is not None:
            predictors += " + strata(stratum)"
        r_code = f"""
suppressPackageStartupMessages(library(survival))
d <- read.csv({json.dumps(str(data_path))}, check.names=FALSE)
control <- coxph.control(iter.max=50, eps=1e-8, timefix=FALSE)
started <- proc.time()[["elapsed"]]
fit <- coxph(
  {response} ~ {predictors},
  data=d,
  ties="exact",
  robust=FALSE,
  control=control,
  model=FALSE,
  x=FALSE,
  y=FALSE
)
elapsed <- proc.time()[["elapsed"]] - started
cat("seconds=", sprintf("%.17g", elapsed), "\\n", sep="")
cat("coef=", paste(sprintf("%.17g", coef(fit)), collapse=","), "\\n", sep="")
cat("log_likelihood=", sprintf("%.17g", fit$loglik[2]), "\\n", sep="")
cat(
  "covariance=",
  paste(sprintf("%.17g", as.vector(fit$var)), collapse=","),
  "\\n",
  sep=""
)
cat("iterations=", fit$iter, "\\n", sep="")
cat("converged=", as.integer(fit$iter < control$iter.max), "\\n", sep="")
"""
        try:
            result = subprocess.run(
                [rscript, "--vanilla", "-e", r_code],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "timeout_seconds": int(timeout),
                "seconds": None,
                "converged": False,
            }
        if result.returncode:
            raise RuntimeError(
                "R survival::coxph failed with exit code "
                f"{result.returncode}; stdout={result.stdout!r}; "
                f"stderr={result.stderr!r}"
            )

    values: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    required = {
        "seconds",
        "coef",
        "log_likelihood",
        "covariance",
        "iterations",
        "converged",
    }
    missing = required.difference(values)
    if missing:
        raise RuntimeError(
            f"missing R coxph output {sorted(missing)}: {result.stdout!r}"
        )
    n_features = X.shape[1]
    covariance = np.fromstring(values["covariance"], sep=",").reshape(
        (n_features, n_features), order="F"
    )
    fitted: Dict[str, Any] = {
        "status": "complete",
        "seconds": float(values["seconds"]),
        "coef": np.fromstring(values["coef"], sep=",").tolist(),
        "log_likelihood": float(values["log_likelihood"]),
        "covariance": covariance.tolist(),
        "iterations": int(values["iterations"]),
        "converged": bool(int(values["converged"])),
    }
    if result.stderr.strip():
        fitted["stderr"] = result.stderr.strip()
    return fitted


SCENARIOS = (
    "right_censored",
    "delayed_entry",
    "strata",
    "delayed_entry_strata",
)


def make_scenario_data(
    scenario: str,
    n_samples: int,
    n_features: int,
    seed: int,
    *,
    strata_count: int = 3,
) -> Dict[str, np.ndarray]:
    """Create one deterministic scenario used by scaling and R alignment."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scaling scenario: {scenario!r}")
    if strata_count <= 0:
        raise ValueError("strata_count must be positive")
    offset = SCENARIOS.index(scenario)
    case_seed = seed + 1009 * (offset + 1)
    X, stop, event, n_bins = make_data(n_samples, n_features, case_seed)
    rng = np.random.default_rng(case_seed)
    case: Dict[str, np.ndarray] = {
        "X": X,
        "stop": stop,
        "event": event,
        "n_bins": np.asarray(n_bins),
    }
    if scenario in {"delayed_entry", "delayed_entry_strata"}:
        case["start"] = stop * rng.uniform(0.0, 0.8, size=n_samples)
    if scenario in {"strata", "delayed_entry_strata"}:
        strata = rng.integers(
            0, strata_count, size=n_samples, dtype=np.int64
        )
        for stratum in range(strata_count):
            indices = np.flatnonzero(strata == stratum)
            if indices.size:
                event[indices[0]] = 1
        case["strata"] = strata
    return case


def make_r_alignment_cases(
    n_samples: int, n_features: int, seed: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create deterministic right-censored, delayed-entry, and strata cases."""
    return {
        scenario: make_scenario_data(
            scenario, n_samples, n_features, seed, strata_count=3
        )
        for scenario in SCENARIOS
    }


def make_scaling_data(
    scenario: str,
    n_samples: int,
    n_features: int,
    seed: int,
    *,
    strata_count: int = 3,
):
    """Create one deterministic scaling case and its optional row metadata."""
    data = make_scenario_data(
        scenario,
        n_samples,
        n_features,
        seed,
        strata_count=strata_count,
    )
    return (
        data["X"],
        data["stop"],
        data["event"],
        data.get("start"),
        data.get("strata"),
        int(data["n_bins"]),
    )


def device_metadata(devices: Iterable[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if "cuda" in devices:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        metadata["cupy_gpu"] = (
            name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        )
        metadata["cupy_version"] = cp.__version__
    if "torch" in devices:
        import torch

        metadata["torch_gpu"] = torch.cuda.get_device_name(0)
        metadata["torch_version"] = torch.__version__
    return metadata


def summarize_runs(runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize repeats using the actual median-ranked fitted result."""
    materialized = list(runs)
    completed = [
        (index, result)
        for index, result in enumerate(materialized)
        if result.get("status", "complete") == "complete"
        and result.get("seconds") is not None
    ]
    if not completed:
        timeout_values = [
            result.get("timeout_seconds")
            for result in materialized
            if result.get("status") == "timeout"
        ]
        return {
            "status": "timeout" if timeout_values else "failed",
            "seconds": [],
            "median_seconds": None,
            "representative_seconds": None,
            "representative_run_index": None,
            "run_converged": [],
            "all_converged": False,
            "all_finite": False,
            "timeout_seconds": timeout_values[0] if timeout_values else None,
        }
    ranked = sorted(completed, key=lambda item: item[1]["seconds"])
    representative_index, representative = ranked[(len(ranked) - 1) // 2]
    run_converged = [bool(result.get("converged", False)) for _, result in completed]
    run_finite = [
        bool(
            np.isfinite(result["seconds"])
            and np.isfinite(result["log_likelihood"])
            and np.all(np.isfinite(np.asarray(result["coef"], dtype=np.float64)))
            and np.all(
                np.isfinite(np.asarray(result["covariance"], dtype=np.float64))
            )
        )
        for _, result in completed
    ]
    return {
        "status": (
            "partial_timeout" if len(completed) != len(materialized) else "complete"
        ),
        "seconds": [result["seconds"] for _, result in completed],
        "median_seconds": statistics.median(
            result["seconds"] for _, result in completed
        ),
        "representative_seconds": representative["seconds"],
        "representative_run_index": representative_index,
        "run_converged": run_converged,
        "all_converged": all(run_converged),
        "all_finite": all(run_finite),
        **{
            key: representative[key]
            for key in representative
            if key not in {"seconds", "status", "converged"}
        },
        "converged": all(run_converged),
    }


def result_differences(
    result: Dict[str, Any], reference: Dict[str, Any]
) -> Dict[str, float]:
    """Compute externally reviewable fit differences."""
    return {
        "coef_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(result["coef"], dtype=np.float64)
                    - np.asarray(reference["coef"], dtype=np.float64)
                )
            )
        ),
        "log_likelihood_abs": float(
            abs(result["log_likelihood"] - reference["log_likelihood"])
        ),
        "covariance_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(result["covariance"], dtype=np.float64)
                    - np.asarray(reference["covariance"], dtype=np.float64)
                )
            )
        ),
    }


def record_alignment_gate(
    failures: list,
    *,
    case_name: str,
    backend_name: str,
    result: Dict[str, Any],
    differences: Dict[str, float],
    thresholds: Dict[str, float],
) -> None:
    """Record convergence or numerical failures without hiding the full report."""
    if not result["converged"]:
        failures.append(f"{case_name}/{backend_name}: did not converge")
    metric_thresholds = {
        "coef_max_abs": thresholds["coef_max_abs"],
        "log_likelihood_abs": thresholds["log_likelihood_abs"],
        "covariance_max_abs": thresholds["covariance_max_abs"],
    }
    for metric, limit in metric_thresholds.items():
        value = differences[metric]
        if not np.isfinite(value) or value > limit:
            failures.append(
                f"{case_name}/{backend_name}: {metric}={value:.17g} > {limit:.17g}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[960, 1920])
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=88031)
    parser.add_argument(
        "--scaling-scenario",
        choices=list(SCENARIOS),
        default="right_censored",
        help="Risk-set scenario used for the requested scaling sizes.",
    )
    parser.add_argument("--strata-count", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--largest-repeats",
        type=int,
        default=1,
        help="Repeat count for the largest size, which can be expensive on NumPy.",
    )
    parser.add_argument(
        "--reduced-repeat-from",
        type=int,
        default=None,
        help="Use --largest-repeats for every size at or above this threshold.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "cuda", "torch"],
        default=["cpu", "cuda", "torch"],
    )
    parser.add_argument(
        "--include-r",
        action="store_true",
        help='Compare with R survival::coxph(ties="exact").',
    )
    parser.add_argument(
        "--r-alignment-size",
        type=int,
        default=160,
        help="Rows per right-censored/delayed-entry/strata R alignment case.",
    )
    parser.add_argument(
        "--skip-r-alignment",
        action="store_true",
        help="Skip the four small external-alignment cases after scaling.",
    )
    parser.add_argument(
        "--r-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each external R fit.",
    )
    parser.add_argument(
        "--r-repeats",
        type=int,
        default=1,
        help="Repeat count for R scaling fits, independent of statgpu repeats.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/exact_ties_scaling.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(size <= 0 for size in args.sizes):
        raise ValueError("sizes must contain only positive integers")
    if args.repeats <= 0 or args.largest_repeats <= 0:
        raise ValueError("repeat counts must be positive")
    if args.r_repeats <= 0:
        raise ValueError("R repeat count must be positive")
    if args.strata_count <= 0:
        raise ValueError("strata count must be positive")
    if args.reduced_repeat_from is not None and args.reduced_repeat_from <= 0:
        raise ValueError("reduced repeat threshold must be positive")
    if args.r_alignment_size <= 0 or args.r_timeout <= 0:
        raise ValueError("R alignment size and timeout must be positive")

    X_warm, stop_warm, event_warm, start_warm, strata_warm, _ = make_scaling_data(
        args.scaling_scenario,
        80,
        args.features,
        args.seed,
        strata_count=args.strata_count,
    )
    for device in args.devices:
        fit_once(
            device,
            X_warm,
            stop_warm,
            event_warm,
            start=start_warm,
            strata=strata_warm,
        )
    r_versions = None
    if args.include_r:
        r_versions = r_metadata()
        fit_r_once(
            X_warm,
            stop_warm,
            event_warm,
            start=start_warm,
            strata=strata_warm,
            timeout=args.r_timeout,
        )

    source_paths = {
        "risk_sets": Path(risk_sets_module.__file__).resolve(),
        "cox_counting": Path(cox_counting_module.__file__).resolve(),
        "cox": Path(cox_module.__file__).resolve(),
    }
    source_path = source_paths["risk_sets"]
    benchmark_path = Path(__file__).resolve()
    thresholds = {
        "coef_max_abs": 1e-6,
        "log_likelihood_abs": 1e-7,
        "covariance_max_abs": 1e-6,
    }
    report: Dict[str, Any] = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statgpu_version": statgpu.__version__,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_paths.items()
        },
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "command_argv": [
            sys.executable,
            str(benchmark_path.relative_to(REPO_ROOT)),
            *sys.argv[1:],
        ],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "features": args.features,
        "seed": args.seed,
        "scaling_scenario": args.scaling_scenario,
        "strata_count_requested": args.strata_count,
        "statgpu_repeats": args.repeats,
        "reduced_repeats": args.largest_repeats,
        "reduced_repeat_from": args.reduced_repeat_from,
        "r_repeats": args.r_repeats,
        "timing_scope": {
            "statgpu": (
                "CoxPH.fit including input conversion and inference; "
                "GPU synchronized immediately before and after fit"
            ),
            "r_survival": (
                "survival::coxph call including inference; R startup, package load, "
                "and CSV parsing excluded"
            ),
        },
        "devices": list(args.devices),
        "device_metadata": device_metadata(args.devices),
        "external_reference": (
            'R survival::coxph(ties="exact", robust=FALSE, timefix=FALSE)'
            if args.include_r
            else None
        ),
        "r_metadata": r_versions,
        "r_alignment_skipped": bool(args.include_r and args.skip_r_alignment),
        "alignment_thresholds": thresholds,
        "gate_failures": [],
        "cases": [],
        "r_alignment_cases": [],
    }
    failures = report["gate_failures"]
    largest = max(args.sizes)
    name_for_device = {"cpu": "numpy", "cuda": "cupy", "torch": "torch"}
    for n_samples in args.sizes:
        reduced = n_samples == largest or (
            args.reduced_repeat_from is not None
            and n_samples >= args.reduced_repeat_from
        )
        repeats = args.largest_repeats if reduced else args.repeats
        X, stop, event, start, strata, n_bins = make_scaling_data(
            args.scaling_scenario,
            n_samples,
            args.features,
            args.seed,
            strata_count=args.strata_count,
        )
        event_mask = event == 1
        failure_strata = (
            np.zeros(int(event_mask.sum()), dtype=np.int64)
            if strata is None
            else strata[event_mask]
        )
        failure_keys = np.column_stack((failure_strata, stop[event_mask]))
        _, tie_counts = np.unique(failure_keys, axis=0, return_counts=True)
        case: Dict[str, Any] = {
            "n": n_samples,
            "repeats": repeats,
            "ties_bins": n_bins,
            "events": int(event.sum()),
            "failure_groups": int(tie_counts.size),
            "max_tie": int(tie_counts.max()),
            "median_tie": float(np.median(tie_counts)),
            "strata_count": 1 if strata is None else int(np.unique(strata).size),
            "backends": {},
        }
        best: Dict[str, Dict[str, Any]] = {}
        for device in args.devices:
            name = name_for_device[device]
            summary = summarize_runs(
                fit_once(device, X, stop, event, start=start, strata=strata)
                for _ in range(repeats)
            )
            case["backends"][name] = summary
            best[name] = summary
            if not summary["all_converged"]:
                failures.append(f"scaling_n={n_samples}/{name}: a repeat did not converge")
            if not summary["all_finite"]:
                failures.append(f"scaling_n={n_samples}/{name}: a repeat was non-finite")
        if args.include_r:
            summary = summarize_runs(
                fit_r_once(
                    X,
                    stop,
                    event,
                    start=start,
                    strata=strata,
                    timeout=args.r_timeout,
                )
                for _ in range(args.r_repeats)
            )
            case["backends"]["r_survival"] = summary
            if summary["status"] in {"complete", "partial_timeout"}:
                best["r_survival"] = summary
            else:
                case["r_scaling_status"] = summary["status"]

        if "numpy" in best:
            numpy_seconds = case["backends"]["numpy"]["median_seconds"]
            for name, result in best.items():
                if name == "numpy":
                    continue
                differences = result_differences(result, best["numpy"])
                backend = case["backends"][name]
                backend.update(
                    {
                        f"{metric}_vs_numpy": value
                        for metric, value in differences.items()
                    }
                )
                backend["speedup_vs_numpy"] = float(
                    numpy_seconds / backend["median_seconds"]
                )
        if "r_survival" in best:
            r_seconds = case["backends"]["r_survival"]["median_seconds"]
            if not best["r_survival"]["converged"]:
                failures.append(f"scaling_n={n_samples}/r_survival: did not converge")
            for name, result in best.items():
                if name == "r_survival":
                    continue
                differences = result_differences(result, best["r_survival"])
                backend = case["backends"][name]
                backend.update(
                    {f"{metric}_vs_r": value for metric, value in differences.items()}
                )
                backend["speedup_vs_r"] = float(r_seconds / backend["median_seconds"])
                record_alignment_gate(
                    failures,
                    case_name=f"scaling_n={n_samples}",
                    backend_name=name,
                    result=result,
                    differences=differences,
                    thresholds=thresholds,
                )
        report["cases"].append(case)

    if args.include_r and not args.skip_r_alignment:
        alignment_cases = make_r_alignment_cases(
            args.r_alignment_size, args.features, args.seed
        )
        for case_name, data in alignment_cases.items():
            start = data.get("start")
            strata = data.get("strata")
            r_result = fit_r_once(
                data["X"],
                data["stop"],
                data["event"],
                start=start,
                strata=strata,
                timeout=args.r_timeout,
            )
            alignment: Dict[str, Any] = {
                "name": case_name,
                "n": args.r_alignment_size,
                "events": int(data["event"].sum()),
                "has_delayed_entry": start is not None,
                "strata_count": (
                    int(np.unique(strata).size) if strata is not None else 1
                ),
                "reference": r_result,
                "backends": {},
            }
            if r_result.get("status") != "complete":
                failures.append(f"{case_name}/r_survival: {r_result['status']}")
                report["r_alignment_cases"].append(alignment)
                continue
            if not r_result["converged"]:
                failures.append(f"{case_name}/r_survival: did not converge")
            for device in args.devices:
                name = name_for_device[device]
                result = fit_once(
                    device,
                    data["X"],
                    data["stop"],
                    data["event"],
                    start=start,
                    strata=strata,
                )
                differences = result_differences(result, r_result)
                result.update(
                    {f"{metric}_vs_r": value for metric, value in differences.items()}
                )
                result["speedup_vs_r"] = float(r_result["seconds"] / result["seconds"])
                alignment["backends"][name] = result
                record_alignment_gate(
                    failures,
                    case_name=case_name,
                    backend_name=name,
                    result=result,
                    differences=differences,
                    thresholds=thresholds,
                )
            report["r_alignment_cases"].append(alignment)

    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
