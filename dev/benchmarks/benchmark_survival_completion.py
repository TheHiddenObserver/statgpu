"""Three-backend correctness and performance gate for Cox survival completion.

The benchmark separates host-to-device transfer from end-to-end ``fit`` time,
synchronizes CUDA around every timed region, and records numerical evidence in
the repository's structured benchmark schema.  It is intentionally a gate,
not a micro-benchmark: coefficients, inference, likelihood, prediction,
convergence, and the external statsmodels baseline are checked together.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.survival import CoxPH, CoxPHCV  # noqa: E402

REQUIRED_SCHEMA_KEYS = {
    "method",
    "backend_times",
    "external_baseline",
    "precision_vs_external",
    "convergence_status",
    "backend_precision",
    "compatibility_matrix",
    "cv_matrix",
    "inference_matrix",
    "threshold_source",
    "objective_scaling",
    "penalty_scale_mapping",
    "cpu_vs_external",
    "gpu_vs_cpu",
    "crossover_n",
    "target_scale_source",
    "optimization_notes",
    "validation_tier",
    "schema_status",
    "gate_failures",
    "timing_scope",
    "reproducibility",
    "uncovered_reasons",
}

GATE_THRESHOLDS = {
    "coef_max_abs": 1e-6,
    "bse_max_abs": 1e-3,
    "pvalue_max_abs": 5e-2,
    "conf_int_max_abs": 5e-3,
    "log_likelihood_abs": 1e-6,
    "prediction_max_abs": 1e-6,
    "cv_best_score_abs": 1e-6,
}


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _sync(backend: str) -> None:
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch

        torch.cuda.synchronize()


def _available_backends() -> tuple[list[str], Dict[str, str]]:
    backends = ["numpy"]
    unavailable: Dict[str, str] = {}
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            backends.append("cupy")
        else:
            unavailable["cupy"] = "no CUDA device"
    except Exception as exc:  # pragma: no cover - host specific
        unavailable["cupy"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        if torch.cuda.is_available():
            backends.append("torch")
        else:
            unavailable["torch"] = "torch.cuda.is_available() is false"
    except Exception as exc:  # pragma: no cover - host specific
        unavailable["torch"] = f"{type(exc).__name__}: {exc}"
    return backends, unavailable


def _gpu_metadata() -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    try:
        import cupy as cp

        output["cupy_device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        output["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            output["torch_device"] = torch.cuda.get_device_name(0)
            output["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return output


def _make_subject_data(
    *,
    n: int,
    p: int,
    seed: int,
    ties_bins: Optional[int],
    n_strata: int = 1,
    delayed_entry: bool = False,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.18, size=p)
    strata = np.arange(n, dtype=np.int64) % n_strata
    rng.shuffle(strata)
    baseline = 0.4 + 0.25 * strata
    event_duration = rng.exponential(
        scale=1.0 / (baseline * np.exp(np.clip(X @ beta, -10.0, 10.0)))
    )
    censor_duration = rng.exponential(scale=2.0, size=n)
    duration = np.minimum(event_duration, censor_duration)
    event = (event_duration <= censor_duration).astype(np.int64)
    if ties_bins is not None:
        width = max(float(np.quantile(duration, 0.95)) / ties_bins, 1e-6)
        duration = np.maximum(np.ceil(duration / width) * width, width)
    if delayed_entry:
        # Discrete entry preserves tied stop times while satisfying start < stop.
        entry = rng.integers(0, 4, size=n).astype(np.float64) * 0.05
    else:
        entry = np.zeros(n, dtype=np.float64)
    stop = entry + duration
    return {
        "X": X.astype(np.float64),
        "start": entry,
        "stop": stop.astype(np.float64),
        "event": event,
        "strata": strata,
        "subject_id": np.arange(n, dtype=np.int64),
    }


def _split_counting_rows(
    data: Dict[str, np.ndarray], seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = data["X"].shape[0]
    # Use an irrational-looking split fraction so a row start is not also a
    # tied failure time.  statsmodels treats entry equality differently from
    # R's ``(start, stop]`` convention, so avoiding equality makes it a valid
    # external coefficient/inference baseline for this benchmark scenario.
    midpoint = data["start"] + 0.371 * (data["stop"] - data["start"])
    X_rows = np.repeat(data["X"], 2, axis=0)
    # The second interval is genuinely time varying while retaining the same
    # subject and stratum.  A small perturbation avoids an artificially easy
    # duplicated-row workload.
    X_rows[1::2] += rng.normal(scale=0.03, size=(n, data["X"].shape[1]))
    return {
        "X": X_rows,
        "start": np.column_stack([data["start"], midpoint]).reshape(-1),
        "stop": np.column_stack([midpoint, data["stop"]]).reshape(-1),
        "event": np.column_stack([np.zeros(n, dtype=np.int64), data["event"]]).reshape(
            -1
        ),
        "strata": np.repeat(data["strata"], 2),
        "subject_id": np.repeat(data["subject_id"], 2),
    }


def _scenario_data(scale: str, seed: int) -> list[Dict[str, Any]]:
    if scale == "quick":
        standard_n, standard_p = 2_000, 12
        entry_n, entry_p = 700, 8
        counting_n, counting_p = 350, 8
        exact_n, exact_p = 70, 3
    else:
        standard_n, standard_p = 20_000, 32
        entry_n, entry_p = 2_500, 16
        counting_n, counting_p = 1_200, 16
        exact_n, exact_p = 120, 4

    standard = _make_subject_data(
        n=standard_n,
        p=standard_p,
        seed=seed,
        ties_bins=80,
    )
    entry = _make_subject_data(
        n=entry_n,
        p=entry_p,
        seed=seed + 1,
        ties_bins=50,
        delayed_entry=True,
    )
    stratified = _make_subject_data(
        n=counting_n,
        p=counting_p,
        seed=seed + 2,
        ties_bins=35,
        n_strata=4,
    )
    counting = _split_counting_rows(stratified, seed + 20)
    exact = _make_subject_data(
        n=exact_n,
        p=exact_p,
        seed=seed + 3,
        ties_bins=14,
    )
    return [
        {
            "name": "standard_heavy_ties",
            "ties": "efron",
            "data": standard,
            "use_start": False,
            "use_strata": False,
            "use_subject_id": False,
            "external": True,
        },
        {
            "name": "delayed_entry",
            "ties": "breslow",
            "data": entry,
            "use_start": True,
            "use_strata": False,
            "use_subject_id": False,
            "external": True,
        },
        {
            "name": "stratified_start_stop",
            "ties": "efron",
            "data": counting,
            "use_start": True,
            "use_strata": True,
            "use_subject_id": True,
            "external": True,
        },
        {
            "name": "exact_ties",
            "ties": "exact",
            "data": exact,
            "use_start": False,
            "use_strata": False,
            "use_subject_id": False,
            "external": False,
        },
    ]


def _convert_array(value: np.ndarray, backend: str):
    if backend == "numpy":
        return np.asarray(value)
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(value)
    import torch

    dtype = torch.float64 if np.issubdtype(value.dtype, np.floating) else torch.int64
    return torch.as_tensor(value, dtype=dtype, device="cuda")


def _convert_data(data: Dict[str, np.ndarray], backend: str) -> Dict[str, Any]:
    return {name: _convert_array(value, backend) for name, value in data.items()}


def _fit_kwargs(scenario: Dict[str, Any], converted: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    if scenario["use_start"]:
        output["start"] = converted["start"]
    if scenario["use_strata"]:
        output["strata"] = converted["strata"]
    if scenario["use_subject_id"]:
        output["subject_id"] = converted["subject_id"]
    return output


def _time_backend(
    scenario: Dict[str, Any],
    backend: str,
    *,
    repeats: int,
    warmups: int,
) -> Dict[str, Any]:
    data = scenario["data"]
    transfer_samples = []
    converted = None
    for _ in range(repeats):
        _sync(backend)
        started = time.perf_counter()
        converted = _convert_data(data, backend)
        _sync(backend)
        transfer_samples.append(time.perf_counter() - started)

    device = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]

    def fit_once():
        model = CoxPH(
            device=device,
            ties=scenario["ties"],
            compute_inference=True,
            compute_cindex=False,
            max_iter=80,
            tol=1e-9,
        )
        return model.fit(
            converted["X"],
            converted["stop"],
            converted["event"],
            **_fit_kwargs(scenario, converted),
        )

    for _ in range(warmups):
        fit_once()
        _sync(backend)

    fit_samples = []
    model = None
    for _ in range(repeats):
        _sync(backend)
        started = time.perf_counter()
        model = fit_once()
        _sync(backend)
        fit_samples.append(time.perf_counter() - started)

    prediction_X = data["X"][: min(8, data["X"].shape[0])]
    prediction_times = np.quantile(data["stop"], [0.2, 0.5, 0.8])
    prediction_strata = (
        data["strata"][: prediction_X.shape[0]] if scenario["use_strata"] else None
    )
    pred_started = time.perf_counter()
    survival, _ = model.predict_survival(
        prediction_X, times=prediction_times, strata=prediction_strata
    )
    prediction_seconds = time.perf_counter() - pred_started

    baseline = model._baseline_by_stratum
    if baseline is None:
        baseline_last = None
    else:
        baseline_last = {
            str(key): float(value["cumulative_hazard"][-1])
            for key, value in baseline.items()
            if value["cumulative_hazard"].size
        }
    return {
        "transfer_seconds": float(np.median(transfer_samples)),
        "fit_seconds": float(np.median(fit_samples)),
        "fit_samples_seconds": [float(value) for value in fit_samples],
        "prediction_seconds": float(prediction_seconds),
        "coef": model.coef_.tolist(),
        "bse": model._bse.tolist(),
        "zvalues": model._zvalues.tolist(),
        "pvalues": model._pvalues.tolist(),
        "conf_int": model._conf_int.tolist(),
        "log_likelihood": float(model._log_likelihood),
        "prediction": survival.tolist(),
        "baseline_last": baseline_last,
        "converged": bool(model._converged),
        "iterations": int(model._iterations),
        "stop_reason": model._stop_reason,
    }


def _statsmodels_reference(scenario: Dict[str, Any]) -> Dict[str, Any]:
    import statsmodels.duration.api as smd

    data = scenario["data"]
    kwargs: Dict[str, Any] = {"status": data["event"], "ties": scenario["ties"]}
    if scenario["use_start"]:
        kwargs["entry"] = data["start"]
    if scenario["use_strata"]:
        kwargs["strata"] = data["strata"]
    started = time.perf_counter()
    result = smd.PHReg(data["stop"], data["X"], **kwargs).fit(disp=0)
    elapsed = time.perf_counter() - started
    params = np.asarray(result.params)
    bse = np.asarray(result.bse)
    return {
        "time_seconds": float(elapsed),
        "coef": params.tolist(),
        "bse": bse.tolist(),
        "zvalues": (params / bse).tolist(),
        "pvalues": np.asarray(result.pvalues).tolist(),
        "conf_int": np.asarray(result.conf_int()).tolist(),
        "log_likelihood": float(result.model.loglike(result.params)),
    }


def _cv_evidence(
    scenario: Dict[str, Any], backends: Iterable[str], seed: int
) -> Dict[str, Any]:
    """Validate grouped counting-process CV selection and final refit."""
    data = scenario["data"]
    penalties = np.asarray([0.0, 0.01, 0.1], dtype=np.float64)
    runs: Dict[str, Any] = {}
    for backend in backends:
        converted = _convert_data(data, backend)
        device = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
        model = CoxPHCV(
            penalties=penalties,
            cv=3,
            ties=scenario["ties"],
            device=device,
            compute_inference=True,
            max_iter=60,
            tol=1e-8,
            random_state=seed,
        )
        try:
            _sync(backend)
            started = time.perf_counter()
            model.fit(
                converted["X"],
                converted["stop"],
                converted["event"],
                start=converted["start"],
                strata=converted["strata"],
                subject_id=converted["subject_id"],
            )
            _sync(backend)
            runs[backend] = {
                "status": "pass",
                "fit_seconds": float(time.perf_counter() - started),
                "selected_penalty": float(model.penalty_),
                "best_score": float(model.best_score_),
                "coef": model.coef_.tolist(),
                "bse": model.estimator_._bse.tolist(),
                "effective_device": model.effective_device_,
                "scoring_device": model.cv_results_["scoring_device"],
                "orchestration_device": model.cv_results_["orchestration_device"],
                "mean_fold_scores": np.asarray(
                    model.cv_results_["mean_pl"], dtype=np.float64
                ).tolist(),
                "effective_folds": np.asarray(
                    model.cv_results_["effective_fold_counts"], dtype=np.int64
                ).tolist(),
                "candidate_complete": np.asarray(
                    model.cv_results_["candidate_complete"], dtype=bool
                ).tolist(),
                "final_converged": bool(model.estimator_._converged),
            }
        except Exception as exc:
            runs[backend] = {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }

    comparisons: Dict[str, Any] = {}
    reference = runs.get("numpy", {})
    for backend in ("cupy", "torch"):
        run = runs.get(backend, {})
        if run.get("status") == "pass" and reference.get("status") == "pass":
            comparisons[backend] = {
                "selected_penalty_equal": (
                    run["selected_penalty"] == reference["selected_penalty"]
                ),
                "best_score_abs": abs(run["best_score"] - reference["best_score"]),
                "refit_coef_max_abs": _max_abs(run["coef"], reference["coef"]),
                "refit_bse_max_abs": _max_abs(run["bse"], reference["bse"]),
            }
    return {
        "scenario": scenario["name"],
        "penalties": penalties.tolist(),
        "folds": 3,
        "subject_grouped": True,
        "runs": runs,
        "backend_comparisons": comparisons,
    }


def _max_abs(a: Iterable[float], b: Iterable[float]) -> float:
    return float(
        np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )


def _precision(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    output = {
        "coef_max_abs": _max_abs(left["coef"], right["coef"]),
        "bse_max_abs": _max_abs(left["bse"], right["bse"]),
        "pvalue_max_abs": _max_abs(left["pvalues"], right["pvalues"]),
        "conf_int_max_abs": _max_abs(left["conf_int"], right["conf_int"]),
        "log_likelihood_abs": abs(left["log_likelihood"] - right["log_likelihood"]),
    }
    if "prediction" in left and "prediction" in right:
        output["prediction_max_abs"] = _max_abs(left["prediction"], right["prediction"])
    return output


def _check_precision(
    label: str,
    values: Dict[str, Any],
    failures: list[str],
    *,
    include_prediction: bool,
) -> None:
    metrics = [
        "coef_max_abs",
        "bse_max_abs",
        "pvalue_max_abs",
        "conf_int_max_abs",
        "log_likelihood_abs",
    ]
    if include_prediction:
        metrics.append("prediction_max_abs")
    for metric in metrics:
        value = values.get(metric)
        threshold = GATE_THRESHOLDS[metric]
        if value is None:
            failures.append(f"{label}: missing {metric}")
        elif not np.isfinite(value):
            failures.append(f"{label}: {metric} is non-finite")
        elif value > threshold:
            failures.append(f"{label}: {metric}={value:.6g} exceeds {threshold:.6g}")


def _collect_gate_failures(
    output: Dict[str, Any],
    scenarios: Iterable[Dict[str, Any]],
    backends: Iterable[str],
) -> list[str]:
    """Turn the benchmark evidence into a strict, machine-checkable gate."""
    failures: list[str] = []
    scenario_list = list(scenarios)
    backend_list = list(backends)

    for scenario in scenario_list:
        name = scenario["name"]
        for backend in backend_list:
            if output["compatibility_matrix"].get(name, {}).get(backend) != "pass":
                failures.append(f"{name}/{backend}: backend compatibility failed")
            if output["inference_matrix"].get(name, {}).get(backend) != "pass":
                failures.append(f"{name}/{backend}: inference validation failed")
            convergence = output["convergence_status"].get(name, {}).get(backend)
            if not convergence or not convergence.get("converged", False):
                failures.append(f"{name}/{backend}: optimizer did not converge")

        if scenario["external"]:
            precision = output["precision_vs_external"].get(name)
            if precision is None:
                failures.append(f"{name}/statsmodels: comparison is missing")
            else:
                _check_precision(
                    f"{name}/statsmodels",
                    precision,
                    failures,
                    include_prediction=False,
                )

        for backend in ("cupy", "torch"):
            if backend not in backend_list:
                continue
            precision = output["backend_precision"].get(name, {}).get(backend)
            if precision is None:
                failures.append(f"{name}/{backend}-vs-numpy: comparison is missing")
            else:
                _check_precision(
                    f"{name}/{backend}-vs-numpy",
                    precision,
                    failures,
                    include_prediction=True,
                )

    cv = output["cv_matrix"]
    runs = cv.get("runs", {})
    for backend in backend_list:
        run = runs.get(backend)
        if not run or run.get("status") != "pass":
            failures.append(f"CV/{backend}: run failed or is missing")
            continue
        if not run.get("final_converged", False):
            failures.append(f"CV/{backend}: final refit did not converge")
        if not all(run.get("candidate_complete", [])):
            failures.append(f"CV/{backend}: at least one candidate is incomplete")

    comparisons = cv.get("backend_comparisons", {})
    for backend in ("cupy", "torch"):
        if backend not in backend_list:
            continue
        comparison = comparisons.get(backend)
        if comparison is None:
            failures.append(f"CV/{backend}-vs-numpy: comparison is missing")
            continue
        if not comparison.get("selected_penalty_equal", False):
            failures.append(f"CV/{backend}: selected penalty differs from NumPy")
        for metric, threshold in (
            ("refit_coef_max_abs", GATE_THRESHOLDS["coef_max_abs"]),
            ("refit_bse_max_abs", GATE_THRESHOLDS["bse_max_abs"]),
            ("best_score_abs", GATE_THRESHOLDS["cv_best_score_abs"]),
        ):
            value = comparison.get(metric)
            if value is None or not np.isfinite(value):
                failures.append(f"CV/{backend}: {metric} is missing or non-finite")
            elif value > threshold:
                failures.append(
                    f"CV/{backend}: {metric}={value:.6g} exceeds {threshold:.6g}"
                )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "results"
        / f"survival_completion_{date.today():%Y-%m-%d}.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be >= 1 and warmups must be >= 0")
    backends, unavailable = _available_backends()
    scenarios = _scenario_data(args.scale, args.seed)

    details: Dict[str, Any] = {}
    backend_times: Dict[str, Dict[str, float]] = {name: {} for name in backends}
    convergence: Dict[str, Dict[str, Any]] = {}
    backend_precision: Dict[str, Dict[str, Any]] = {}
    external_precision: Dict[str, Dict[str, Any]] = {}
    compatibility: Dict[str, Dict[str, str]] = {}
    inference: Dict[str, Dict[str, str]] = {}
    cpu_vs_external: Dict[str, Optional[float]] = {}
    gpu_vs_cpu: Dict[str, Dict[str, float]] = {}

    for scenario in scenarios:
        name = scenario["name"]
        details[name] = {
            "config": {
                "n_rows": int(scenario["data"]["X"].shape[0]),
                "p": int(scenario["data"]["X"].shape[1]),
                "events": int(scenario["data"]["event"].sum()),
                "ties": scenario["ties"],
                "start_stop": bool(scenario["use_start"]),
                "strata": (
                    int(np.unique(scenario["data"]["strata"]).size)
                    if scenario["use_strata"]
                    else 1
                ),
            },
            "backends": {},
        }
        compatibility[name] = {}
        inference[name] = {}
        for backend in backends:
            try:
                run = _time_backend(
                    scenario,
                    backend,
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
                details[name]["backends"][backend] = run
                backend_times[backend][name] = run["fit_seconds"]
                convergence.setdefault(name, {})[backend] = {
                    "converged": run["converged"],
                    "iterations": run["iterations"],
                    "stop_reason": run["stop_reason"],
                }
                compatibility[name][backend] = "pass"
                inference[name][backend] = (
                    "pass"
                    if all(
                        np.all(np.isfinite(run[field]))
                        for field in ("bse", "zvalues", "pvalues", "conf_int")
                    )
                    else "fail-nonfinite"
                )
            except Exception as exc:
                details[name]["backends"][backend] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
                compatibility[name][backend] = "fail"
                inference[name][backend] = "fail"

        numpy_run = details[name]["backends"].get("numpy", {})
        backend_precision[name] = {}
        gpu_vs_cpu[name] = {}
        for backend in ("cupy", "torch"):
            other = details[name]["backends"].get(backend, {})
            if "coef" in numpy_run and "coef" in other:
                backend_precision[name][backend] = _precision(other, numpy_run)
                gpu_vs_cpu[name][backend] = (
                    numpy_run["fit_seconds"] / other["fit_seconds"]
                )

        if scenario["external"]:
            try:
                reference = _statsmodels_reference(scenario)
                details[name]["statsmodels"] = reference
                external_precision[name] = _precision(numpy_run, reference)
                cpu_vs_external[name] = (
                    reference["time_seconds"] / numpy_run["fit_seconds"]
                )
            except Exception as exc:
                details[name]["statsmodels"] = {"error": f"{type(exc).__name__}: {exc}"}
                cpu_vs_external[name] = None
        else:
            cpu_vs_external[name] = None

    missing_backend_notes = [
        f"{name}: {reason}" for name, reason in unavailable.items()
    ]
    uncovered = [
        "R survival is not invoked; Exact ties are validated by brute-force tests in "
        "dev/tests/test_survival_risk_sets.py and test_cox_phase1_completion.py.",
        "Exact ties use only a small workload because elementary-symmetric dynamic "
        "programming scales with risk-set size and tied-event multiplicity.",
        "Crossover n is not estimated by the single quick/full target scale; use both "
        "scales before making a deployment threshold claim.",
    ] + missing_backend_notes

    cv_scenario = next(
        scenario
        for scenario in scenarios
        if scenario["name"] == "stratified_start_stop"
    )
    cv_matrix = _cv_evidence(cv_scenario, backends, args.seed + 100)

    output: Dict[str, Any] = {
        "method": "CoxPH survival Phase-1 completion",
        "backend_times": backend_times,
        "external_baseline": {
            "name": "statsmodels.duration.PHReg",
            "time": {
                name: value.get("statsmodels", {}).get("time_seconds")
                for name, value in details.items()
            },
            "version": _version("statsmodels"),
        },
        "precision_vs_external": external_precision,
        "convergence_status": convergence,
        "backend_precision": backend_precision,
        "compatibility_matrix": compatibility,
        "cv_matrix": cv_matrix,
        "inference_matrix": inference,
        "threshold_source": {
            "source": "dev/AGENTS.md strict inference gate",
            **GATE_THRESHOLDS,
        },
        "objective_scaling": (
            "un-normalized Cox log partial likelihood summed over observed events; "
            "timed backend scenarios use penalty=0, while CV evaluates explicit "
            "ridge candidates"
        ),
        "penalty_scale_mapping": (
            "CoxPH penalty lambda maximizes log_partial_likelihood - "
            "lambda * ||beta||^2 (information adds 2 * lambda * I); CV uses "
            "[0.0, 0.01, 0.1] with this same unnormalized scale and does not map "
            "to an external regularized estimator"
        ),
        "cpu_vs_external": cpu_vs_external,
        "gpu_vs_cpu": gpu_vs_cpu,
        "crossover_n": None,
        "target_scale_source": (
            "dev/plans/plan_survival.md and existing dev/benchmarks Cox scales"
        ),
        "optimization_notes": [
            "The standard no-entry path uses specialized vectorized kernels.",
            "Entry/start-stop/strata and Exact use the shared backend-native "
            "counting-process correctness engine.",
            "Fit timings include optimization, inference, and baseline estimation; "
            "C-index is disabled and transfer is reported separately.",
        ],
        "validation_tier": (
            "remote-full"
            if {"numpy", "cupy", "torch"}.issubset(backends)
            else "local-full"
        ),
        "schema_status": "unchecked",
        "gate_failures": [],
        "timing_scope": {
            "transfer": "host arrays to backend arrays, separately synchronized",
            "fit": "warm backend arrays through optimization + inference + baseline",
            "prediction": "host-side public predict_survival after fit",
            "gpu_sync": "before and after every transfer/fit timing",
        },
        "reproducibility": {
            "seed": args.seed,
            "scale": args.scale,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "dtype": "float64",
            "tol": 1e-9,
            "max_iter": 80,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "statgpu": _version("statgpu"),
                "numpy": _version("numpy"),
                "cupy": _version("cupy-cuda12x") or _version("cupy"),
                "torch": _version("torch"),
                "statsmodels": _version("statsmodels"),
            },
            "hardware": _gpu_metadata(),
        },
        "uncovered_reasons": uncovered,
        "details": details,
    }
    output["gate_failures"] = _collect_gate_failures(output, scenarios, backends)
    missing = sorted(REQUIRED_SCHEMA_KEYS - output.keys())
    if missing:
        output["uncovered_reasons"].append(f"missing schema keys: {missing}")
        output["schema_status"] = "missing-keys"
    elif output["gate_failures"]:
        output["schema_status"] = "failed-gates"
    else:
        output["schema_status"] = "ok"

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "backends": backends,
                "schema_status": output["schema_status"],
                "gate_failures": output["gate_failures"],
                "gpu_vs_cpu": gpu_vs_cpu,
                "precision_vs_external": external_precision,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if output["schema_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
