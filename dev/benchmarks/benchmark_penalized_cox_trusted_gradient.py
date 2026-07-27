"""Audit the extreme-range Cox SCAD/MCP trusted-gradient path on all backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import statgpu  # noqa: E402
from statgpu.linear_model import PenalizedCoxPHModel  # noqa: E402
from statgpu.losses import CoxPartialLikelihoodLoss  # noqa: E402
from statgpu.losses import _cox_ph as cox_loss_module  # noqa: E402


DEVICE_NAMES = {"cpu": "numpy", "cuda": "cupy", "torch": "torch"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _tracked_dirty() -> bool:
    return bool(_git_output("status", "--porcelain", "--untracked-files=no"))


def _synchronize(device: str) -> None:
    if device == "cuda":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif device == "torch":
        import torch

        torch.cuda.synchronize()


def _to_backend(device: str, value: np.ndarray):
    if device == "cuda":
        import cupy as cp

        return cp.asarray(value)
    if device == "torch":
        import torch

        return torch.as_tensor(value, dtype=torch.float64, device="cuda")
    return value.copy()


def _to_numpy(value) -> np.ndarray:
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    if module == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value) -> float:
    return float(np.asarray(_to_numpy(value)).reshape(()))


def _device_metadata(devices):
    metadata = {}
    if "cuda" in devices:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        metadata.update(
            {
                "cupy_version": cp.__version__,
                "cupy_gpu": (
                    name.decode("utf-8", "replace")
                    if isinstance(name, bytes)
                    else str(name)
                ),
                "cupy_compute_capability": [
                    int(properties["major"]),
                    int(properties["minor"]),
                ],
            }
        )
    if "torch" in devices:
        import torch

        metadata.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "torch_gpu": torch.cuda.get_device_name(0),
                "torch_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            }
        )
    return metadata


def _scenario_data(device: str, scenario: str, scale=None):
    if scenario == "departing_maximum":
        X = np.array([[1000.0], [0.0]], dtype=np.float64)
        y = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        coef = np.array([1.0], dtype=np.float64)
    elif scenario == "signed_moment_cancellation":
        if scale is None:
            raise ValueError("cancellation scenario requires a scale")
        X = np.array(
            [[-1.5], [0.5], [scale], [-scale + 1.0]],
            dtype=np.float64,
        )
        y = np.array(
            [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 0.0]],
            dtype=np.float64,
        )
        coef = np.array([0.0], dtype=np.float64)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return tuple(_to_backend(device, value) for value in (X, y, coef))


def _gradient_case(
    device: str,
    ties: str,
    scenario: str,
    scale=None,
):
    X, y, coef = _scenario_data(device, scenario, scale)
    loss = CoxPartialLikelihoodLoss(ties=ties)
    X_pre, y_pre = loss.preprocess(X, y)
    host_scalar_sync_calls = 0
    original_to_float_scalar = cox_loss_module._to_float_scalar

    def counting_to_float_scalar(value):
        nonlocal host_scalar_sync_calls
        host_scalar_sync_calls += 1
        return original_to_float_scalar(value)

    cox_loss_module._to_float_scalar = counting_to_float_scalar
    try:
        trusted = _to_numpy(loss.gradient_preprocessed(coef)).astype(np.float64)
    finally:
        cox_loss_module._to_float_scalar = original_to_float_scalar
    public = _to_numpy(loss.gradient(X_pre, y_pre, coef)).astype(np.float64)
    shared = _to_numpy(
        -loss._shared_objective(coef, compute_derivatives=True)["score"]
        / X.shape[0]
    ).astype(np.float64)
    return {
        "backend": DEVICE_NAMES[device],
        "device_argument": device,
        "scenario": scenario,
        "scale": scale,
        "ties": ties,
        "trusted_gradient": trusted.tolist(),
        "public_gradient": public.tolist(),
        "shared_gradient": shared.tolist(),
        "trusted_finite": bool(np.all(np.isfinite(trusted))),
        "trusted_host_scalar_sync_calls": host_scalar_sync_calls,
        "trusted_public_max_abs": float(np.max(np.abs(trusted - public))),
        "trusted_shared_max_abs": float(np.max(np.abs(trusted - shared))),
    }


def _kkt_residual(coef, smooth_gradient, penalty_gradient, alpha: float):
    coef = np.asarray(coef, dtype=np.float64)
    smooth_gradient = np.asarray(smooth_gradient, dtype=np.float64)
    penalty_gradient = np.asarray(penalty_gradient, dtype=np.float64)
    nonzero = np.abs(coef) > 1e-10
    residual = np.empty_like(coef)
    residual[nonzero] = np.abs(
        smooth_gradient[nonzero] + penalty_gradient[nonzero]
    )
    residual[~nonzero] = np.maximum(
        np.abs(smooth_gradient[~nonzero]) - alpha, 0.0
    )
    return float(np.max(residual))


def _fit_case(
    device: str,
    ties: str,
    penalty: str,
    alpha: float,
    scenario: str,
    scale=None,
):
    X, y, initial_coef = _scenario_data(device, scenario, scale)

    def timed_fit():
        model = PenalizedCoxPHModel(
            penalty=penalty,
            alpha=alpha,
            ties=ties,
            max_iter=5,
            max_lla_iters=1,
            tol=1e-6,
            device=device,
            gpu_memory_cleanup=True,
        )
        model._init_coef = _to_numpy(initial_coef).astype(np.float64)
        _synchronize(device)
        started = time.perf_counter()
        model.fit(X, y)
        _synchronize(device)
        return model, time.perf_counter() - started

    _, first_fit_seconds = timed_fit()
    model, steady_state_fit_seconds = timed_fit()

    coef = np.asarray(model.coef_, dtype=np.float64)
    coef_dev = _to_backend(device, coef)
    audit_loss = CoxPartialLikelihoodLoss(ties=ties)
    X_pre, y_pre = audit_loss.preprocess(X, y)
    loss_value = float(audit_loss.value(X_pre, y_pre, coef_dev))
    smooth_gradient = _to_numpy(
        audit_loss.gradient_preprocessed(coef_dev)
    ).astype(np.float64)
    penalty_value = float(model._penalty.value(coef_dev))
    penalty_gradient = _to_numpy(
        model._penalty.gradient(coef_dev)
    ).astype(np.float64)
    objective = loss_value + penalty_value
    kkt = _kkt_residual(
        coef, smooth_gradient, penalty_gradient, alpha
    )
    numeric_values = np.concatenate(
        [coef, smooth_gradient, penalty_gradient, [loss_value, objective, kkt]]
    )
    return {
        "backend": DEVICE_NAMES[device],
        "device_argument": device,
        "scenario": scenario,
        "scale": scale,
        "ties": ties,
        "penalty": penalty,
        "alpha": alpha,
        "initial_coef": _to_numpy(initial_coef).astype(np.float64).tolist(),
        "coef": coef.tolist(),
        "loss_value": loss_value,
        "penalty_value": penalty_value,
        "objective": objective,
        "smooth_gradient": smooth_gradient.tolist(),
        "penalty_gradient": penalty_gradient.tolist(),
        "kkt_max_abs": kkt,
        "all_finite": bool(np.all(np.isfinite(numeric_values))),
        "first_fit_in_warm_process_seconds": first_fit_seconds,
        "steady_state_fit_seconds": steady_state_fit_seconds,
        "n_iter": int(getattr(model, "n_iter_", 0)),
    }


def _workspace_case(device: str, n: int = 300_000):
    X_np = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
    y_np = np.column_stack(
        (
            np.arange(1, n + 1, dtype=np.float64),
            np.r_[np.zeros(n - 1), 1.0],
        )
    )
    if device == "cuda":
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        X, y = cp.asarray(X_np), cp.asarray(y_np)
        loss = CoxPartialLikelihoodLoss(ties="breslow")
        loss.preprocess(X, y)
        cp.cuda.Stream.null.synchronize()
        pool.free_all_blocks()
        baseline_bytes = pool.total_bytes()
        gradient = loss.gradient_preprocessed(cp.zeros(1, dtype=cp.float64))
        cp.cuda.Stream.null.synchronize()
        measured_bytes = max(0, pool.total_bytes() - baseline_bytes)
        measurement = "cupy_allocator_total_growth"
    elif device == "torch":
        import torch

        torch.cuda.empty_cache()
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        loss = CoxPartialLikelihoodLoss(ties="breslow")
        loss.preprocess(X, y)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gradient = loss.gradient_preprocessed(
            torch.zeros(1, dtype=torch.float64, device="cuda")
        )
        torch.cuda.synchronize()
        measured_bytes = max(
            0, torch.cuda.max_memory_allocated() - baseline_bytes
        )
        measurement = "torch_peak_active_delta"
    else:
        raise ValueError("workspace audit requires a physical GPU backend")
    return {
        "backend": DEVICE_NAMES[device],
        "device_argument": device,
        "n": n,
        "p": 1,
        "measurement": measurement,
        "baseline_bytes": int(baseline_bytes),
        "workspace_bytes": int(measured_bytes),
        "gradient_finite": bool(
            np.all(np.isfinite(_to_numpy(gradient)))
        ),
    }


def _performance_case(
    device: str,
    penalty: str,
    alpha: float,
    *,
    n: int = 4096,
    p: int = 12,
    time_bins: int = 64,
    warmups: int = 1,
    repeats: int = 5,
):
    rng = np.random.default_rng(8181)
    X_np = rng.normal(size=(n, p))
    time_np = rng.integers(1, time_bins + 1, size=n).astype(np.float64)
    event_np = rng.binomial(1, 0.7, size=n).astype(np.float64)
    event_np[0] = 1.0
    y_np = np.column_stack((time_np, event_np))
    X, y = _to_backend(device, X_np), _to_backend(device, y_np)

    def fit_once():
        model = PenalizedCoxPHModel(
            penalty=penalty,
            alpha=alpha,
            ties="efron",
            max_iter=35,
            max_lla_iters=4,
            tol=1e-4,
            device=device,
            gpu_memory_cleanup=False,
        )
        _synchronize(device)
        started = time.perf_counter()
        model.fit(X, y)
        _synchronize(device)
        return model, time.perf_counter() - started

    for _ in range(warmups):
        fit_once()
    runs = [fit_once() for _ in range(repeats)]
    timings = np.asarray([seconds for _, seconds in runs])
    representative_index = int(np.argsort(timings)[len(timings) // 2])
    representative = runs[representative_index][0]
    coef = np.asarray(representative.coef_, dtype=np.float64)
    return {
        "backend": DEVICE_NAMES[device],
        "device_argument": device,
        "penalty": penalty,
        "ties": "efron",
        "alpha": alpha,
        "n": n,
        "p": p,
        "time_bins": time_bins,
        "warmups": warmups,
        "repeats": repeats,
        "seconds": timings.tolist(),
        "median_seconds": float(np.median(timings)),
        "representative_run_index": representative_index,
        "coef": coef.tolist(),
        "n_iter": int(getattr(representative, "n_iter_", 0)),
        "all_finite": bool(np.all(np.isfinite(coef))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=list(DEVICE_NAMES),
        default=list(DEVICE_NAMES),
    )
    parser.add_argument(
        "--ties", nargs="+", choices=["breslow", "efron"], default=["breslow", "efron"]
    )
    parser.add_argument(
        "--penalties", nargs="+", choices=["scad", "mcp"], default=["scad", "mcp"]
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/benchmark_frontend_sources/"
            "penalized_cox_trusted_gradient_pr80_20260727.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    script_path = Path(__file__).resolve()
    source_paths = {
        "cox_ph_loss": REPO_ROOT / "statgpu/losses/_cox_ph.py",
        "fista_lla": REPO_ROOT / "statgpu/solvers/_fista_lla.py",
        "fit_mixin": REPO_ROOT / "statgpu/linear_model/penalized/_fit_mixin.py",
        "penalized_cox_estimator": (
            REPO_ROOT / "statgpu/linear_model/penalized/_penalized_cox.py"
        ),
        "risk_sets": REPO_ROOT / "statgpu/survival/_risk_sets.py",
        "backend_array_ops": REPO_ROOT / "statgpu/backends/_array_ops.py",
        "backend_utils": REPO_ROOT / "statgpu/backends/_utils.py",
    }
    report = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "tracked_worktree_dirty_before_run": _tracked_dirty(),
        "statgpu_version": statgpu.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_hashes": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "benchmark_sha256": _sha256(script_path),
        "command_argv": [
            sys.executable,
            str(script_path.relative_to(REPO_ROOT)),
            *sys.argv[1:],
        ],
        "device_metadata": _device_metadata(args.devices),
        "scenarios": {
            "departing_maximum": {
                "X": [[1000.0], [0.0]],
                "time": [1.0, 2.0],
                "event": [0.0, 1.0],
                "initial_coef": [1.0],
                "expected_gradient": [0.0],
            },
            "signed_moment_cancellation": {
                "X_template": [[-1.5], [0.5], ["scale"], ["-scale + 1"]],
                "scales": [1e8, 1e12, 1e15],
                "time": [1.0, 2.0, 3.0, 4.0],
                "event": [0.0, 1.0, 0.0, 0.0],
                "initial_coef": [0.0],
                "expected_gradient": [0.0],
            },
        },
        "timing_protocol": {
            "cold_process_measured": False,
            "cold_process_fit_seconds": None,
            "python_import_and_backend_initialization_included": False,
            "gradient_path_preheated_before_fit_cases": True,
            "first_fit_in_warm_process": (
                "first estimator fit for that scenario/ties/penalty/backend; "
                "may include estimator or proximal first-use compilation"
            ),
            "steady_state_fit": (
                "second fresh estimator fit immediately after the first fit"
            ),
            "performance_timing": (
                "one synchronized warmup is excluded, then five synchronized "
                "fresh-estimator fits are timed and summarized by the median"
            ),
            "cupy_rawkernel_jit_applicable": False,
            "interpretation": (
                "diagnostic warm-process timing, not fresh-Python cold-start latency"
            ),
        },
        "thresholds": {
            "gradient_max_abs": 1e-12,
            "coefficient_max_abs_vs_numpy": 1e-12,
            "objective_abs_vs_numpy": 1e-10,
            "kkt_max_abs": 1e-8,
            "expected_predictor_range_sync_calls": 1,
            "workspace_max_bytes": 64 * 1024 * 1024,
            "performance_coefficient_max_abs_vs_numpy": 1e-8,
            "performance_gpu_speedup_min": 1.0,
        },
        "gradient_cases": [],
        "fit_cases": [],
        "performance_cases": [],
        "workspace_cases": [],
        "gate_failures": [],
    }
    failures = report["gate_failures"]
    if report["tracked_worktree_dirty_before_run"]:
        failures.append("tracked worktree differs from recorded git commit")

    scenarios = [("departing_maximum", None)] + [
        ("signed_moment_cancellation", scale)
        for scale in (1e8, 1e12, 1e15)
    ]
    for scenario, scale in scenarios:
        for ties in args.ties:
            for device in args.devices:
                result = _gradient_case(
                    device, ties, scenario, scale
                )
                report["gradient_cases"].append(result)
                expected_syncs = report["thresholds"][
                    "expected_predictor_range_sync_calls"
                ]
                if result["trusted_host_scalar_sync_calls"] != expected_syncs:
                    failures.append(
                        f"gradient/{scenario}/{scale}/{ties}/"
                        f"{result['backend']}: unexpected scalar-sync count"
                    )
                for metric in (
                    "trusted_public_max_abs",
                    "trusted_shared_max_abs",
                ):
                    if (
                        not result["trusted_finite"]
                        or result[metric]
                        > report["thresholds"]["gradient_max_abs"]
                    ):
                        failures.append(
                            f"gradient/{scenario}/{scale}/{ties}/"
                            f"{result['backend']}: {metric} failed"
                        )

    for scenario, scale in scenarios:
        for ties in args.ties:
            for penalty in args.penalties:
                reference = None
                for device in args.devices:
                    result = _fit_case(
                        device,
                        ties,
                        penalty,
                        args.alpha,
                        scenario,
                        scale,
                    )
                    report["fit_cases"].append(result)
                    if reference is None:
                        reference = result
                    result["coef_max_abs_vs_numpy"] = float(
                        np.max(
                            np.abs(
                                np.asarray(result["coef"])
                                - np.asarray(reference["coef"])
                            )
                        )
                    )
                    result["objective_abs_vs_numpy"] = abs(
                        result["objective"] - reference["objective"]
                    )
                    case_label = (
                        f"fit/{scenario}/{scale}/{ties}/{penalty}/"
                        f"{result['backend']}"
                    )
                    if not result["all_finite"]:
                        failures.append(f"{case_label}: non-finite")
                    for metric in (
                        "coef_max_abs_vs_numpy",
                        "objective_abs_vs_numpy",
                        "kkt_max_abs",
                    ):
                        limit_name = (
                            "coefficient_max_abs_vs_numpy"
                            if metric == "coef_max_abs_vs_numpy"
                            else metric
                        )
                        if result[metric] > report["thresholds"][limit_name]:
                            failures.append(f"{case_label}: {metric} failed")

    for device in args.devices:
        if device == "cpu":
            continue
        result = _workspace_case(device)
        report["workspace_cases"].append(result)
        if (
            not result["gradient_finite"]
            or result["workspace_bytes"]
            > report["thresholds"]["workspace_max_bytes"]
        ):
            failures.append(f"workspace/{result['backend']}: gate failed")

    for penalty in args.penalties:
        performance_results = [
            _performance_case(device, penalty, 0.04)
            for device in args.devices
        ]
        reference = next(
            (
                result
                for result in performance_results
                if result["device_argument"] == "cpu"
            ),
            None,
        )
        for result in performance_results:
            if reference is None:
                result["coef_max_abs_vs_numpy"] = None
                result["speedup_vs_numpy"] = None
            else:
                result["coef_max_abs_vs_numpy"] = float(
                    np.max(
                        np.abs(
                            np.asarray(result["coef"])
                            - np.asarray(reference["coef"])
                        )
                    )
                )
                result["speedup_vs_numpy"] = float(
                    reference["median_seconds"] / result["median_seconds"]
                )
            report["performance_cases"].append(result)
            label = f"performance/{penalty}/{result['backend']}"
            if not result["all_finite"]:
                failures.append(f"{label}: non-finite coefficient")
            if (
                result["coef_max_abs_vs_numpy"] is not None
                and result["coef_max_abs_vs_numpy"]
                > report["thresholds"][
                    "performance_coefficient_max_abs_vs_numpy"
                ]
            ):
                failures.append(f"{label}: coefficient parity failed")
            if (
                result["device_argument"] != "cpu"
                and result["speedup_vs_numpy"] is not None
                and result["speedup_vs_numpy"]
                < report["thresholds"]["performance_gpu_speedup_min"]
            ):
                failures.append(f"{label}: GPU speedup gate failed")

    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
