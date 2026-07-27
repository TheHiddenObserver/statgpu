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


def _extreme_data(device: str):
    X = np.array([[1000.0], [0.0]], dtype=np.float64)
    y = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
    coef = np.array([1.0], dtype=np.float64)
    return tuple(_to_backend(device, value) for value in (X, y, coef))


def _gradient_case(device: str, ties: str):
    X, y, coef = _extreme_data(device)
    loss = CoxPartialLikelihoodLoss(ties=ties)
    X_pre, y_pre = loss.preprocess(X, y)
    trusted = _to_numpy(loss.gradient_preprocessed(coef)).astype(np.float64)
    public = _to_numpy(loss.gradient(X_pre, y_pre, coef)).astype(np.float64)
    shared = _to_numpy(
        -loss._shared_objective(coef, compute_derivatives=True)["score"]
        / X.shape[0]
    ).astype(np.float64)
    return {
        "backend": DEVICE_NAMES[device],
        "device_argument": device,
        "ties": ties,
        "trusted_gradient": trusted.tolist(),
        "public_gradient": public.tolist(),
        "shared_gradient": shared.tolist(),
        "trusted_finite": bool(np.all(np.isfinite(trusted))),
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


def _fit_case(device: str, ties: str, penalty: str, alpha: float):
    X, y, _ = _extreme_data(device)
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
    model._init_coef = np.array([1.0], dtype=np.float64)
    _synchronize(device)
    started = time.perf_counter()
    model.fit(X, y)
    _synchronize(device)
    seconds = time.perf_counter() - started

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
        "ties": ties,
        "penalty": penalty,
        "alpha": alpha,
        "initial_coef": [1.0],
        "coef": coef.tolist(),
        "loss_value": loss_value,
        "penalty_value": penalty_value,
        "objective": objective,
        "smooth_gradient": smooth_gradient.tolist(),
        "penalty_gradient": penalty_gradient.tolist(),
        "kkt_max_abs": kkt,
        "all_finite": bool(np.all(np.isfinite(numeric_values))),
        "seconds": seconds,
        "n_iter": int(getattr(model, "n_iter_", 0)),
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
        "risk_sets": REPO_ROOT / "statgpu/survival/_risk_sets.py",
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
        "scenario": {
            "X": [[1000.0], [0.0]],
            "time": [1.0, 2.0],
            "event": [0.0, 1.0],
            "initial_coef": [1.0],
            "expected_gradient": [0.0],
        },
        "thresholds": {
            "gradient_max_abs": 1e-12,
            "coefficient_max_abs_vs_numpy": 1e-12,
            "objective_abs_vs_numpy": 1e-12,
            "kkt_max_abs": 1e-8,
        },
        "gradient_cases": [],
        "fit_cases": [],
        "gate_failures": [],
    }
    failures = report["gate_failures"]
    if report["tracked_worktree_dirty_before_run"]:
        failures.append("tracked worktree differs from recorded git commit")

    for ties in args.ties:
        for device in args.devices:
            result = _gradient_case(device, ties)
            report["gradient_cases"].append(result)
            for metric in (
                "trusted_public_max_abs",
                "trusted_shared_max_abs",
            ):
                if (
                    not result["trusted_finite"]
                    or result[metric] > report["thresholds"]["gradient_max_abs"]
                ):
                    failures.append(
                        f"gradient/{ties}/{result['backend']}: {metric} failed"
                    )

    for ties in args.ties:
        for penalty in args.penalties:
            reference = None
            for device in args.devices:
                result = _fit_case(device, ties, penalty, args.alpha)
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
                if not result["all_finite"]:
                    failures.append(
                        f"fit/{ties}/{penalty}/{result['backend']}: non-finite"
                    )
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
                        failures.append(
                            f"fit/{ties}/{penalty}/{result['backend']}: "
                            f"{metric} failed"
                        )

    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
