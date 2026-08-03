#!/usr/bin/env python3
"""Exact-source physical-GPU gate for weighted Group MCP/SCAD direct fit and CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


SOURCE_FILES = (
    "dev/benchmarks/benchmark_group_nonconvex_weighted_gpu.py",
    "dev/tests/test_pr80_group_nonconvex_weighted_contract.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/penalties/__init__.py",
    "statgpu/penalties/_group_lasso_layout.py",
    "statgpu/penalties/_group_nonconvex_layout.py",
    "statgpu/solvers/__init__.py",
    "statgpu/solvers/_fista_lla.py",
    "statgpu/solvers/_fista_lla_group_contract.py",
)
GROUPS = [[0, 3], [1, 2]]


def _git(*args):
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _data():
    rng = np.random.default_rng(9901)
    X = rng.normal(size=(96, 4))
    y = 0.25 + X @ np.array([0.8, -0.5, 0.3, 0.65])
    y += rng.normal(scale=0.08, size=X.shape[0])
    weights = np.linspace(0.4, 1.8, X.shape[0])
    return X, y, weights


def _backend(name, X, y, weights):
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
            cp.asarray(y),
            cp.asarray(weights),
            device_name,
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        torch.as_tensor(weights, dtype=torch.float64, device="cuda"),
        torch.cuda.get_device_name(0),
    )


def _kwargs(kind):
    kwargs = {"groups": GROUPS}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    else:
        kwargs["a"] = 3.7
    return kwargs


def _fit(kind, X, y, weights, device):
    return PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha=0.16,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=500,
        tol=1e-8,
        max_lla_iters=20,
        lla_tol=1e-8,
    ).fit(X, y, sample_weight=weights)


def _cv(kind, X, y, weights, device):
    return PenalizedGLM_CV(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha_grid=[0.2, 0.1],
        cv=2,
        random_state=23,
        device=device,
        max_iter=400,
        tol=1e-7,
    ).fit(X, y, sample_weight=weights)


def _run_backend(name):
    X, y, weights = _data()
    device, Xb, yb, wb, device_name = _backend(name, X, y, weights)
    cases = {}
    for kind in ("group_mcp", "group_scad"):
        cpu = _fit(kind, X, y, weights, "cpu")
        gpu = _fit(kind, Xb, yb, wb, device)
        coef_error = float(
            np.max(np.abs(_as_numpy(gpu.coef_) - np.asarray(cpu.coef_)))
        )
        pred_error = float(
            np.max(
                np.abs(_as_numpy(gpu.predict(Xb)) - np.asarray(cpu.predict(X)))
            )
        )
        intercept_error = abs(float(gpu.intercept_) - float(cpu.intercept_))

        cpu_cv = _cv(kind, X, y, weights, "cpu")
        gpu_cv = _cv(kind, Xb, yb, wb, device)
        score_error = float(
            np.max(
                np.abs(
                    np.asarray(gpu_cv.cv_results_["all_scores"])
                    - np.asarray(cpu_cv.cv_results_["all_scores"])
                )
            )
        )
        cv_coef_error = float(
            np.max(
                np.abs(_as_numpy(gpu_cv.coef_) - np.asarray(cpu_cv.coef_))
            )
        )
        selected_equal = bool(np.isclose(gpu_cv.alpha_, cpu_cv.alpha_))
        refit_equal = bool(np.isclose(gpu_cv.estimator_.alpha, gpu_cv.alpha_))
        passed = all(
            (
                coef_error <= 5e-5,
                pred_error <= 5e-5,
                intercept_error <= 5e-5,
                score_error <= 6e-5,
                cv_coef_error <= 6e-5,
                selected_equal,
                refit_equal,
            )
        )
        cases[kind] = {
            "direct_coef_max_abs_error": coef_error,
            "direct_prediction_max_abs_error": pred_error,
            "direct_intercept_abs_error": intercept_error,
            "cv_score_max_abs_error": score_error,
            "cv_coef_max_abs_error": cv_coef_error,
            "selected_alpha": float(gpu_cv.alpha_),
            "cpu_selected_alpha": float(cpu_cv.alpha_),
            "final_refit_alpha": float(gpu_cv.estimator_.alpha),
            "passed": bool(passed),
        }
    return device_name, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dirty = bool(_git("status", "--porcelain"))
    report = {
        "schema_version": 1,
        "validation_tier": "remote-full",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_clean": not dirty,
        "source_sha256": {path: _sha256(path) for path in SOURCE_FILES},
        "backends": {},
        "gate_failures": [],
    }
    for name in ("cupy", "torch"):
        try:
            device_name, cases = _run_backend(name)
            passed = all(case["passed"] for case in cases.values())
            report["backends"][name] = {
                "device": device_name,
                "cases": cases,
                "passed": bool(passed),
            }
            if not passed:
                report["gate_failures"].append(
                    f"{name}: weighted group nonconvex parity"
                )
        except Exception as exc:
            report["backends"][name] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["gate_failures"].append(f"{name}: {type(exc).__name__}")
    if dirty:
        report["gate_failures"].append("source tree is dirty")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
