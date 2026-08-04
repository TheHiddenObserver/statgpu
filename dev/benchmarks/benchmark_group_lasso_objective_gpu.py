#!/usr/bin/env python3
"""Exact-source physical-GPU gate for public Group Lasso objectives.

This runner certifies the post-review contract on both CuPy and Torch:
- correlated squared-error Group Lasso satisfies the composite KKT system;
- Huber Group Lasso optimizes Huber rather than the removed Gaussian block path;
- weighted squared-error Group Lasso honors sample weights;
- direct fit, predictions, CV scores, selected alpha, and final refit match CPU.
"""

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
    "dev/benchmarks/benchmark_group_lasso_objective_gpu.py",
    "dev/tests/test_pr80_group_lasso_exact_objective_contract.py",
    "dev/tests/test_pr80_group_lasso_nonquadratic_contract.py",
    "dev/tests/test_pr80_group_lasso_weighted_contract.py",
    "dev/tests/test_pr80_group_input_contract.py",
    "dev/tests/test_pr80_group_cv_list_input_contract.py",
    "dev/tests/test_pr80_group_formula_contract.py",
    "dev/tests/test_pr80_group_dimension_contract.py",
    "statgpu/linear_model/penalized/__init__.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/linear_model/penalized/_group_penalty_model_contract.py",
    "statgpu/penalties/__init__.py",
    "statgpu/penalties/_group_lasso.py",
    "statgpu/penalties/_group_lasso_layout.py",
    "statgpu/penalties/_group_dimension_contract.py",
    "statgpu/solvers/_fista.py",
    "statgpu/solvers/_utils.py",
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


def _backend(name, X, y, weights=None):
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
            None if weights is None else cp.asarray(weights),
            device_name,
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        None
        if weights is None
        else torch.as_tensor(weights, dtype=torch.float64, device="cuda"),
        torch.cuda.get_device_name(0),
    )


def _correlated_data(seed):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=180)
    z1 = rng.normal(size=180)
    X = np.column_stack(
        [
            z0 + 0.05 * rng.normal(size=180),
            z1 + 0.08 * rng.normal(size=180),
            0.85 * z1 + 0.12 * rng.normal(size=180),
            0.9 * z0 + 0.1 * rng.normal(size=180),
        ]
    )
    y = 0.35 + X @ np.array([0.7, -0.45, 0.25, 0.55])
    y += rng.normal(scale=0.07, size=X.shape[0])
    return X, y


def _outlier_data(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(140, 4))
    y = 0.25 + X @ np.array([0.9, -0.55, 0.3, 0.7])
    y += rng.normal(scale=0.08, size=X.shape[0])
    y[:8] += np.array([18.0, -16.0, 15.0, -14.0, 13.0, -12.0, 11.0, -10.0])
    return X, y


def _weighted_data(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    y = 0.2 + X @ np.array([0.9, -0.5, 0.25, 0.65])
    y += rng.normal(scale=0.07, size=X.shape[0])
    y[:6] += np.array([15.0, -13.0, 11.0, -9.0, 8.0, -7.0])
    weights = np.ones(X.shape[0])
    weights[:6] = 0.02
    weights[60:] = 1.7
    return X, y, weights


def _model(loss, device, alpha, X, y, weights=None):
    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        loss_kwargs={"delta": 1.0} if loss == "huber" else None,
        penalty="group_lasso",
        penalty_kwargs={"groups": GROUPS},
        alpha=alpha,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=5000,
        tol=1e-10,
    )
    return model.fit(X, y, sample_weight=weights)


def _cv(loss, device, X, y, weights=None):
    return PenalizedGLM_CV(
        loss=loss,
        loss_kwargs={"delta": 1.0} if loss == "huber" else None,
        penalty="group_lasso",
        penalty_kwargs={"groups": GROUPS},
        alpha_grid=[0.18, 0.09],
        cv=2,
        random_state=37,
        device=device,
        max_iter=2500,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)


def _composite_kkt(model, X, y, weights=None):
    X_work = np.column_stack([X, np.ones(X.shape[0])])
    params = np.append(np.asarray(model.coef_), float(model.intercept_))
    gradient = np.asarray(
        model._loss.gradient(
            X_work,
            y,
            params,
            sample_weight=weights,
        )
    )
    residuals = [abs(float(gradient[-1]))]
    for group in GROUPS:
        idx = np.asarray(group, dtype=np.int64)
        beta_g = np.asarray(model.coef_)[idx]
        grad_g = gradient[idx]
        norm = np.linalg.norm(beta_g)
        threshold = model.alpha * np.sqrt(idx.size)
        if norm > 1e-9:
            residuals.append(
                np.linalg.norm(grad_g + threshold * beta_g / norm)
            )
        else:
            residuals.append(max(np.linalg.norm(grad_g) - threshold, 0.0))
    return float(max(residuals))


def _direct_case(name, loss, X, y, weights, alpha, backend_name):
    cpu = _model(loss, "cpu", alpha, X, y, weights)
    device, Xb, yb, wb, device_name = _backend(
        backend_name, X, y, weights
    )
    gpu = _model(loss, device, alpha, Xb, yb, wb)
    coef_error = float(
        np.max(np.abs(_as_numpy(gpu.coef_) - np.asarray(cpu.coef_)))
    )
    pred_error = float(
        np.max(np.abs(_as_numpy(gpu.predict(Xb)) - np.asarray(cpu.predict(X))))
    )
    intercept_error = abs(float(gpu.intercept_) - float(cpu.intercept_))
    cpu_kkt = _composite_kkt(cpu, X, y, weights)
    gpu_kkt = _composite_kkt(gpu, X, y, weights)
    passed = all(
        (
            coef_error <= 6e-5,
            pred_error <= 6e-5,
            intercept_error <= 6e-5,
            cpu_kkt <= 4e-4,
            gpu_kkt <= 4e-4,
            cpu._selected_solver == "fista",
            gpu._selected_solver == "fista",
        )
    )
    return device_name, {
        "name": name,
        "coef_max_abs_error": coef_error,
        "prediction_max_abs_error": pred_error,
        "intercept_abs_error": intercept_error,
        "cpu_composite_kkt": cpu_kkt,
        "gpu_composite_kkt": gpu_kkt,
        "cpu_solver": cpu._selected_solver,
        "gpu_solver": gpu._selected_solver,
        "passed": bool(passed),
    }


def _cv_case(name, loss, X, y, weights, backend_name):
    cpu = _cv(loss, "cpu", X, y, weights)
    device, Xb, yb, wb, device_name = _backend(
        backend_name, X, y, weights
    )
    gpu = _cv(loss, device, Xb, yb, wb)
    score_error = float(
        np.max(
            np.abs(
                np.asarray(gpu.cv_results_["all_scores"])
                - np.asarray(cpu.cv_results_["all_scores"])
            )
        )
    )
    coef_error = float(
        np.max(np.abs(_as_numpy(gpu.coef_) - np.asarray(cpu.coef_)))
    )
    selected_equal = bool(np.isclose(gpu.alpha_, cpu.alpha_))
    refit_equal = bool(np.isclose(gpu.estimator_.alpha, gpu.alpha_))
    passed = all(
        (
            score_error <= 8e-5,
            coef_error <= 8e-5,
            selected_equal,
            refit_equal,
        )
    )
    return device_name, {
        "name": name,
        "score_max_abs_error": score_error,
        "coef_max_abs_error": coef_error,
        "selected_alpha": float(gpu.alpha_),
        "cpu_selected_alpha": float(cpu.alpha_),
        "final_refit_alpha": float(gpu.estimator_.alpha),
        "passed": bool(passed),
    }


def _backend_cases(name):
    X_corr, y_corr = _correlated_data(10601)
    X_huber, y_huber = _outlier_data(10602)
    X_weighted, y_weighted, weights = _weighted_data(10603)
    cases = {}
    device_names = []
    for case_name, loss, X, y, w, alpha in (
        ("correlated_squared", "squared_error", X_corr, y_corr, None, 0.08),
        ("huber_outliers", "huber", X_huber, y_huber, None, 0.12),
        ("weighted_squared", "squared_error", X_weighted, y_weighted, weights, 0.11),
    ):
        device_name, result = _direct_case(
            case_name, loss, X, y, w, alpha, name
        )
        device_names.append(device_name)
        cases[f"direct_{case_name}"] = result
    for case_name, loss, X, y, w in (
        ("huber", "huber", X_huber, y_huber, None),
        ("weighted_squared", "squared_error", X_weighted, y_weighted, weights),
    ):
        device_name, result = _cv_case(case_name, loss, X, y, w, name)
        device_names.append(device_name)
        cases[f"cv_{case_name}"] = result
    return device_names[0], cases


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
            device_name, cases = _backend_cases(name)
            passed = all(case["passed"] for case in cases.values())
            report["backends"][name] = {
                "device": device_name,
                "cases": cases,
                "passed": bool(passed),
            }
            if not passed:
                report["gate_failures"].append(
                    f"{name}: Group Lasso objective/CV parity"
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
