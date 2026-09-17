#!/usr/bin/env python3
"""Physical CUDA acceptance for Quantile Group SCAD/MCP FISTA-LLA routing."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
import statgpu.solvers as solvers


SCHEMA_VERSION = 1
Q = 0.35
GROUPS = [[0, 1], [2, 3]]
DIRECT_ALPHA = 0.04
ALPHA_GRID = np.asarray([0.05, 0.03], dtype=np.float64)
ATOL_OBJECTIVE = 8e-5
ATOL_PARAM = 1e-4
ATOL_CV_SCORE = 1e-4


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_state():
    return _git("rev-parse", "HEAD"), not bool(_git("status", "--porcelain"))


def _require_gpu_backends():
    import cupy as cp
    import torch

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    cp.cuda.Device(0).use()
    torch.cuda.set_device(0)
    return cp, torch


def _data(seed=166401, n=64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    beta = np.array([0.85, -0.42, 0.28, 0.16], dtype=np.float64)
    y = (0.25 + X @ beta + rng.laplace(scale=0.16, size=n)).astype(np.float64)
    weights = np.linspace(0.45, 1.85, n, dtype=np.float64)
    rng.shuffle(weights)
    half = n // 2
    folds = [
        (np.arange(0, half), np.arange(half, n)),
        (np.arange(half, n), np.arange(0, half)),
    ]
    return X, y, weights, folds


def _penalty_kwargs(kind):
    result = {"groups": GROUPS}
    if kind == "group_scad":
        result["a"] = 3.7
    else:
        result["gamma"] = 3.0
    return result


def _native_inputs(backend, X, y, weights, cp, torch):
    if backend == "cupy":
        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            cp.asarray(weights, dtype=cp.float64),
            "cuda",
        )
    device = torch.device("cuda:0")
    return (
        torch.as_tensor(X, dtype=torch.float64, device=device),
        torch.as_tensor(y, dtype=torch.float64, device=device),
        torch.as_tensor(weights, dtype=torch.float64, device=device),
        "torch",
    )


def _with_lla_counter(fn):
    original = solvers.fista_lla_path
    count = {"value": 0}

    def counted(*args, **kwargs):
        count["value"] += 1
        return original(*args, **kwargs)

    solvers.fista_lla_path = counted
    try:
        result = fn()
    finally:
        solvers.fista_lla_path = original
    if count["value"] <= 0:
        raise AssertionError("Quantile group route did not execute fista_lla_path")
    return result, int(count["value"])


def _fit(kind, X, y, weights, device):
    return PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha=DIRECT_ALPHA,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=500,
        tol=1e-7,
        max_lla_iters=18,
        lla_tol=1e-7,
    ).fit(X, y, sample_weight=weights)


def _cv(kind, X, y, weights, folds, device):
    return PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind),
        alpha_grid=ALPHA_GRID,
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="auto",
        device=device,
        max_iter=400,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)


def _host(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _parameter_error(actual, reference):
    coef_error = float(
        np.max(
            np.abs(
                _host(actual.coef_).reshape(-1)
                - _host(reference.coef_).reshape(-1)
            )
        )
    )
    intercept_error = abs(float(actual.intercept_) - float(reference.intercept_))
    return max(coef_error, intercept_error), coef_error, intercept_error


def _objective(model, X, y, weights):
    coef = _host(model.coef_).reshape(-1)
    residual = y - (X @ coef + float(model.intercept_))
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(np.average(pinball, weights=weights))
    penalty = float(model._penalty.value(coef))
    return fit + penalty


def _provenance(model, backend):
    observed = {
        "solver": str(getattr(model, "_selected_solver", "") or ""),
        "backend": str(getattr(model, "_selected_backend_name", "") or ""),
        "device": str(getattr(model, "_selected_backend_device", "") or ""),
    }
    if observed["solver"] != "fista":
        raise AssertionError(f"resolved solver drifted: {observed['solver']!r}")
    if observed["backend"] != backend:
        raise AssertionError(f"backend drifted: {observed['backend']!r} != {backend!r}")
    if observed["device"] != "cuda:0":
        raise AssertionError(f"device drifted: {observed['device']!r}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="dev/reviews/pr166_quantile_group_lla_gpu.json"
    )
    args = parser.parse_args()

    source_sha, source_clean = _source_state()
    cp, torch = _require_gpu_backends()
    X, y, weights, folds = _data()

    cpu = {}
    for kind in ("group_scad", "group_mcp"):
        direct, direct_calls = _with_lla_counter(
            lambda kind=kind: _fit(kind, X, y, weights, "cpu")
        )
        cv, cv_calls = _with_lla_counter(
            lambda kind=kind: _cv(kind, X, y, weights, folds, "cpu")
        )
        cpu[kind] = {
            "direct": direct,
            "direct_calls": direct_calls,
            "objective": _objective(direct, X, y, weights),
            "cv": cv,
            "cv_calls": cv_calls,
            "scores": np.asarray(cv.cv_results_["all_scores"], dtype=np.float64),
        }

    cases = []
    max_objective_error = 0.0
    max_direct_param_error = 0.0
    max_cv_score_error = 0.0
    max_cv_param_error = 0.0
    for backend in ("cupy", "torch"):
        Xb, yb, wb, device = _native_inputs(backend, X, y, weights, cp, torch)
        for kind in ("group_scad", "group_mcp"):
            direct, direct_calls = _with_lla_counter(
                lambda kind=kind: _fit(kind, Xb, yb, wb, device)
            )
            param_error, coef_error, intercept_error = _parameter_error(
                direct, cpu[kind]["direct"]
            )
            objective = _objective(direct, X, y, weights)
            objective_error = abs(objective - float(cpu[kind]["objective"]))
            max_direct_param_error = max(max_direct_param_error, param_error)
            max_objective_error = max(max_objective_error, objective_error)
            if param_error > ATOL_PARAM:
                raise AssertionError(
                    f"{backend}/direct/{kind}: parameter error {param_error:.3e} > {ATOL_PARAM:.3e}"
                )
            if objective_error > ATOL_OBJECTIVE:
                raise AssertionError(
                    f"{backend}/direct/{kind}: objective error {objective_error:.3e} > {ATOL_OBJECTIVE:.3e}"
                )
            cases.append(
                {
                    "name": f"{backend}/direct/{kind}",
                    "provenance": _provenance(direct, backend),
                    "fista_lla_calls": direct_calls,
                    "parameter_error": param_error,
                    "coef_error": coef_error,
                    "intercept_error": intercept_error,
                    "objective": objective,
                    "cpu_objective": float(cpu[kind]["objective"]),
                    "objective_error": objective_error,
                }
            )

            cv, cv_calls = _with_lla_counter(
                lambda kind=kind: _cv(kind, Xb, yb, wb, folds, device)
            )
            scores = np.asarray(cv.cv_results_["all_scores"], dtype=np.float64)
            score_error = float(np.max(np.abs(scores - cpu[kind]["scores"])))
            cv_param_error, cv_coef_error, cv_intercept_error = _parameter_error(
                cv, cpu[kind]["cv"]
            )
            max_cv_score_error = max(max_cv_score_error, score_error)
            max_cv_param_error = max(max_cv_param_error, cv_param_error)
            if score_error > ATOL_CV_SCORE:
                raise AssertionError(
                    f"{backend}/cv/{kind}: score error {score_error:.3e} > {ATOL_CV_SCORE:.3e}"
                )
            if cv_param_error > ATOL_PARAM:
                raise AssertionError(
                    f"{backend}/cv/{kind}: final-refit parameter error {cv_param_error:.3e} > {ATOL_PARAM:.3e}"
                )
            if float(cv.alpha_) != float(cpu[kind]["cv"].alpha_):
                raise AssertionError(
                    f"{backend}/cv/{kind}: alpha {cv.alpha_!r} != CPU {cpu[kind]['cv'].alpha_!r}"
                )
            cases.append(
                {
                    "name": f"{backend}/cv/{kind}",
                    "provenance": _provenance(cv.estimator_, backend),
                    "fista_lla_calls": cv_calls,
                    "selected_alpha": float(cv.alpha_),
                    "cpu_selected_alpha": float(cpu[kind]["cv"].alpha_),
                    "score_error": score_error,
                    "final_refit_parameter_error": cv_param_error,
                    "final_refit_coef_error": cv_coef_error,
                    "final_refit_intercept_error": cv_intercept_error,
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "quantile": Q,
        "groups": GROUPS,
        "cases": cases,
        "max_errors": {
            "direct_objective": max_objective_error,
            "direct_parameter": max_direct_param_error,
            "cv_score": max_cv_score_error,
            "cv_final_refit_parameter": max_cv_param_error,
        },
        "tolerances": {
            "direct_objective": ATOL_OBJECTIVE,
            "parameter": ATOL_PARAM,
            "cv_score": ATOL_CV_SCORE,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_ordinal": 0,
            "cupy_device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "torch_device_name": torch.cuda.get_device_name(0),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
