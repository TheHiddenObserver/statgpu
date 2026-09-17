#!/usr/bin/env python3
"""Physical CUDA acceptance for automatic Quantile Group Proximal IRLS-LLA."""

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
from statgpu.solvers import _quantile_group_proximal_irls_lla as group_solver


SCHEMA_VERSION = 2
Q = 0.35
GROUPS = [[0, 1], [2, 3]]
DIRECT_ALPHA = 0.04
ALPHA_GRID = np.asarray([0.05, 0.03], dtype=np.float64)
ATOL_OBJECTIVE = 8e-5
ATOL_PARAM = 1e-4
ATOL_CV_SCORE = 1e-4
ATOL_FLAT_REFERENCE = 5e-5
EXECUTED_SOLVER = "group_proximal_irls_lla"


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


def _array_backend_and_device(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        return "cupy", f"cuda:{int(value.device.id)}"
    if module.startswith("torch"):
        return "torch", str(value.device)
    return "numpy", "cpu"


def _with_group_solver_counter(fn, *, expected_backend, min_calls=1):
    """Prove every automatic Group solver call owns the expected arrays."""
    original = group_solver.quantile_group_proximal_irls_lla_solver
    count = {"value": 0}
    observed_locations = set()

    def counted(*args, **kwargs):
        count["value"] += 1
        if len(args) < 4:
            raise AssertionError("Group Proximal IRLS-LLA did not expose X/y positionally")
        X_arg, y_arg = args[2], args[3]
        sw_arg = kwargs.get("sample_weight")
        X_backend, X_device = _array_backend_and_device(X_arg)
        y_backend, y_device = _array_backend_and_device(y_arg)
        if X_backend != expected_backend or y_backend != expected_backend:
            raise AssertionError(
                "Quantile Group solver backend drifted: "
                f"X={X_backend!r}, y={y_backend!r}, expected={expected_backend!r}"
            )
        if expected_backend in ("cupy", "torch"):
            if X_device != "cuda:0" or y_device != "cuda:0":
                raise AssertionError(
                    f"Quantile Group solver device drifted: X={X_device!r}, y={y_device!r}"
                )
        if sw_arg is None:
            raise AssertionError("weighted Quantile Group solver lost sample_weight")
        sw_backend, sw_device = _array_backend_and_device(sw_arg)
        if sw_backend != expected_backend:
            raise AssertionError(
                f"Quantile Group weight backend drifted: {sw_backend!r} != {expected_backend!r}"
            )
        if expected_backend in ("cupy", "torch") and sw_device != "cuda:0":
            raise AssertionError(
                f"Quantile Group weight device drifted: {sw_device!r}"
            )
        observed_locations.add((X_backend, X_device, sw_backend, sw_device))
        return original(*args, **kwargs)

    group_solver.quantile_group_proximal_irls_lla_solver = counted
    try:
        result = fn()
    finally:
        group_solver.quantile_group_proximal_irls_lla_solver = original

    if count["value"] < int(min_calls):
        raise AssertionError(
            f"Quantile Group auto route executed {count['value']} calls, expected >= {min_calls}"
        )
    return result, int(count["value"]), sorted(observed_locations)


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


def _unpenalized_reference(X, y, weights):
    return PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="l2",
        alpha=0.0,
        solver="irls",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=800,
        tol=1e-9,
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


def _pinball_fit(coef, intercept, X, y, weights):
    residual = y - (X @ coef + float(intercept))
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    return float(np.average(pinball, weights=weights))


def _objective(model, X, y, weights):
    coef = _host(model.coef_).reshape(-1)
    fit = _pinball_fit(coef, model.intercept_, X, y, weights)
    return fit + float(model._penalty.value(coef))


def _flat_threshold(kind):
    group_scale = np.sqrt(2.0)
    if kind == "group_scad":
        return 3.7 * DIRECT_ALPHA * group_scale
    return 3.0 * DIRECT_ALPHA * group_scale


def _validate_flat_reference(kind, model, reference, X, y, weights):
    coef = _host(model.coef_).reshape(-1)
    ref_coef = _host(reference.coef_).reshape(-1)
    threshold = _flat_threshold(kind)
    group_norms = np.asarray(
        [np.linalg.norm(coef[np.asarray(group, dtype=int)]) for group in GROUPS],
        dtype=np.float64,
    )
    ref_group_norms = np.asarray(
        [np.linalg.norm(ref_coef[np.asarray(group, dtype=int)]) for group in GROUPS],
        dtype=np.float64,
    )
    if not np.all(ref_group_norms > threshold):
        raise AssertionError(
            f"{kind}: reference fixture is not in the flat region: "
            f"norms={ref_group_norms!r}, threshold={threshold:.6g}"
        )
    if not np.all(group_norms > threshold):
        raise AssertionError(
            f"{kind}: fitted groups did not reach the flat region: "
            f"norms={group_norms!r}, threshold={threshold:.6g}"
        )
    fit = _pinball_fit(coef, model.intercept_, X, y, weights)
    ref_fit = _pinball_fit(ref_coef, reference.intercept_, X, y, weights)
    param_error, coef_error, intercept_error = _parameter_error(model, reference)
    fit_error = abs(fit - ref_fit)
    if param_error > ATOL_FLAT_REFERENCE:
        raise AssertionError(
            f"{kind}: flat-region parameter closure {param_error:.3e} > {ATOL_FLAT_REFERENCE:.3e}"
        )
    if fit_error > ATOL_FLAT_REFERENCE:
        raise AssertionError(
            f"{kind}: flat-region data-fit closure {fit_error:.3e} > {ATOL_FLAT_REFERENCE:.3e}"
        )
    return {
        "threshold": threshold,
        "group_norms": group_norms.tolist(),
        "reference_group_norms": ref_group_norms.tolist(),
        "parameter_error": param_error,
        "coef_error": coef_error,
        "intercept_error": intercept_error,
        "data_fit_error": fit_error,
    }


def _provenance(model, backend):
    observed = {
        "solver": str(getattr(model, "_selected_solver", "") or ""),
        "backend": str(getattr(model, "_selected_backend_name", "") or ""),
        "device": str(getattr(model, "_selected_backend_device", "") or ""),
    }
    if observed["solver"] != EXECUTED_SOLVER:
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
    expected_cv_calls = len(folds) * len(ALPHA_GRID) + 1
    unpenalized = _unpenalized_reference(X, y, weights)

    cpu = {}
    for kind in ("group_scad", "group_mcp"):
        direct, direct_calls, direct_locations = _with_group_solver_counter(
            lambda kind=kind: _fit(kind, X, y, weights, "cpu"),
            expected_backend="numpy",
        )
        flat_reference = _validate_flat_reference(
            kind, direct, unpenalized, X, y, weights
        )
        cv, cv_calls, cv_locations = _with_group_solver_counter(
            lambda kind=kind: _cv(kind, X, y, weights, folds, "cpu"),
            expected_backend="numpy",
            min_calls=expected_cv_calls,
        )
        cpu[kind] = {
            "direct": direct,
            "direct_calls": direct_calls,
            "direct_locations": direct_locations,
            "flat_reference": flat_reference,
            "objective": _objective(direct, X, y, weights),
            "cv": cv,
            "cv_calls": cv_calls,
            "cv_locations": cv_locations,
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
            direct, direct_calls, direct_locations = _with_group_solver_counter(
                lambda kind=kind: _fit(kind, Xb, yb, wb, device),
                expected_backend=backend,
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
                    "group_solver_calls": direct_calls,
                    "group_solver_input_locations": direct_locations,
                    "cpu_flat_reference": cpu[kind]["flat_reference"],
                    "parameter_error": param_error,
                    "coef_error": coef_error,
                    "intercept_error": intercept_error,
                    "objective": objective,
                    "cpu_objective": float(cpu[kind]["objective"]),
                    "objective_error": objective_error,
                }
            )

            cv, cv_calls, cv_locations = _with_group_solver_counter(
                lambda kind=kind: _cv(kind, Xb, yb, wb, folds, device),
                expected_backend=backend,
                min_calls=expected_cv_calls,
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
                    "group_solver_calls": cv_calls,
                    "group_solver_input_locations": cv_locations,
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
        "executed_solver": EXECUTED_SOLVER,
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
            "cupy_device_name": (
                cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                if isinstance(cp.cuda.runtime.getDeviceProperties(0)["name"], bytes)
                else str(cp.cuda.runtime.getDeviceProperties(0)["name"])
            ),
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
