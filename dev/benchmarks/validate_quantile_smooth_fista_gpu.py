#!/usr/bin/env python3
"""Physical CUDA acceptance for explicit smooth Quantile FISTA in PR #166.

This focused validator covers explicit ``solver='fista'`` on Quantile L2 and
no-penalty objectives and the standalone non-median batched-bootstrap path
repaired during the PR #166 review/fix loop. ``solver='auto'`` intentionally
remains IRLS and is validated by the existing PR #164 artifact.

The gate exercises both CuPy CUDA and Torch CUDA, including non-uniform analytic
weights, direct L2/no-penalty fits, strict weighted L2/L1 CV, a direct
non-uniform weighted L1 `cv_mode=True` async-FISTA case, a tau=0.20 standalone
bootstrap direction check, and a full public standalone bootstrap-inference fit.
It also replaces
``QuantileLoss.irls`` with a forbidden sentinel while the explicit-FISTA cases
run, so a passing result proves that the estimator/CV path did not silently
substitute IRLS.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model import PenalizedGLM_CV, QuantileRegression
from statgpu.linear_model.penalized import PenalizedQuantileRegression
from statgpu.losses import QuantileLoss
from statgpu.penalties import L1Penalty
from statgpu.solvers import fista_solver
from statgpu.solvers._convergence import ConvergenceWarning


SCHEMA_VERSION = 16
Q = 0.35
ATOL_OBJECTIVE = 2e-5
ATOL_CV_SCORE = 2e-5
ATOL_CV_L1_SCORE = 2e-4
ATOL_ASYNC_L1_OBJECTIVE = 5e-4
ATOL_BOOTSTRAP_OBJECTIVE = 2e-6
ATOL_BOOTSTRAP_INFERENCE = 2e-5
ASYNC_L1_ALPHA = 0.02
CV_ALPHA_GRID = np.asarray([0.03, 0.015], dtype=np.float64)

DIRECT_MAX_ITER = 5000
DIRECT_TOL = 1e-8
CV_MAX_ITER = 6000
CV_L2_TOL = 1e-7
CV_L1_TOL = 1e-5
ASYNC_MAX_ITER = 6000
ASYNC_TOL = 1e-7

BOOTSTRAP_Q = 0.20
BOOTSTRAP_N = 80
BOOTSTRAP_B = 12
BOOTSTRAP_SEED = 16692
BOOTSTRAP_DIRECTION_MAX_ITER = 400
BOOTSTRAP_PUBLIC_MAX_ITER = 1600
BOOTSTRAP_TOL = 1e-7


def _solver_controls():
    """Return JSON-serializable numerical controls owned by this validator."""
    return {
        "direct": {
            "max_iter": DIRECT_MAX_ITER,
            "tol": DIRECT_TOL,
        },
        "cv": {
            "max_iter": CV_MAX_ITER,
            "l2_tol": CV_L2_TOL,
            "l1_tol": CV_L1_TOL,
            "alpha_grid": CV_ALPHA_GRID.tolist(),
        },
        "async_weighted_l1": {
            "alpha": ASYNC_L1_ALPHA,
            "max_iter": ASYNC_MAX_ITER,
            "tol": ASYNC_TOL,
        },
        "bootstrap_direction": {
            "quantile": BOOTSTRAP_Q,
            "n": BOOTSTRAP_N,
            "n_bootstrap": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "max_iter": BOOTSTRAP_DIRECTION_MAX_ITER,
            "tol": BOOTSTRAP_TOL,
        },
        "bootstrap_public": {
            "quantile": BOOTSTRAP_Q,
            "n": BOOTSTRAP_N,
            "n_bootstrap": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "max_iter": BOOTSTRAP_PUBLIC_MAX_ITER,
            "tol": BOOTSTRAP_TOL,
            "fit_intercept_cases": [False, True],
        },
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_state():
    return _git("rev-parse", "HEAD"), not bool(_git("status", "--porcelain"))


def _require_gpu_backends():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("CuPy is required for PR166 smooth-FISTA validation") from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("Torch is required for PR166 smooth-FISTA validation") from exc

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")
    cp.cuda.Device(0).use()
    torch.cuda.set_device(0)
    return cp, torch


def _data(seed=16691, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.82, -0.37, 0.21], dtype=np.float64)[:p]
    y = (0.28 + X @ beta + rng.laplace(scale=0.20, size=n)).astype(np.float64)
    weights = np.linspace(0.55, 1.75, n, dtype=np.float64)
    rng.shuffle(weights)
    half = n // 2
    folds = [
        (np.arange(0, half), np.arange(half, n)),
        (np.arange(half, n), np.arange(0, half)),
    ]
    return X, y, weights, folds


def _async_weighted_data(seed=16693, n=96):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    X = np.column_stack(
        [
            base,
            base + 0.03 * rng.normal(size=n),
            0.5 * base + 0.20 * rng.normal(size=n),
        ]
    ).astype(np.float64)
    beta = np.asarray([0.75, -0.45, 0.30], dtype=np.float64)
    y = (
        X @ beta
        + rng.laplace(scale=0.18, size=n)
    ).astype(np.float64)
    weights = np.linspace(0.45, 1.85, n, dtype=np.float64)
    rng.shuffle(weights)
    gram = X.T @ (X * weights[:, None]) / float(np.sum(weights))
    spectral = float(np.linalg.eigvalsh(gram)[-1])
    max_diagonal = float(np.max(np.diag(gram)))
    ratio = spectral / max(max_diagonal, 1e-30)
    if ratio <= 1.5:
        raise AssertionError(
            f"async weighted fixture is not sufficiently correlated: ratio={ratio:.3f}"
        )
    return X, y, weights, ratio


def _host(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _objective(X, y, weights, coef, intercept, alpha):
    coef = np.asarray(coef, dtype=np.float64).ravel()
    residual = np.asarray(y, dtype=np.float64) - (
        np.asarray(X, dtype=np.float64) @ coef + float(intercept)
    )
    pinball = np.where(
        residual >= 0.0,
        Q * residual,
        (Q - 1.0) * residual,
    )
    data_fit = float(np.average(pinball, weights=np.asarray(weights, dtype=np.float64)))
    return data_fit + 0.5 * float(alpha) * float(np.dot(coef, coef))


def _l1_objective(X, y, weights, coef, alpha):
    coef = np.asarray(coef, dtype=np.float64).ravel()
    residual = np.asarray(y, dtype=np.float64) - (
        np.asarray(X, dtype=np.float64) @ coef
    )
    pinball = np.where(
        residual >= 0.0,
        Q * residual,
        (Q - 1.0) * residual,
    )
    data_fit = float(
        np.average(
            pinball,
            weights=np.asarray(weights, dtype=np.float64),
        )
    )
    return data_fit + float(alpha) * float(np.sum(np.abs(coef)))


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


def _standalone_bootstrap_direction_case(backend, cp, torch):
    """Exercise the backend-specific batched Quantile bootstrap at tau != 0.5."""
    X = np.ones((BOOTSTRAP_N, 1), dtype=np.float64)
    y = np.linspace(-4.0, 4.0, BOOTSTRAP_N, dtype=np.float64)
    if backend == "cupy":
        Xb = cp.asarray(X, dtype=cp.float64)
        yb = cp.asarray(y, dtype=cp.float64)
        device = "cuda:0"
    else:
        torch_device = torch.device("cuda:0")
        Xb = torch.as_tensor(X, dtype=torch.float64, device=torch_device)
        yb = torch.as_tensor(y, dtype=torch.float64, device=torch_device)
        device = str(torch_device)

    model = QuantileRegression(
        quantile=BOOTSTRAP_Q,
        fit_intercept=False,
        max_iter=BOOTSTRAP_DIRECTION_MAX_ITER,
        tol=BOOTSTRAP_TOL,
        n_bootstrap=BOOTSTRAP_B,
        random_state=BOOTSTRAP_SEED,
        device="cuda" if backend == "cupy" else "torch",
    )
    # The batched bootstrap solver only needs the already-fitted parameter
    # snapshot. A zero coefficient makes fitted residuals equal y; the public
    # bootstrap contract centers those residuals at their empirical target
    # quantile before resampling, so the oracle below uses the same centered
    # error distribution while remaining independent of the numerical solver.
    model.coef_ = np.zeros(1, dtype=np.float64)
    model.intercept_ = 0.0
    boot_params, params_native, design_native = model._compute_bootstrap_batched(Xb, yb)
    estimated = np.asarray(boot_params[:, 0], dtype=np.float64)

    if backend == "cupy":
        if not isinstance(design_native, cp.ndarray):
            raise AssertionError(
                "cupy/standalone/bootstrap: numerical design left the CuPy backend"
            )
        if int(design_native.device.id) != 0:
            raise AssertionError(
                "cupy/standalone/bootstrap: numerical design left CUDA device 0"
            )
        if not isinstance(params_native, cp.ndarray):
            raise AssertionError(
                "cupy/standalone/bootstrap: parameter snapshot left the CuPy backend"
            )
    else:
        if not torch.is_tensor(design_native) or not design_native.is_cuda:
            raise AssertionError(
                "torch/standalone/bootstrap: numerical design left Torch CUDA"
            )
        if str(design_native.device) != "cuda:0":
            raise AssertionError(
                "torch/standalone/bootstrap: numerical design left CUDA device 0"
            )
        if not torch.is_tensor(params_native) or not params_native.is_cuda:
            raise AssertionError(
                "torch/standalone/bootstrap: parameter snapshot left Torch CUDA"
            )

    solver_n_iter = int(model._bootstrap_n_iter_)
    if not 1 <= solver_n_iter <= int(model.max_iter):
        raise AssertionError(
            f"{backend}/standalone/bootstrap: invalid solver_n_iter={solver_n_iter}"
        )

    center_index = min(
        max(int(np.ceil(BOOTSTRAP_Q * BOOTSTRAP_N)) - 1, 0),
        BOOTSTRAP_N - 1,
    )
    residual_center = np.sort(y)[center_index]
    centered_residual = y - residual_center

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_batch = np.array(
        [
            centered_residual[
                rng.integers(0, BOOTSTRAP_N, size=BOOTSTRAP_N)
            ]
            for _ in range(BOOTSTRAP_B)
        ]
    )
    sorted_batch = np.sort(y_batch, axis=1)
    requested = sorted_batch[:, center_index]
    complementary_index = min(
        max(int(np.ceil((1.0 - BOOTSTRAP_Q) * BOOTSTRAP_N)) - 1, 0),
        BOOTSTRAP_N - 1,
    )
    complementary = sorted_batch[:, complementary_index]
    target_error = float(np.mean(np.abs(estimated - requested)))
    wrong_direction_error = float(np.mean(np.abs(estimated - complementary)))
    median_estimate = float(np.median(estimated))
    wrong_median = float(np.median(complementary))

    if not np.all(np.isfinite(estimated)):
        raise AssertionError(f"{backend}/standalone/bootstrap: non-finite parameters")
    if not target_error < wrong_direction_error:
        raise AssertionError(
            f"{backend}/standalone/bootstrap: requested-quantile error "
            f"{target_error:.3e} is not below complementary-quantile error "
            f"{wrong_direction_error:.3e}"
        )
    if not abs(median_estimate) < abs(wrong_median):
        raise AssertionError(
            f"{backend}/standalone/bootstrap: median estimate {median_estimate:.3e} "
            "is not closer to the centered requested quantile than the "
            f"complementary oracle median {wrong_median:.3e}"
        )
    if model._bootstrap_residual_centering_ != "empirical_tau_quantile":
        raise AssertionError(
            f"{backend}/standalone/bootstrap: residual centering provenance "
            f"{model._bootstrap_residual_centering_!r} != "
            "'empirical_tau_quantile'"
        )

    return {
        "name": f"{backend}/standalone/bootstrap/q{BOOTSTRAP_Q:.2f}",
        "backend": backend,
        "device": device,
        "quantile": BOOTSTRAP_Q,
        "n_bootstrap": BOOTSTRAP_B,
        "solver": "batched_quantile_irls",
        "solver_n_iter": solver_n_iter,
        "residual_centering": model._bootstrap_residual_centering_,
        "target_error": target_error,
        "complementary_quantile_error": wrong_direction_error,
        "median_estimate": median_estimate,
    }


def _standalone_bootstrap_multifeature_parity_case(
    backend,
    cp,
    torch,
):
    """Compare multifeature bootstrap IRLS objectives with the CPU path."""
    tau = 0.30
    n = 64
    B = 8
    seed = 16697
    rng = np.random.default_rng(16696)
    X = rng.normal(size=(n, 2)).astype(np.float64)
    coef0 = np.asarray([0.72, -0.31], dtype=np.float64)
    intercept0 = 0.27
    residual = (
        np.linspace(-1.15, 1.05, n, dtype=np.float64)
        + 0.025 * rng.normal(size=n)
    )
    y = intercept0 + X @ coef0 + residual

    def solve(X_arg, y_arg, *, device):
        model = QuantileRegression(
            quantile=tau,
            fit_intercept=True,
            max_iter=1200,
            tol=1e-8,
            n_bootstrap=B,
            random_state=seed,
            device=device,
        )
        model.coef_ = coef0.copy()
        model.intercept_ = float(intercept0)
        params, _, _ = model._compute_bootstrap_batched(X_arg, y_arg)
        return np.asarray(params, dtype=np.float64), model

    cpu_params, cpu_model = solve(X, y, device="cpu")
    if backend == "cupy":
        Xb = cp.asarray(X, dtype=cp.float64)
        yb = cp.asarray(y, dtype=cp.float64)
        device = "cuda"
        expected_device = "cuda:0"
    else:
        torch_device = torch.device("cuda:0")
        Xb = torch.as_tensor(X, dtype=torch.float64, device=torch_device)
        yb = torch.as_tensor(y, dtype=torch.float64, device=torch_device)
        device = "torch"
        expected_device = str(torch_device)
    gpu_params, gpu_model = solve(Xb, yb, device=device)

    center_index = min(max(int(np.ceil(tau * n)) - 1, 0), n - 1)
    centered_residual = residual - np.sort(residual)[center_index]
    eta = intercept0 + X @ coef0
    schedule_rng = np.random.default_rng(seed)
    schedule = np.stack(
        [schedule_rng.integers(0, n, size=n, dtype=np.int64) for _ in range(B)]
    )
    y_batch = np.asarray(
        [eta + centered_residual[schedule[draw]] for draw in range(B)],
        dtype=np.float64,
    )

    def objectives(params):
        values = []
        for draw in range(B):
            intercept = float(params[draw, 0])
            coef = np.asarray(params[draw, 1:], dtype=np.float64)
            u = y_batch[draw] - (X @ coef + intercept)
            values.append(
                float(
                    np.mean(
                        np.where(
                            u >= 0.0,
                            tau * u,
                            (tau - 1.0) * u,
                        )
                    )
                )
            )
        return np.asarray(values, dtype=np.float64)

    cpu_objective = objectives(cpu_params)
    gpu_objective = objectives(gpu_params)
    objective_error = float(np.max(np.abs(gpu_objective - cpu_objective)))
    if objective_error > ATOL_BOOTSTRAP_OBJECTIVE:
        raise AssertionError(
            f"{backend}/standalone/bootstrap-multifeature: objective parity "
            f"error {objective_error:.3e} > {ATOL_BOOTSTRAP_OBJECTIVE:.3e}"
        )
    if gpu_model._bootstrap_schedule_sha256_ != cpu_model._bootstrap_schedule_sha256_:
        raise AssertionError(
            f"{backend}/standalone/bootstrap-multifeature: schedule identity drifted"
        )
    if not 1 <= int(gpu_model._bootstrap_n_iter_) <= int(gpu_model.max_iter):
        raise AssertionError(
            f"{backend}/standalone/bootstrap-multifeature: invalid iteration count"
        )

    return {
        "name": f"{backend}/standalone/bootstrap-multifeature/q{tau:.2f}",
        "backend": backend,
        "device": expected_device,
        "quantile": tau,
        "fit_intercept": True,
        "n_bootstrap": B,
        "solver": "batched_quantile_irls",
        "cpu_solver_n_iter": int(cpu_model._bootstrap_n_iter_),
        "solver_n_iter": int(gpu_model._bootstrap_n_iter_),
        "objective_error_vs_cpu": objective_error,
        "tolerance": ATOL_BOOTSTRAP_OBJECTIVE,
        "resampling_schedule_sha256": gpu_model._bootstrap_schedule_sha256_,
    }


def _standalone_bootstrap_public_case(
    backend,
    cp,
    torch,
    *,
    fit_intercept,
):
    """Exercise public standalone bootstrap inference with both intercept modes."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + 31)
    X = rng.normal(size=(BOOTSTRAP_N, 1)).astype(np.float64)
    y = (
        0.35
        + 0.75 * X[:, 0]
        + rng.laplace(scale=0.22, size=BOOTSTRAP_N)
    ).astype(np.float64)
    if backend == "cupy":
        Xb = cp.asarray(X, dtype=cp.float64)
        yb = cp.asarray(y, dtype=cp.float64)
        device_request = "cuda"
        expected_device = "cuda:0"
    else:
        torch_device = torch.device("cuda:0")
        Xb = torch.as_tensor(X, dtype=torch.float64, device=torch_device)
        yb = torch.as_tensor(y, dtype=torch.float64, device=torch_device)
        device_request = "torch"
        expected_device = str(torch_device)

    common = dict(
        quantile=BOOTSTRAP_Q,
        fit_intercept=bool(fit_intercept),
        max_iter=BOOTSTRAP_PUBLIC_MAX_ITER,
        tol=BOOTSTRAP_TOL,
        compute_inference=True,
        inference_method="bootstrap",
        n_bootstrap=BOOTSTRAP_B,
        random_state=BOOTSTRAP_SEED,
    )
    cpu_model = QuantileRegression(device="cpu", **common).fit(X, y)
    model = QuantileRegression(device=device_request, **common).fit(Xb, yb)

    if not bool(getattr(model, "_fitted", False)):
        raise AssertionError(f"{backend}/standalone/public: model was not fitted")
    if model._selected_backend_name != backend:
        raise AssertionError(
            f"{backend}/standalone/public: backend provenance "
            f"{model._selected_backend_name!r} != {backend!r}"
        )
    if model._selected_backend_device != expected_device:
        raise AssertionError(
            f"{backend}/standalone/public: device provenance "
            f"{model._selected_backend_device!r} != {expected_device!r}"
        )
    if not 1 <= int(model.n_iter_) <= int(model.max_iter):
        raise AssertionError(
            f"{backend}/standalone/public: invalid point-fit n_iter={model.n_iter_}"
        )

    result = model._inference_result
    if result is None or result.method != "bootstrap":
        raise AssertionError(
            f"{backend}/standalone/public: bootstrap result was not published"
        )
    metadata = dict(result.metadata)
    expected_meta = {
        "solver": "batched_quantile_irls",
        "numerical_backend": backend,
        "numerical_device": expected_device,
        "reporting_backend": "numpy",
        "response_construction": "backend_native",
        "residual_centering": "empirical_tau_quantile",
        "resampling_schedule": "numpy_generator_control_plane",
    }
    for key, expected in expected_meta.items():
        if metadata.get(key) != expected:
            raise AssertionError(
                f"{backend}/standalone/public: metadata[{key!r}]="
                f"{metadata.get(key)!r} != {expected!r}"
            )
    schedule_hash = str(metadata.get("resampling_schedule_sha256", ""))
    if len(schedule_hash) != 64:
        raise AssertionError(
            f"{backend}/standalone/public: invalid schedule hash {schedule_hash!r}"
        )
    try:
        int(schedule_hash, 16)
    except ValueError as exc:
        raise AssertionError(
            f"{backend}/standalone/public: non-hex schedule hash {schedule_hash!r}"
        ) from exc

    solver_n_iter = int(metadata.get("solver_n_iter", 0))
    if not 1 <= solver_n_iter <= int(model.max_iter):
        raise AssertionError(
            f"{backend}/standalone/public: invalid bootstrap solver_n_iter="
            f"{solver_n_iter}"
        )

    if not np.isfinite(float(model.intercept_)):
        raise AssertionError(
            f"{backend}/standalone/public: non-finite intercept"
        )

    snapshots = {
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "bse": np.asarray(result.bse, dtype=np.float64),
        "pvalues": np.asarray(result.pvalues, dtype=np.float64),
        "conf_int": np.asarray(result.conf_int, dtype=np.float64),
    }
    cpu_result = cpu_model._inference_result
    cpu_snapshots = {
        "coef": np.asarray(cpu_model.coef_, dtype=np.float64),
        "bse": np.asarray(cpu_result.bse, dtype=np.float64),
        "pvalues": np.asarray(cpu_result.pvalues, dtype=np.float64),
        "conf_int": np.asarray(cpu_result.conf_int, dtype=np.float64),
    }
    for name, values in snapshots.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(
                f"{backend}/standalone/public: non-finite {name}"
            )
    if np.any((snapshots["pvalues"] < 0.0) | (snapshots["pvalues"] > 1.0)):
        raise AssertionError(
            f"{backend}/standalone/public: p-values outside [0, 1]"
        )

    inference_errors = {
        name: float(np.max(np.abs(snapshots[name] - cpu_snapshots[name])))
        for name in ("coef", "bse", "pvalues", "conf_int")
    }
    inference_errors["intercept"] = abs(
        float(model.intercept_) - float(cpu_model.intercept_)
    )
    max_inference_error = max(inference_errors.values())
    if max_inference_error > ATOL_BOOTSTRAP_INFERENCE:
        raise AssertionError(
            f"{backend}/standalone/public: CPU parity error "
            f"{max_inference_error:.3e} > {ATOL_BOOTSTRAP_INFERENCE:.3e}; "
            f"details={inference_errors!r}"
        )
    if (
        model._bootstrap_schedule_sha256_
        != cpu_model._bootstrap_schedule_sha256_
    ):
        raise AssertionError(
            f"{backend}/standalone/public: CPU/GPU bootstrap schedule drifted"
        )

    return {
        "name": (
            f"{backend}/standalone/public-bootstrap/"
            f"{'intercept' if fit_intercept else 'no-intercept'}/"
            f"q{BOOTSTRAP_Q:.2f}"
        ),
        "backend": backend,
        "fit_intercept": bool(fit_intercept),
        "device": expected_device,
        "quantile": BOOTSTRAP_Q,
        "point_solver": "fista",
        "point_n_iter": int(model.n_iter_),
        "inference_method": str(result.method),
        "bootstrap_solver": str(metadata["solver"]),
        "bootstrap_solver_n_iter": solver_n_iter,
        "residual_centering": str(metadata["residual_centering"]),
        "resampling_schedule_sha256": schedule_hash,
        "inference_errors_vs_cpu": inference_errors,
        "max_inference_error_vs_cpu": max_inference_error,
        "inference_tolerance": ATOL_BOOTSTRAP_INFERENCE,
    }


def _async_weighted_l1_case(
    backend,
    cp,
    torch,
    X,
    y,
    weights,
    cpu_objective,
    cpu_n_iter,
    spectral_ratio,
):
    """Exercise the actual non-smooth GPU async FISTA branch."""
    Xb, yb, wb, _ = _native_inputs(
        backend,
        X,
        y,
        weights,
        cp,
        torch,
    )
    loss = QuantileLoss(quantile=Q)
    penalty = L1Penalty(alpha=ASYNC_L1_ALPHA)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, n_iter = fista_solver(
            loss,
            penalty,
            Xb,
            yb,
            max_iter=ASYNC_MAX_ITER,
            tol=ASYNC_TOL,
            sample_weight=wb,
            cv_mode=True,
        )
    coef_host = _host(coef).ravel()
    if not np.all(np.isfinite(coef_host)):
        raise AssertionError(
            f"{backend}/async-weighted-l1: non-finite coefficients"
        )
    objective = _l1_objective(
        X,
        y,
        weights,
        coef_host,
        ASYNC_L1_ALPHA,
    )
    error = abs(objective - cpu_objective)
    if error > ATOL_ASYNC_L1_OBJECTIVE:
        raise AssertionError(
            f"{backend}/async-weighted-l1: objective error "
            f"{error:.3e} > {ATOL_ASYNC_L1_OBJECTIVE:.3e}"
        )

    if backend == "cupy":
        if not isinstance(coef, cp.ndarray):
            raise AssertionError(
                "cupy/async-weighted-l1: coefficient left CuPy"
            )
        device = f"cuda:{int(coef.device.id)}"
    else:
        if not torch.is_tensor(coef) or not coef.is_cuda:
            raise AssertionError(
                "torch/async-weighted-l1: coefficient left Torch CUDA"
            )
        device = str(coef.device)
    if device != "cuda:0":
        raise AssertionError(
            f"{backend}/async-weighted-l1: device drifted to {device!r}"
        )

    return {
        "name": f"{backend}/async-weighted-l1/cv-mode",
        "backend": backend,
        "device": device,
        "solver": "fista",
        "cv_mode": True,
        "n_iter": int(n_iter),
        "alpha": ASYNC_L1_ALPHA,
        "weighted_gram_spectral_to_maxdiag_ratio": spectral_ratio,
        "cpu_objective": cpu_objective,
        "cpu_n_iter": int(cpu_n_iter),
        "objective": objective,
        "objective_error": error,
    }


def _provenance(model, backend):
    expected_backend = backend
    solver = str(getattr(model, "_selected_solver", "") or "")
    selected_backend = str(getattr(model, "_selected_backend_name", "") or "")
    device = str(getattr(model, "_selected_backend_device", "") or "")
    if solver != "fista":
        raise AssertionError(f"executed solver drifted: {solver!r} != 'fista'")
    if selected_backend != expected_backend:
        raise AssertionError(
            f"backend drifted: {selected_backend!r} != {expected_backend!r}"
        )
    if device != "cuda:0":
        raise AssertionError(f"device drifted: {device!r} != 'cuda:0'")
    return {"solver": solver, "backend": selected_backend, "device": device}


def _direct(X, y, weights, *, penalty, alpha, device):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = PenalizedQuantileRegression(
            quantile=Q,
            penalty=penalty,
            alpha=alpha,
            solver="fista",
            device=device,
            max_iter=DIRECT_MAX_ITER,
            tol=DIRECT_TOL,
        ).fit(X, y, sample_weight=weights)
    coef = _host(model.coef_).ravel()
    intercept = float(model.intercept_)
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        raise AssertionError(f"direct {penalty}: non-finite parameters")
    return model, coef, intercept


def _cv(X, y, weights, folds, *, device, penalty="l2"):
    # Quantile FISTA is nonsmooth even with an L2 penalty. The half-sample CV
    # folds use 1e-7 so the converged CPU oracle stops before the strict proximal
    # line search reaches the floating-point floor at a pinball kink. L1 keeps
    # the looser 1e-5 control. These are solver stopping controls; the separate
    # CV-score tolerances below measure cross-backend prediction-score parity.
    solver_tol = CV_L1_TOL if penalty == "l1" else CV_L2_TOL
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return PenalizedGLM_CV(
            loss="quantile",
            loss_kwargs={"quantile": Q},
            penalty=penalty,
            alpha_grid=CV_ALPHA_GRID,
            cv=2,
            cv_splits=folds,
            random_state=166,
            solver="fista",
            device=device,
            cv_strategy="strict",
            max_iter=CV_MAX_ITER,
            tol=solver_tol,
        ).fit(X, y, sample_weight=weights)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dev/reviews/pr166_quantile_smooth_fista_gpu.json",
    )
    args = parser.parse_args()

    source_sha, source_clean = _source_state()
    cp, torch = _require_gpu_backends()
    X, y, weights, folds = _data()

    # CPU references use the same explicit public route. Hosted tests separately
    # align L2 objective with IRLS and no-penalty objective with sklearn HiGHS.
    cpu_l2, cpu_l2_coef, cpu_l2_intercept = _direct(
        X, y, weights, penalty="l2", alpha=0.02, device="cpu"
    )
    cpu_none, cpu_none_coef, cpu_none_intercept = _direct(
        X, y, weights, penalty="none", alpha=0.0, device="cpu"
    )
    cpu_cv = {
        penalty: _cv(
            X,
            y,
            weights,
            folds,
            device="cpu",
            penalty=penalty,
        )
        for penalty in ("l2", "l1")
    }
    cpu_cv_scores = {
        penalty: np.asarray(
            model.cv_results_["all_scores"],
            dtype=np.float64,
        )
        for penalty, model in cpu_cv.items()
    }
    async_X, async_y, async_weights, async_spectral_ratio = (
        _async_weighted_data()
    )
    # The GPU parity target must itself be a converged solve. A finite
    # coefficient vector from an exhausted/line-search-failed CPU run is not a
    # valid numerical oracle merely because its objective is finite.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        async_cpu_coef, async_cpu_iter = fista_solver(
            QuantileLoss(quantile=Q),
            L1Penalty(alpha=ASYNC_L1_ALPHA),
            async_X,
            async_y,
            max_iter=ASYNC_MAX_ITER,
            tol=ASYNC_TOL,
            sample_weight=async_weights,
            cv_mode=False,
        )
    async_cpu_objective = _l1_objective(
        async_X,
        async_y,
        async_weights,
        async_cpu_coef,
        ASYNC_L1_ALPHA,
    )

    def forbidden_irls(*_args, **_kwargs):
        raise AssertionError("explicit Quantile FISTA physically executed IRLS")

    original_irls = QuantileLoss.irls
    QuantileLoss.irls = forbidden_irls
    cases = []
    max_objective_error = 0.0
    max_cv_score_error = 0.0
    try:
        for backend in ("cupy", "torch"):
            Xb, yb, wb, device = _native_inputs(backend, X, y, weights, cp, torch)
            cases.append(_standalone_bootstrap_direction_case(backend, cp, torch))
            cases.append(
                _standalone_bootstrap_multifeature_parity_case(
                    backend,
                    cp,
                    torch,
                )
            )
            for fit_intercept in (False, True):
                cases.append(
                    _standalone_bootstrap_public_case(
                        backend,
                        cp,
                        torch,
                        fit_intercept=fit_intercept,
                    )
                )
            cases.append(
                _async_weighted_l1_case(
                    backend,
                    cp,
                    torch,
                    async_X,
                    async_y,
                    async_weights,
                    async_cpu_objective,
                    async_cpu_iter,
                    async_spectral_ratio,
                )
            )

            for penalty, alpha, cpu_coef, cpu_intercept in (
                ("l2", 0.02, cpu_l2_coef, cpu_l2_intercept),
                ("none", 0.0, cpu_none_coef, cpu_none_intercept),
            ):
                model, coef, intercept = _direct(
                    Xb, yb, wb, penalty=penalty, alpha=alpha, device=device
                )
                gpu_obj = _objective(X, y, weights, coef, intercept, alpha)
                cpu_obj = _objective(
                    X, y, weights, cpu_coef, cpu_intercept, alpha
                )
                objective_error = abs(gpu_obj - cpu_obj)
                max_objective_error = max(max_objective_error, objective_error)
                if objective_error > ATOL_OBJECTIVE:
                    raise AssertionError(
                        f"{backend}/direct/{penalty}: objective error "
                        f"{objective_error:.3e} > {ATOL_OBJECTIVE:.3e}"
                    )
                cases.append(
                    {
                        "name": f"{backend}/direct/{penalty}",
                        "provenance": _provenance(model, backend),
                        "cpu_objective": cpu_obj,
                        "objective": gpu_obj,
                        "objective_error": objective_error,
                    }
                )

            for cv_penalty in ("l2", "l1"):
                cv = _cv(
                    Xb,
                    yb,
                    wb,
                    folds,
                    device=device,
                    penalty=cv_penalty,
                )
                cpu_reference = cpu_cv[cv_penalty]
                if float(cv.alpha_) != float(cpu_reference.alpha_):
                    raise AssertionError(
                        f"{backend}/cv/{cv_penalty}: selected alpha "
                        f"{cv.alpha_!r} != CPU {cpu_reference.alpha_!r}"
                    )
                scores = np.asarray(
                    cv.cv_results_["all_scores"],
                    dtype=np.float64,
                )
                score_error = float(
                    np.max(
                        np.abs(
                            scores - cpu_cv_scores[cv_penalty]
                        )
                    )
                )
                max_cv_score_error = max(
                    max_cv_score_error,
                    score_error,
                )
                tolerance = (
                    ATOL_CV_L1_SCORE
                    if cv_penalty == "l1"
                    else ATOL_CV_SCORE
                )
                if score_error > tolerance:
                    raise AssertionError(
                        f"{backend}/cv/{cv_penalty}: score error "
                        f"{score_error:.3e} > {tolerance:.3e}"
                    )
                cases.append(
                    {
                        "name": f"{backend}/cv/{cv_penalty}",
                        "provenance": _provenance(
                            cv.estimator_,
                            backend,
                        ),
                        "selected_alpha": float(cv.alpha_),
                        "cpu_selected_alpha": float(
                            cpu_reference.alpha_
                        ),
                        "scores": scores.tolist(),
                        "score_error": score_error,
                        "tolerance": tolerance,
                    }
                )
    finally:
        QuantileLoss.irls = original_irls

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "quantile": Q,
        "solver_controls": _solver_controls(),
        "cases": cases,
        "max_errors": {
            "direct_objective": max_objective_error,
            "cv_score": max_cv_score_error,
        },
        "tolerances": {
            "direct_objective": ATOL_OBJECTIVE,
            "cv_score": ATOL_CV_SCORE,
            "cv_l1_score": ATOL_CV_L1_SCORE,
            "async_weighted_l1_objective": ATOL_ASYNC_L1_OBJECTIVE,
            "bootstrap_objective_vs_cpu": ATOL_BOOTSTRAP_OBJECTIVE,
            "bootstrap_inference_vs_cpu": ATOL_BOOTSTRAP_INFERENCE,
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
