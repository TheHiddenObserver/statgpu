#!/usr/bin/env python3
"""Physical CUDA case for weighted low-level Quantile FISTA-LLA refreshes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_lla_path
from statgpu.solvers._convergence import ConvergenceWarning


SCHEMA_VERSION = 2
Q = 0.35
ALPHA = 0.04
ATOL_PARAM = 1e-4
ATOL_OBJECTIVE = 8e-5
PARITY_MAX_LLA_PER_STEP = 12
PARITY_MAX_ITER = 1000
PARITY_TOL = 1e-4
PARITY_LLA_TOL = 1e-4
PROBE_MAX_ITER = 30
PROBE_TOL = 1e-30
PROBE_LLA_TOL = 1e-30


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_state():
    return _git("rev-parse", "HEAD"), not bool(_git("status", "--porcelain"))


def _backend_and_device(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        return "cupy", f"cuda:{int(value.device.id)}"
    if module.startswith("torch"):
        return "torch", str(value.device)
    return "numpy", "cpu"


class _RecordingQuantileLoss(QuantileLoss):
    def __init__(self, quantile=Q):
        super().__init__(quantile=quantile)
        self.weight_locations = []

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        self.weight_locations.append(
            None if sample_weight is None else _backend_and_device(sample_weight)
        )
        return super().lipschitz(
            X,
            coef,
            y=y,
            sample_weight=sample_weight,
        )


def _data(seed=166451, n=48):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float64)
    beta = np.array([0.72, -0.41, 0.27, 0.14], dtype=np.float64)
    y = (0.18 + X @ beta + rng.laplace(scale=0.17, size=n)).astype(np.float64)
    weights = np.linspace(0.35, 1.95, n, dtype=np.float64)
    rng.shuffle(weights)
    return X, y, weights


def _native_inputs(backend, X, y, weights, cp, torch):
    if backend == "cupy":
        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            cp.asarray(weights, dtype=cp.float64),
        )
    device = torch.device("cuda:0")
    return (
        torch.as_tensor(X, dtype=torch.float64, device=device),
        torch.as_tensor(y, dtype=torch.float64, device=device),
        torch.as_tensor(weights, dtype=torch.float64, device=device),
    )


def _run(X, y, weights):
    """Run the accepted numerical parity solve; warnings are gate failures."""
    loss = _RecordingQuantileLoss(Q)
    penalty = SCADPenalty(alpha=ALPHA, a=3.7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, intercept, n_iter = fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([ALPHA], dtype=np.float64),
            max_lla_per_step=PARITY_MAX_LLA_PER_STEP,
            max_iter=PARITY_MAX_ITER,
            lla_tol=PARITY_LLA_TOL,
            tol=PARITY_TOL,
            fit_intercept=True,
            sample_weight=weights,
        )
    if int(n_iter) < 1:
        raise AssertionError(f"scalar Quantile LLA reported invalid n_iter={n_iter}")
    if not loss.weight_locations:
        raise AssertionError("scalar Quantile LLA did not evaluate weighted step scale")
    if any(location is None for location in loss.weight_locations):
        raise AssertionError(
            f"scalar Quantile LLA lost analytic weights: {loss.weight_locations!r}"
        )
    return (
        np.asarray(_to_numpy(coef), dtype=np.float64).reshape(-1),
        float(intercept),
        int(n_iter),
        list(loss.weight_locations),
    )


def _run_periodic_refresh_probe(X, y, weights):
    """Force the periodic step-scale refresh without accepting its final iterate.

    The probe deliberately uses a tiny positive stopping tolerance so it reaches
    the iteration-20 refresh. Its expected target-exhaustion warning is captured
    locally; only the separately converged _run() result is used for numerical
    CPU/GPU parity.
    """
    loss = _RecordingQuantileLoss(Q)
    penalty = SCADPenalty(alpha=ALPHA, a=3.7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _coef, _intercept, n_iter = fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([ALPHA], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=PROBE_MAX_ITER,
            lla_tol=PROBE_LLA_TOL,
            tol=PROBE_TOL,
            fit_intercept=True,
            sample_weight=weights,
        )
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if not convergence_warnings:
        raise AssertionError(
            "scalar Quantile LLA periodic-refresh probe unexpectedly converged; "
            "the probe no longer guarantees the refresh branch is exercised"
        )
    if int(n_iter) < 21:
        raise AssertionError(
            f"scalar Quantile LLA did not reach periodic step refresh: n_iter={n_iter}"
        )
    if len(loss.weight_locations) < 2:
        raise AssertionError(
            "scalar Quantile LLA did not execute both initial and periodic step-scale calls"
        )
    if any(location is None for location in loss.weight_locations):
        raise AssertionError(
            f"scalar Quantile LLA periodic refresh lost analytic weights: "
            f"{loss.weight_locations!r}"
        )
    return int(n_iter), list(loss.weight_locations), convergence_warnings


def _objective(coef, intercept, X, y, weights):
    residual = y - (X @ coef + intercept)
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(np.average(pinball, weights=weights))
    penalty = SCADPenalty(alpha=ALPHA, a=3.7)
    return fit + float(penalty.value(coef))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_sha, source_clean = _source_state()
    if not source_clean:
        raise RuntimeError("scalar Quantile LLA physical case requires clean source")

    import cupy as cp
    import torch

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    cp.cuda.Device(0).use()
    torch.cuda.set_device(0)

    X, y, weights = _data()
    cpu_coef, cpu_intercept, cpu_iter, cpu_locations = _run(X, y, weights)
    cpu_probe_iter, cpu_probe_locations, cpu_probe_warnings = (
        _run_periodic_refresh_probe(X, y, weights)
    )
    cpu_objective = _objective(cpu_coef, cpu_intercept, X, y, weights)

    cases = []
    max_param_error = 0.0
    max_objective_error = 0.0
    for backend in ("cupy", "torch"):
        Xb, yb, wb = _native_inputs(backend, X, y, weights, cp, torch)
        coef, intercept, n_iter, locations = _run(Xb, yb, wb)
        probe_iter, probe_locations, probe_warnings = _run_periodic_refresh_probe(
            Xb, yb, wb
        )
        expected_location = (backend, "cuda:0")
        if any(tuple(location) != expected_location for location in locations):
            raise AssertionError(
                f"{backend}: scalar Quantile LLA converged-solve step-scale "
                f"device drift: {locations!r}"
            )
        if any(tuple(location) != expected_location for location in probe_locations):
            raise AssertionError(
                f"{backend}: scalar Quantile LLA periodic-refresh step-scale "
                f"device drift: {probe_locations!r}"
            )
        coef_error = float(np.max(np.abs(coef - cpu_coef)))
        intercept_error = abs(intercept - cpu_intercept)
        param_error = max(coef_error, intercept_error)
        objective = _objective(coef, intercept, X, y, weights)
        objective_error = abs(objective - cpu_objective)
        max_param_error = max(max_param_error, param_error)
        max_objective_error = max(max_objective_error, objective_error)
        if param_error > ATOL_PARAM:
            raise AssertionError(
                f"{backend}: scalar LLA parameter error {param_error:.3e} > {ATOL_PARAM:.3e}"
            )
        if objective_error > ATOL_OBJECTIVE:
            raise AssertionError(
                f"{backend}: scalar LLA objective error {objective_error:.3e} > {ATOL_OBJECTIVE:.3e}"
            )
        cases.append(
            {
                "backend": backend,
                "converged_step_scale_weight_locations": locations,
                "step_scale_weight_locations": probe_locations,
                "n_iter": n_iter,
                "periodic_refresh_probe_n_iter": probe_iter,
                "periodic_refresh_warning_count": len(probe_warnings),
                "coef_error": coef_error,
                "intercept_error": intercept_error,
                "parameter_error": param_error,
                "objective_error": objective_error,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "quantile": Q,
        "alpha": ALPHA,
        "cpu_n_iter": cpu_iter,
        "cpu_converged_step_scale_weight_locations": cpu_locations,
        "cpu_step_scale_weight_locations": cpu_probe_locations,
        "cpu_periodic_refresh_probe_n_iter": cpu_probe_iter,
        "cpu_periodic_refresh_warning_count": len(cpu_probe_warnings),
        "solver_controls": {
            "parity_max_lla_per_step": PARITY_MAX_LLA_PER_STEP,
            "parity_max_iter": PARITY_MAX_ITER,
            "parity_tol": PARITY_TOL,
            "parity_lla_tol": PARITY_LLA_TOL,
            "probe_max_iter": PROBE_MAX_ITER,
            "probe_tol": PROBE_TOL,
            "probe_lla_tol": PROBE_LLA_TOL,
        },
        "cases": cases,
        "max_errors": {
            "parameter": max_param_error,
            "objective": max_objective_error,
        },
        "tolerances": {
            "parameter": ATOL_PARAM,
            "objective": ATOL_OBJECTIVE,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
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
