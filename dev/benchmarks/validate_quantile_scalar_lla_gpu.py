#!/usr/bin/env python3
"""Physical CUDA acceptance for low-level scalar Quantile LLA ownership."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import _fista as fista_module
from statgpu.solvers import fista_lla_path


SCHEMA_VERSION = 1
Q = 0.35
ALPHA = 0.04
ATOL_PARAM = 1e-4
ATOL_OBJECTIVE = 8e-5


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
        self.irls_locations = []

    def irls(self, X, y, *args, sample_weight=None, **kwargs):
        self.irls_locations.append(
            {
                "X": _backend_and_device(X),
                "y": _backend_and_device(y),
                "sample_weight": (
                    None
                    if sample_weight is None
                    else _backend_and_device(sample_weight)
                ),
            }
        )
        return super().irls(
            X,
            y,
            *args,
            sample_weight=sample_weight,
            **kwargs,
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


def _objective(coef, intercept, X, y, weights):
    residual = y - (X @ coef + intercept)
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(np.average(pinball, weights=weights))
    penalty = SCADPenalty(alpha=ALPHA, a=3.7)
    return fit + float(penalty.value(coef))


def _run_active(X, y, weights, expected_backend):
    """Start at zero so the SCAD LLA surrogate is active and must use WLS FISTA."""
    loss = _RecordingQuantileLoss(Q)
    penalty = SCADPenalty(alpha=ALPHA, a=3.7)
    original = fista_module.fista_solver
    calls = []

    def counted(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        location = {
            "loss": str(getattr(loss_arg, "name", "")),
            "penalty": str(getattr(penalty_arg, "name", "")),
            "X": _backend_and_device(X_arg),
            "y": _backend_and_device(y_arg),
            "sample_weight": kwargs.get("sample_weight"),
        }
        calls.append(location)
        if location["loss"] != "squared_error":
            raise AssertionError(
                f"scalar Quantile LLA inner loss drifted: {location['loss']!r}"
            )
        if location["penalty"] != "adaptive_l1":
            raise AssertionError(
                f"scalar Quantile LLA inner penalty drifted: {location['penalty']!r}"
            )
        if location["X"][0] != expected_backend or location["y"][0] != expected_backend:
            raise AssertionError(
                f"scalar Quantile WLS backend drifted: {location!r}"
            )
        if expected_backend in ("cupy", "torch"):
            if location["X"][1] != "cuda:0" or location["y"][1] != "cuda:0":
                raise AssertionError(
                    f"scalar Quantile WLS device drifted: {location!r}"
                )
        if kwargs.get("sample_weight") is not None:
            raise AssertionError(
                "IRLS analytic weights must be absorbed into the WLS design, "
                "not passed a second time to squared-error FISTA"
            )
        return original(loss_arg, penalty_arg, X_arg, y_arg, **kwargs)

    fista_module.fista_solver = counted
    try:
        coef, intercept, n_iter = fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([ALPHA], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=60,
            lla_tol=1e-7,
            tol=1e-6,
            fit_intercept=True,
            sample_weight=weights,
        )
    finally:
        fista_module.fista_solver = original

    if not calls:
        raise AssertionError("active scalar Quantile LLA did not execute WLS FISTA")
    return (
        np.asarray(_to_numpy(coef), dtype=np.float64).reshape(-1),
        float(intercept),
        int(n_iter),
        calls,
    )


def _run_flat(X, y, weights, expected_backend):
    """Large initial coefficients force a zero SCAD derivative and exact IRLS closure."""
    loss = _RecordingQuantileLoss(Q)
    penalty = SCADPenalty(alpha=ALPHA, a=3.7)
    # Every feature starts well beyond a*alpha, so the single LLA surrogate is flat.
    init = np.full(int(X.shape[1]), 1.0, dtype=np.float64)
    original = fista_module.fista_solver

    def forbidden(*args, **kwargs):
        raise AssertionError("flat scalar Quantile LLA must close through IRLS, not FISTA")

    fista_module.fista_solver = forbidden
    try:
        coef, intercept, n_iter = fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([ALPHA], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=300,
            lla_tol=1e-7,
            tol=1e-8,
            fit_intercept=True,
            sample_weight=weights,
            init_coef=init,
            init_intercept=0.0,
        )
    finally:
        fista_module.fista_solver = original

    if len(loss.irls_locations) != 1:
        raise AssertionError(
            f"flat scalar Quantile LLA expected one IRLS solve, got {loss.irls_locations!r}"
        )
    location = loss.irls_locations[0]
    if location["X"][0] != expected_backend or location["y"][0] != expected_backend:
        raise AssertionError(f"scalar Quantile IRLS backend drifted: {location!r}")
    if location["sample_weight"] is None:
        raise AssertionError("flat scalar Quantile IRLS lost analytic weights")
    if location["sample_weight"][0] != expected_backend:
        raise AssertionError(f"scalar Quantile IRLS weight backend drifted: {location!r}")
    if expected_backend in ("cupy", "torch"):
        if (
            location["X"][1] != "cuda:0"
            or location["y"][1] != "cuda:0"
            or location["sample_weight"][1] != "cuda:0"
        ):
            raise AssertionError(f"scalar Quantile IRLS device drifted: {location!r}")
    return (
        np.asarray(_to_numpy(coef), dtype=np.float64).reshape(-1),
        float(intercept),
        int(n_iter),
        location,
    )


def _param_error(actual, reference):
    coef_a, int_a = actual[0], actual[1]
    coef_r, int_r = reference[0], reference[1]
    coef_error = float(np.max(np.abs(coef_a - coef_r)))
    intercept_error = abs(float(int_a) - float(int_r))
    return max(coef_error, intercept_error), coef_error, intercept_error


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
    cpu_active = _run_active(X, y, weights, "numpy")
    cpu_flat = _run_flat(X, y, weights, "numpy")
    cpu_active_obj = _objective(cpu_active[0], cpu_active[1], X, y, weights)
    cpu_flat_obj = _objective(cpu_flat[0], cpu_flat[1], X, y, weights)

    cases = []
    max_param_error = 0.0
    max_objective_error = 0.0
    for backend in ("cupy", "torch"):
        Xb, yb, wb = _native_inputs(backend, X, y, weights, cp, torch)
        for mode, runner, reference, cpu_obj in (
            ("active", _run_active, cpu_active, cpu_active_obj),
            ("flat", _run_flat, cpu_flat, cpu_flat_obj),
        ):
            result = runner(Xb, yb, wb, backend)
            param_error, coef_error, intercept_error = _param_error(result, reference)
            objective = _objective(result[0], result[1], X, y, weights)
            objective_error = abs(objective - cpu_obj)
            max_param_error = max(max_param_error, param_error)
            max_objective_error = max(max_objective_error, objective_error)
            if param_error > ATOL_PARAM:
                raise AssertionError(
                    f"{backend}/{mode}: scalar LLA parameter error "
                    f"{param_error:.3e} > {ATOL_PARAM:.3e}"
                )
            if objective_error > ATOL_OBJECTIVE:
                raise AssertionError(
                    f"{backend}/{mode}: scalar LLA objective error "
                    f"{objective_error:.3e} > {ATOL_OBJECTIVE:.3e}"
                )
            cases.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "n_iter": int(result[2]),
                    "coef_error": coef_error,
                    "intercept_error": intercept_error,
                    "parameter_error": param_error,
                    "objective_error": objective_error,
                    "inner_contract": result[3],
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "quantile": Q,
        "alpha": ALPHA,
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
