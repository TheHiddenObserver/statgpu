#!/usr/bin/env python3
"""Physical CUDA cases for low-level Quantile LLA accuracy and convergence.

The public scalar Quantile SCAD/MCP low-level route delegates to the dedicated
Proximal IRLS-LLA engine. This validator certifies, on NumPy/CuPy/Torch:

- a nontrivial converged fixture with an active SCAD interior fixed point
  reaches an independent HiGHS LP fixed-point oracle within
  ``ATOL_FIXED_POINT``;
- CPU/GPU parameter and objective parity stay within ``ATOL_PARAM`` /
  ``ATOL_OBJECTIVE``;
- the exact zero fixed point still converges cleanly;
- budget exhaustion reports exactly one dedicated ``ConvergenceWarning``.

The dedicated engine returns host NumPy by public contract, so execution
provenance is asserted on the backend-native inputs it executes on. Silent host
fallback inside the engine is not detectable from this validator and remains a
residual evidence limitation.
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
from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_lla_path
from statgpu.solvers._convergence import ConvergenceWarning


SCHEMA_VERSION = 3
Q = 0.35
ALPHA = 0.5
SCAD_A = 3.7
ATOL_PARAM = 1e-4
ATOL_OBJECTIVE = 8e-5
ATOL_FIXED_POINT = 1e-8
TARGET_MAX_LLA_PER_STEP = 8
TARGET_MAX_ITER = 20000
TARGET_TOL = 1e-10
TARGET_LLA_TOL = 1e-10
EXHAUSTION_MAX_LLA_PER_STEP = 1
EXHAUSTION_MAX_ITER = 1
EXHAUSTION_TOL = 1e-12
EXHAUSTION_LLA_TOL = 1e-12
EXHAUSTION_WARNING = "Quantile Proximal IRLS-CD target reached"


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


def _converged_data():
    """Return an exact weighted fixed-point case that must converge cleanly.

    QuantileLoss deliberately chooses the -tau subgradient at zero residual, so
    y=0 alone does not make beta=0 stationary for an arbitrary design. Use
    paired +/- coordinate rows with equal weights inside each pair. Their
    weighted column sums cancel exactly while the weights remain non-uniform
    across coordinates, making beta=0 a genuine Quantile+SCAD fixed point.
    """
    eye = np.eye(4, dtype=np.float64)
    rows = []
    weights = []
    pair_weights = np.asarray([0.4, 0.9, 1.4, 1.9], dtype=np.float64)
    for row, weight in zip(eye, pair_weights):
        rows.extend([row, -row])
        weights.extend([weight, weight])
    X = np.asarray(rows, dtype=np.float64)
    y = np.zeros(X.shape[0], dtype=np.float64)
    return X, y, np.asarray(weights, dtype=np.float64)


def _nontrivial_data():
    """Return a weighted nontrivial fixture with a well-posed SCAD optimum."""
    rng = np.random.default_rng(4242)
    X = np.linspace(0.5, 1.5, 48).reshape(-1, 1)
    y = (1.0 + 0.8 * X[:, 0] + rng.laplace(scale=0.3, size=48)).astype(
        np.float64
    )
    weights = np.linspace(0.4, 1.7, 48, dtype=np.float64)
    rng.shuffle(weights)
    return X, y, weights


def _exhaustion_data():
    """Return a weighted fixture whose tiny budget must exhaust the solver."""
    rng = np.random.default_rng(166500)
    q_matrix, _ = np.linalg.qr(rng.normal(size=(48, 4)))
    X = (q_matrix * np.sqrt(48)).astype(np.float64)
    beta = np.asarray([0.9, -0.7, 0.0, 0.35], dtype=np.float64)
    y = (X @ beta + rng.laplace(scale=0.2, size=48)).astype(np.float64)
    weights = np.linspace(0.35, 1.95, 48, dtype=np.float64)
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


def _require_finite(values, label):
    """Fail closed before any NaN-blind comparison can mask a bad result."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{label}: non-finite values {array!r}")


def _solve(
    X,
    y,
    weights,
    *,
    max_lla_per_step,
    max_iter,
    tol,
    lla_tol,
    label,
):
    """Solve through the public low-level route; warnings are gate failures.

    The dedicated engine returns host NumPy by contract, so provenance is
    recorded from the backend-native inputs it executes on.
    """
    input_provenance = (
        _backend_and_device(X),
        _backend_and_device(weights),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, intercept, n_iter = fista_lla_path(
            QuantileLoss(Q),
            SCADPenalty(alpha=ALPHA, a=SCAD_A),
            X,
            y,
            alpha_path=np.asarray([ALPHA], dtype=np.float64),
            max_lla_per_step=max_lla_per_step,
            max_iter=max_iter,
            lla_tol=lla_tol,
            tol=tol,
            fit_intercept=False,
            sample_weight=weights,
        )
    coef_np = np.asarray(_to_numpy(coef), dtype=np.float64).reshape(-1)
    intercept_f = float(intercept)
    _require_finite(coef_np, f"{label} coefficients")
    if not np.isfinite(intercept_f):
        raise AssertionError(f"{label} intercept is non-finite")
    if int(n_iter) < 1:
        raise AssertionError(f"{label} reported invalid n_iter={n_iter}")
    return coef_np, intercept_f, int(n_iter), input_provenance


def _run_nontrivial(X, y, weights, label):
    return _solve(
        X,
        y,
        weights,
        max_lla_per_step=TARGET_MAX_LLA_PER_STEP,
        max_iter=TARGET_MAX_ITER,
        tol=TARGET_TOL,
        lla_tol=TARGET_LLA_TOL,
        label=label,
    )


def _run_zero(X, y, weights, label):
    return _solve(
        X,
        y,
        weights,
        max_lla_per_step=TARGET_MAX_LLA_PER_STEP,
        max_iter=TARGET_MAX_ITER,
        tol=TARGET_TOL,
        lla_tol=TARGET_LLA_TOL,
        label=label,
    )


def _run_exhaustion_probe(X, y, weights, label):
    """Run a tiny budget and require exactly one target-exhaustion warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        coef, intercept, n_iter = fista_lla_path(
            QuantileLoss(Q),
            SCADPenalty(alpha=ALPHA, a=SCAD_A),
            X,
            y,
            alpha_path=np.asarray([ALPHA], dtype=np.float64),
            max_lla_per_step=EXHAUSTION_MAX_LLA_PER_STEP,
            max_iter=EXHAUSTION_MAX_ITER,
            lla_tol=EXHAUSTION_LLA_TOL,
            tol=EXHAUSTION_TOL,
            fit_intercept=False,
            sample_weight=weights,
        )
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    matching = [
        message for message in convergence_warnings
        if EXHAUSTION_WARNING in message
    ]
    unexpected = [
        message for message in convergence_warnings
        if EXHAUSTION_WARNING not in message
    ]
    if len(matching) != 1 or unexpected:
        raise AssertionError(
            f"{label}: budget probe must emit exactly one expected exhaustion "
            f"warning and no other convergence warning: {convergence_warnings!r}"
        )
    coef_np = np.asarray(_to_numpy(coef), dtype=np.float64).reshape(-1)
    intercept_f = float(intercept)
    _require_finite(coef_np, f"{label} exhaustion-probe coefficients")
    if not np.isfinite(intercept_f):
        raise AssertionError(
            f"{label} exhaustion-probe intercept is non-finite"
        )
    return int(n_iter), len(matching)


def _objective(coef, intercept, X, y, weights):
    residual = y - (X @ coef + intercept)
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(np.average(pinball, weights=weights))
    penalty = SCADPenalty(alpha=ALPHA, a=SCAD_A)
    return fit + float(penalty.value(coef))


def _scad_lla_weights(beta):
    abs_beta = np.abs(beta)
    weights = np.full_like(abs_beta, ALPHA)
    interior = (abs_beta > ALPHA) & (abs_beta <= SCAD_A * ALPHA)
    weights[interior] = (
        SCAD_A * ALPHA - abs_beta[interior]
    ) / (SCAD_A - 1.0)
    weights[abs_beta > SCAD_A * ALPHA] = 0.0
    return weights


def _weighted_quantile_objective(X, y, weights, beta, l1_coeffs):
    residual = np.asarray(y, dtype=np.float64) - np.asarray(
        X, dtype=np.float64
    ) @ np.asarray(beta, dtype=np.float64).reshape(-1)
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(
        np.average(pinball, weights=np.asarray(weights, dtype=np.float64))
    )
    return fit + float(np.sum(np.asarray(l1_coeffs) * np.abs(beta)))


def _lp_fixed_point_reference(X, y, weights, beta):
    """Return the independent HiGHS LP optimum of the LLA surrogate at beta."""
    from scipy.optimize import linprog

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    l1_coeffs = _scad_lla_weights(np.asarray(beta, dtype=np.float64))
    if not np.any(l1_coeffs > 0.0):
        raise AssertionError(
            "nontrivial fixture did not activate the SCAD LLA surrogate"
        )
    n, p = X.shape
    up, um = 2 * p, 2 * p + n
    objective = np.zeros(2 * p + 2 * n, dtype=np.float64)
    objective[:p] = l1_coeffs
    objective[p : 2 * p] = l1_coeffs
    normalized_weight = weights / float(np.sum(weights))
    objective[up : up + n] = Q * normalized_weight
    objective[um:] = (1.0 - Q) * normalized_weight
    A_eq = np.zeros((n, 2 * p + 2 * n), dtype=np.float64)
    A_eq[:, :p] = X
    A_eq[:, p : 2 * p] = -X
    A_eq[:, up : up + n] = np.eye(n, dtype=np.float64)
    A_eq[:, um:] = -np.eye(n, dtype=np.float64)
    result = linprog(
        objective,
        A_eq=A_eq,
        b_eq=y,
        bounds=[(0.0, None)] * (2 * p + 2 * n),
        method="highs",
    )
    if not bool(result.success):
        raise AssertionError(
            "weighted Quantile-L1 LP reference failed: "
            f"status={result.status}, message={result.message!r}"
        )
    beta_lp = result.x[:p] - result.x[p : 2 * p]
    reconstructed = _weighted_quantile_objective(
        X, y, weights, beta_lp, l1_coeffs
    )
    if abs(reconstructed - float(result.fun)) > 1e-10:
        raise AssertionError(
            "weighted Quantile-L1 LP objective reconstruction drifted: "
            f"{reconstructed:.16g} vs solver {float(result.fun):.16g}"
        )
    return beta_lp, float(result.fun)


def _require_interior_fixed_point(beta, label):
    """Fail closed unless the SCAD surrogate has an active interior optimum."""
    abs_beta = np.abs(np.asarray(beta, dtype=np.float64))
    interior = (abs_beta > ALPHA) & (abs_beta <= SCAD_A * ALPHA)
    if not np.any(interior):
        raise AssertionError(
            f"{label}: fixed point {abs_beta!r} does not activate the SCAD "
            "curvature interior"
        )


def _fixed_point_gap(X, y, weights, beta, expected_lp_value):
    l1_coeffs = _scad_lla_weights(np.asarray(beta, dtype=np.float64))
    gap = (
        _weighted_quantile_objective(X, y, weights, beta, l1_coeffs)
        - expected_lp_value
    )
    if not np.isfinite(gap):
        raise AssertionError("fixed-point gap is non-finite")
    return float(gap)


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

    zero_X, zero_y, zero_weights = _converged_data()
    X, y, weights = _nontrivial_data()
    probe_X, probe_y, probe_weights = _exhaustion_data()

    (
        cpu_zero_coef,
        cpu_zero_intercept,
        cpu_zero_iter,
        cpu_zero_provenance,
    ) = _run_zero(zero_X, zero_y, zero_weights, "cpu zero fixture")
    if cpu_zero_provenance != (("numpy", "cpu"), ("numpy", "cpu")):
        raise AssertionError(
            f"cpu zero fixture ran on unexpected inputs {cpu_zero_provenance!r}"
        )
    if np.max(np.abs(cpu_zero_coef)) > 1e-12 or cpu_zero_intercept != 0.0:
        raise AssertionError(
            "scalar Quantile LLA converged fixed-point CPU oracle drifted from zero"
        )

    (
        cpu_coef,
        cpu_intercept,
        cpu_iter,
        cpu_provenance,
    ) = _run_nontrivial(X, y, weights, "cpu nontrivial fixture")
    if cpu_provenance != (("numpy", "cpu"), ("numpy", "cpu")):
        raise AssertionError(
            f"cpu nontrivial fixture ran on unexpected inputs {cpu_provenance!r}"
        )
    _require_interior_fixed_point(cpu_coef, "cpu nontrivial fixture")
    _, cpu_lp_value = _lp_fixed_point_reference(X, y, weights, cpu_coef)
    cpu_gap = _fixed_point_gap(X, y, weights, cpu_coef, cpu_lp_value)
    if abs(cpu_gap) > ATOL_FIXED_POINT:
        raise AssertionError(
            f"cpu nontrivial fixture fixed-point gap {cpu_gap:.3e} exceeds "
            f"{ATOL_FIXED_POINT:.3e} in magnitude"
        )
    cpu_objective = _objective(cpu_coef, cpu_intercept, X, y, weights)
    cpu_probe_iter, cpu_probe_warnings = _run_exhaustion_probe(
        probe_X, probe_y, probe_weights, "cpu budget probe"
    )

    cases = []
    max_zero_param_error = 0.0
    max_param_error = 0.0
    max_objective_error = 0.0
    max_fixed_point_gap = abs(cpu_gap)
    for backend in ("cupy", "torch"):
        zero_Xb, zero_yb, zero_wb = _native_inputs(
            backend, zero_X, zero_y, zero_weights, cp, torch
        )
        Xb, yb, wb = _native_inputs(backend, X, y, weights, cp, torch)
        probe_Xb, probe_yb, probe_wb = _native_inputs(
            backend, probe_X, probe_y, probe_weights, cp, torch
        )

        zero_coef, zero_intercept, zero_iter, zero_provenance = _run_zero(
            zero_Xb, zero_yb, zero_wb, f"{backend} zero fixture"
        )
        expected_provenance = (
            (backend, "cuda:0"),
            (backend, "cuda:0"),
        )
        if zero_provenance != expected_provenance:
            raise AssertionError(
                f"{backend}: zero fixture ran on unexpected inputs "
                f"{zero_provenance!r}"
            )
        zero_param_error = max(
            float(np.max(np.abs(zero_coef - cpu_zero_coef))),
            abs(zero_intercept - cpu_zero_intercept),
        )
        if zero_param_error > ATOL_PARAM:
            raise AssertionError(
                f"{backend}: zero fixture parameter error {zero_param_error:.3e} "
                f"> {ATOL_PARAM:.3e}"
            )

        coef, intercept, n_iter, provenance = _run_nontrivial(
            Xb, yb, wb, f"{backend} nontrivial fixture"
        )
        if provenance != expected_provenance:
            raise AssertionError(
                f"{backend}: nontrivial fixture ran on unexpected inputs "
                f"{provenance!r}"
            )
        _require_interior_fixed_point(coef, f"{backend} nontrivial fixture")
        param_error = max(
            float(np.max(np.abs(coef - cpu_coef))),
            abs(intercept - cpu_intercept),
        )
        objective = _objective(coef, intercept, X, y, weights)
        objective_error = abs(objective - cpu_objective)
        if not (
            np.isfinite(objective)
            and np.isfinite(cpu_objective)
            and np.isfinite(objective_error)
        ):
            raise AssertionError(
                f"{backend}: non-finite objective cannot be compared"
            )
        _, lp_value = _lp_fixed_point_reference(X, y, weights, coef)
        gap = _fixed_point_gap(X, y, weights, coef, lp_value)
        if param_error > ATOL_PARAM:
            raise AssertionError(
                f"{backend}: nontrivial parameter error {param_error:.3e} > "
                f"{ATOL_PARAM:.3e}"
            )
        if objective_error > ATOL_OBJECTIVE:
            raise AssertionError(
                f"{backend}: nontrivial objective error {objective_error:.3e} > "
                f"{ATOL_OBJECTIVE:.3e}"
            )
        if abs(gap) > ATOL_FIXED_POINT:
            raise AssertionError(
                f"{backend}: nontrivial fixed-point gap {gap:.3e} exceeds "
                f"{ATOL_FIXED_POINT:.3e} in magnitude"
            )

        probe_iter, probe_warnings = _run_exhaustion_probe(
            probe_Xb, probe_yb, probe_wb, f"{backend} budget probe"
        )

        max_zero_param_error = max(max_zero_param_error, zero_param_error)
        max_param_error = max(max_param_error, param_error)
        max_objective_error = max(max_objective_error, objective_error)
        max_fixed_point_gap = max(max_fixed_point_gap, abs(gap))
        cases.append(
            {
                "backend": backend,
                "zero_n_iter": zero_iter,
                "zero_parameter_error": zero_param_error,
                "n_iter": n_iter,
                "coef_error": float(np.max(np.abs(coef - cpu_coef))),
                "intercept_error": abs(intercept - cpu_intercept),
                "parameter_error": param_error,
                "objective": objective,
                "objective_error": objective_error,
                "fixed_point_gap": gap,
                "exhaustion_n_iter": probe_iter,
                "exhaustion_warning_count": probe_warnings,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "quantile": Q,
        "alpha": ALPHA,
        "fixtures": {
            "zero_fixed_point": "paired_sign_symmetric_orthogonal",
            "nontrivial_fixed_point": "weighted_single_feature_scad_interior",
            "budget_exhaustion": "weighted_orthogonal_scad",
        },
        "cpu": {
            "zero_n_iter": cpu_zero_iter,
            "zero_coef_abs_max": float(np.max(np.abs(cpu_zero_coef))),
            "nontrivial_n_iter": cpu_iter,
            "nontrivial_objective": cpu_objective,
            "nontrivial_fixed_point_gap": cpu_gap,
            "nontrivial_fixed_point_l1_weight_max": float(
                np.max(_scad_lla_weights(cpu_coef))
            ),
            "exhaustion_n_iter": cpu_probe_iter,
            "exhaustion_warning_count": cpu_probe_warnings,
        },
        "solver_controls": {
            "target_max_lla_per_step": TARGET_MAX_LLA_PER_STEP,
            "target_max_iter": TARGET_MAX_ITER,
            "target_tol": TARGET_TOL,
            "target_lla_tol": TARGET_LLA_TOL,
            "exhaustion_max_lla_per_step": EXHAUSTION_MAX_LLA_PER_STEP,
            "exhaustion_max_iter": EXHAUSTION_MAX_ITER,
            "exhaustion_tol": EXHAUSTION_TOL,
            "exhaustion_lla_tol": EXHAUSTION_LLA_TOL,
            "exhaustion_warning": EXHAUSTION_WARNING,
        },
        "cases": cases,
        "max_errors": {
            "zero_parameter": max_zero_param_error,
            "nontrivial_parameter": max_param_error,
            "nontrivial_objective": max_objective_error,
            "nontrivial_fixed_point_gap": max_fixed_point_gap,
        },
        "tolerances": {
            "parameter": ATOL_PARAM,
            "objective": ATOL_OBJECTIVE,
            "fixed_point_gap": ATOL_FIXED_POINT,
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
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
