#!/usr/bin/env python3
"""Three-backend penalized CoxPH parity and P100 timing diagnostic."""

from __future__ import annotations

import statistics
import time

import cupy as cp
import numpy as np
import torch

from dev.benchmarks.pr79.generators.survival import generate_coxph_penalized
from statgpu.survival import CoxPH

PENALTY = 0.1
# The canonical stored timings were produced with one untimed warmup fit
# followed by ten measured fits. Keep this script aligned with that evidence.
WARMUPS = 1
MEASURED_REPETITIONS = 10
COEF_TOL = 1e-6
LL_TOL = 1e-9
KKT_TOL = 1e-7
BSE_TOL = 1e-5


def _synchronize(device: str) -> None:
    if device == "cuda":
        cp.cuda.Stream.null.synchronize()
    elif device == "torch":
        torch.cuda.synchronize()


def _backend_input(X: np.ndarray, device: str):
    if device == "cuda":
        return cp.asarray(X)
    if device == "torch":
        return torch.as_tensor(X, dtype=torch.float64, device="cuda")
    return X


def _fit(X, time_values, event_values, device: str, *, inference: bool = True):
    model = CoxPH(
        ties="efron",
        penalty=PENALTY,
        compute_inference=inference,
        compute_cindex=False,
        tol=1e-6,
        max_iter=30,
        device=device,
    )
    model.fit(_backend_input(X, device), time=time_values, event=event_values)
    return model


def _relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1e-30)


def _fixed_beta_torch_bse_error(
    X: np.ndarray,
    time_values: np.ndarray,
    event_values: np.ndarray,
    beta_ref: np.ndarray,
    bse_ref: np.ndarray,
) -> float:
    model = CoxPH(
        ties="efron",
        penalty=PENALTY,
        compute_inference=False,
        compute_cindex=False,
        tol=1e-6,
        max_iter=30,
    )
    efron_pre = model._efron_unique_failure_indices(time_values, event_values)

    X_torch = torch.as_tensor(X, dtype=torch.float64, device="cuda")
    time_torch = torch.as_tensor(time_values, dtype=torch.float64, device="cuda")
    event_torch = torch.as_tensor(event_values, dtype=torch.int32, device="cuda")
    beta_torch = torch.as_tensor(beta_ref, dtype=torch.float64, device="cuda")

    _, hessian, _ = model._compute_gradient_hessian_torch(
        beta_torch,
        X_torch,
        time_torch,
        event_torch,
        efron_pre,
        return_aux=True,
    )
    penalized_hessian = hessian.cpu().numpy()
    penalized_hessian -= 2.0 * PENALTY * np.eye(X.shape[1])
    covariance = np.linalg.solve(-penalized_hessian, np.eye(X.shape[1]))
    bse_torch = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return float(
        np.max(
            np.abs(bse_torch - bse_ref)
            / np.maximum(np.abs(bse_ref), 1e-30)
        )
    )


def _benchmark(
    X: np.ndarray,
    time_values: np.ndarray,
    event_values: np.ndarray,
    device: str,
) -> dict[str, float]:
    backend_X = _backend_input(X, device)

    for _ in range(WARMUPS):
        model = CoxPH(
            ties="efron",
            penalty=PENALTY,
            compute_inference=True,
            compute_cindex=False,
            tol=1e-6,
            max_iter=30,
            device=device,
        )
        model.fit(backend_X, time=time_values, event=event_values)
    _synchronize(device)

    samples = []
    iterations = None
    for _ in range(MEASURED_REPETITIONS):
        model = CoxPH(
            ties="efron",
            penalty=PENALTY,
            compute_inference=True,
            compute_cindex=False,
            tol=1e-6,
            max_iter=30,
            device=device,
        )
        _synchronize(device)
        started = time.perf_counter()
        model.fit(backend_X, time=time_values, event=event_values)
        _synchronize(device)
        samples.append(time.perf_counter() - started)
        iterations = model._iterations

    return {
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "iterations": float(iterations),
    }


def main() -> None:
    X, time_values, event_values, _ = generate_coxph_penalized(100, 8, 42)
    order = np.argsort(time_values, kind="stable")
    X = np.asarray(X[order], dtype=np.float64)
    time_values = np.asarray(time_values[order], dtype=np.float64)
    event_values = np.asarray(event_values[order], dtype=np.int32)

    models = {
        "NumPy": _fit(X, time_values, event_values, "cpu"),
        "CuPy": _fit(X, time_values, event_values, "cuda"),
        "Torch": _fit(X, time_values, event_values, "torch"),
    }
    reference = models["NumPy"]
    beta_ref = reference.coef_.copy()
    bse_ref = np.sqrt(np.maximum(np.diag(reference._var_matrix), 0.0))

    torch_bse_error = _fixed_beta_torch_bse_error(
        X, time_values, event_values, beta_ref, bse_ref
    )

    print("=== Three-backend parity ===")
    for label, model in models.items():
        coef_diff = float(np.linalg.norm(model.coef_ - beta_ref))
        ll_error = _relative_error(model._log_likelihood, reference._log_likelihood)
        kkt = float(model._final_kkt_inf)
        print(
            f"{label}: LL={model._log_likelihood:.6f}, "
            f"KKT={kkt:.2e}, coef_diff={coef_diff:.2e}, "
            f"iters={model._iterations}, termination={model._termination_reason}"
        )
        assert coef_diff <= COEF_TOL
        assert ll_error <= LL_TOL
        assert kkt <= KKT_TOL
        assert model._converged
        assert model._termination_reason == "kkt_converged"

    print(f"Torch fixed-beta BSE error: {torch_bse_error:.6e}")
    assert torch_bse_error <= BSE_TOL

    print(
        f"\n=== Timing ({WARMUPS} warmup + "
        f"{MEASURED_REPETITIONS} measured fits) ==="
    )
    timings = {
        label: _benchmark(X, time_values, event_values, device)
        for label, device in (
            ("NumPy", "cpu"),
            ("CuPy", "cuda"),
            ("Torch", "torch"),
        )
    }
    numpy_median = timings["NumPy"]["median_ms"]

    for label, result in timings.items():
        speedup = numpy_median / result["median_ms"]
        print(
            f"{label}: median={result['median_ms']:.1f}ms, "
            f"min={result['min_ms']:.1f}ms, "
            f"max={result['max_ms']:.1f}ms, "
            f"speedup={speedup:.2f}x, "
            f"iters={int(result['iterations'])}"
        )


if __name__ == "__main__":
    main()
