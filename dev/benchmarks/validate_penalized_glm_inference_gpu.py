#!/usr/bin/env python3
"""Physical CUDA acceptance for the penalized-GLM inference contract.

This validator is intentionally not a hosted-GPU substitute. It requires a
clean git worktree plus operational CuPy CUDA and Torch CUDA in the same run,
records the exact source SHA, and checks representative non-Gaussian L2
M-estimation inference against the CPU implementation. It also feeds Torch CUDA
containers into CuPy execution and CuPy containers into Torch execution so the
post-fit inference boundary proves DLPack/concrete-device alignment rather than
only same-container happy paths.

Schema v3 additionally proves that weighted smooth L2 ``solver="auto"`` uses
the canonical Newton dispatch now that Newton supports analytic weights. The
unweighted cases stay on explicit FISTA so the validator continues to cover the
independent FISTA fit plus post-fit inference/device path.

Example
-------
python dev/benchmarks/validate_penalized_glm_inference_gpu.py \
  --output results/pr142_penalized_glm_inference_gpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from statgpu.linear_model import (
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
)


SCHEMA_VERSION = 3
ATOL_COEF = 2e-6
ATOL_INFERENCE = 1e-5


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "physical penalized-GLM inference validation requires a clean worktree"
        )
    return sha


def _require_gpu_backends():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("CuPy is required for physical validation") from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("Torch is required for physical validation") from exc

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")

    # Use one concrete ordinal in both libraries so container-crossing tests
    # exercise conversion semantics rather than comparing different hardware.
    cupy_device = 0
    torch_device = torch.device("cuda:0")
    return cp, torch, cupy_device, torch_device


def _logistic_data(seed=14201, n=512, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.62, -0.41, 0.28, 0.18, -0.12])[:p]
    eta = -0.2 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X.astype(np.float64), y


def _poisson_data(seed=14202, n=512, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p))
    beta = np.array([0.22, -0.16, 0.12, 0.08, -0.05])[:p]
    mu = np.exp(0.15 + X @ beta)
    y = rng.poisson(mu).astype(np.float64)
    return X.astype(np.float64), y


def _snapshot(model):
    result = model._inference_result
    if result is None:
        raise AssertionError("inference result was not published")
    return {
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "intercept": float(model.intercept_),
        "bse": np.asarray(model._bse, dtype=np.float64),
        "pvalues": np.asarray(model._pvalues, dtype=np.float64),
        "conf_int": np.asarray(model._conf_int, dtype=np.float64),
        "requested": model.inference_requested_method_,
        "resolved": model.inference_resolved_method_,
        "reported": model.inference_method_,
        "target": model.inference_target_,
        "public_solver": str(model.solver),
        "selected_solver": str(getattr(model, "_selected_solver", "")),
        "metadata": dict(result.metadata),
    }


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _assert_close_case(name, cpu, gpu):
    errors = {
        "coef": _max_abs(cpu["coef"], gpu["coef"]),
        "intercept": abs(cpu["intercept"] - gpu["intercept"]),
        "bse": _max_abs(cpu["bse"], gpu["bse"]),
        "pvalues": _max_abs(cpu["pvalues"], gpu["pvalues"]),
        "conf_int": _max_abs(cpu["conf_int"], gpu["conf_int"]),
    }
    if errors["coef"] > ATOL_COEF or errors["intercept"] > ATOL_COEF:
        raise AssertionError(f"{name} coefficient parity failed: {errors}")
    for key in ("bse", "pvalues", "conf_int"):
        if errors[key] > ATOL_INFERENCE:
            raise AssertionError(f"{name} {key} parity failed: {errors[key]:.3e}")
    if gpu["requested"] != "auto":
        raise AssertionError(f"{name} requested method drifted: {gpu['requested']!r}")
    if gpu["resolved"] != "m_estimation" or gpu["reported"] != "m_estimation":
        raise AssertionError(
            f"{name} method identity mismatch: "
            f"{gpu['resolved']!r}/{gpu['reported']!r}"
        )
    if gpu["target"] != "penalized_estimating_equation":
        raise AssertionError(f"{name} target mismatch: {gpu['target']!r}")
    return errors


def _assert_backend(name, result, backend, device):
    metadata = result["metadata"]
    if metadata.get("numerical_backend") != backend:
        raise AssertionError(
            f"{name} numerical backend mismatch: {metadata.get('numerical_backend')!r}"
        )
    if metadata.get("numerical_device") != device:
        raise AssertionError(
            f"{name} concrete device mismatch: {metadata.get('numerical_device')!r}"
        )


def _assert_weighted_auto_solver(name, result, weighted):
    if not weighted:
        return
    if result["public_solver"] != "auto":
        raise AssertionError(
            f"{name} public weighted solver request drifted: {result['public_solver']!r}"
        )
    if result["selected_solver"] != "newton":
        raise AssertionError(
            f"{name} weighted auto execution did not select Newton: "
            f"{result['selected_solver']!r}"
        )


def _fit_case(cls, X, y, *, device, sample_weight=None, solver="fista"):
    model = cls(
        penalty="l2",
        alpha=0.035,
        solver=solver,
        device=device,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
        max_iter=4000,
        tol=1e-9,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return _snapshot(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/pr142_penalized_glm_inference_gpu.json",
    )
    args = parser.parse_args()

    sha = _require_clean_source()
    cp, torch, cupy_device, torch_device = _require_gpu_backends()

    cases = []
    for family, cls, data_fn in (
        ("logistic", PenalizedLogisticRegression, _logistic_data),
        ("poisson", PenalizedPoissonRegression, _poisson_data),
    ):
        X, y = data_fn()
        weights = np.linspace(0.45, 1.75, X.shape[0], dtype=np.float64)

        for weighted in (False, True):
            sw = weights if weighted else None
            solver = "auto" if weighted else "fista"
            cpu = _fit_case(
                cls, X, y, device="cpu", sample_weight=sw, solver=solver
            )

            with cp.cuda.Device(cupy_device):
                X_cp = cp.asarray(X)
                y_cp = cp.asarray(y)
                sw_cp = None if sw is None else cp.asarray(sw)

            X_t = torch.as_tensor(X, dtype=torch.float64, device=torch_device)
            y_t = torch.as_tensor(y, dtype=torch.float64, device=torch_device)
            sw_t = (
                None
                if sw is None
                else torch.as_tensor(sw, dtype=torch.float64, device=torch_device)
            )

            with cp.cuda.Device(cupy_device):
                cupy_result = _fit_case(
                    cls,
                    X_cp,
                    y_cp,
                    device="cuda",
                    sample_weight=sw_cp,
                    solver=solver,
                )
                # Cross-container: Torch CUDA input, CuPy execution/inference.
                cupy_from_torch = _fit_case(
                    cls,
                    X_t,
                    y_t,
                    device="cuda",
                    sample_weight=sw_t,
                    solver=solver,
                )

            torch_result = _fit_case(
                cls,
                X_t,
                y_t,
                device="torch",
                sample_weight=sw_t,
                solver=solver,
            )
            # Cross-container: CuPy input, Torch execution/inference.
            torch_from_cupy = _fit_case(
                cls,
                X_cp,
                y_cp,
                device="torch",
                sample_weight=sw_cp,
                solver=solver,
            )

            cuda_label = f"cuda:{cupy_device}"
            for case_name, result, backend in (
                ("cupy", cupy_result, "cupy"),
                ("cupy_from_torch", cupy_from_torch, "cupy"),
                ("torch", torch_result, "torch"),
                ("torch_from_cupy", torch_from_cupy, "torch"),
            ):
                label = f"{family}/weighted={weighted}/{case_name}"
                _assert_backend(label, result, backend, cuda_label)
                _assert_weighted_auto_solver(label, result, weighted)

            cases.append(
                {
                    "family": family,
                    "weighted": weighted,
                    "solver_request": solver,
                    "selected_solver_cpu": cpu["selected_solver"],
                    "selected_solver_cupy": cupy_result["selected_solver"],
                    "selected_solver_torch": torch_result["selected_solver"],
                    "cupy_errors": _assert_close_case(
                        f"{family}/weighted={weighted}/cupy", cpu, cupy_result
                    ),
                    "cupy_from_torch_errors": _assert_close_case(
                        f"{family}/weighted={weighted}/cupy_from_torch",
                        cpu,
                        cupy_from_torch,
                    ),
                    "torch_errors": _assert_close_case(
                        f"{family}/weighted={weighted}/torch", cpu, torch_result
                    ),
                    "torch_from_cupy_errors": _assert_close_case(
                        f"{family}/weighted={weighted}/torch_from_cupy",
                        cpu,
                        torch_from_cupy,
                    ),
                    "cupy_metadata": cupy_result["metadata"],
                    "cupy_from_torch_metadata": cupy_from_torch["metadata"],
                    "torch_metadata": torch_result["metadata"],
                    "torch_from_cupy_metadata": torch_from_cupy["metadata"],
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "git_sha": sha,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "torch": torch.__version__,
        "cupy_device_id": cupy_device,
        "cupy_device_name": cp.cuda.runtime.getDeviceProperties(cupy_device)["name"].decode(),
        "torch_device_name": torch.cuda.get_device_name(torch_device),
        "thresholds": {
            "coef_abs": ATOL_COEF,
            "inference_abs": ATOL_INFERENCE,
        },
        "cases": cases,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
