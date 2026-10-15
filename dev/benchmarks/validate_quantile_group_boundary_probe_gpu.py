#!/usr/bin/env python3
"""Physical CUDA gate for Quantile Group flat-IRLS boundary convergence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty
from statgpu.solvers._quantile_group_proximal_irls_lla import (
    quantile_group_proximal_irls_lla_solver,
)


SCHEMA_VERSION = 1
GROUPS = [[0, 1], [2, 3]]
ALPHA = 0.3


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_state():
    return _git("rev-parse", "HEAD"), not bool(_git("status", "--porcelain"))


def _array_location(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        return "cupy", f"cuda:{int(value.device.id)}"
    if module.startswith("torch"):
        return "torch", str(value.device)
    return "numpy", "cpu"


class _BoundaryConvergedQuantileLoss(QuantileLoss):
    """Return a stable flat point while reporting the full requested budget."""

    def __init__(self):
        super().__init__(0.35)
        self.calls = []

    def irls(
        self,
        X,
        y,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        X_backend, X_device = _array_location(X)
        weight_backend, weight_device = _array_location(sample_weight)
        init_location = None if init_coef is None else _array_location(init_coef)
        self.calls.append(
            {
                "X": [X_backend, X_device],
                "sample_weight": [weight_backend, weight_device],
                "init_coef": None if init_location is None else list(init_location),
                "max_iter": int(max_iter),
            }
        )

        if X_backend == "torch":
            import torch

            point = torch.full(
                (int(X.shape[1]),),
                2.0,
                dtype=X.dtype,
                device=X.device,
            )
        elif X_backend == "cupy":
            import cupy as cp

            point = cp.full((int(X.shape[1]),), 2.0, dtype=X.dtype)
        else:
            point = np.full((int(X.shape[1]),), 2.0, dtype=np.float64)
        # This deliberately exercises the ambiguous boundary: the ordinary
        # IRLS API reports n_iter == max_iter even though the returned point is
        # already a fixed point. The Group solver's one-step probe must accept
        # it rather than misclassifying it as exhaustion.
        return point, int(max_iter)


def _native_inputs(backend, X, y, weights, cp, torch):
    if backend == "cupy":
        cp.cuda.Device(0).use()
        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            cp.asarray(weights, dtype=cp.float64),
            cp.full(X.shape[1], 2.0, dtype=cp.float64),
        )
    if backend == "torch":
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return (
            torch.as_tensor(X, dtype=torch.float64, device=device),
            torch.as_tensor(y, dtype=torch.float64, device=device),
            torch.as_tensor(weights, dtype=torch.float64, device=device),
            torch.full((X.shape[1],), 2.0, dtype=torch.float64, device=device),
        )
    return X, y, weights, np.full(X.shape[1], 2.0, dtype=np.float64)


def _run_case(backend, X, y, weights, cp=None, torch=None):
    Xb, yb, wb, init = _native_inputs(backend, X, y, weights, cp, torch)
    loss = _BoundaryConvergedQuantileLoss()
    penalty = GroupSCADPenalty(alpha=ALPHA, a=3.7, groups=GROUPS)
    coef, intercept, n_iter = quantile_group_proximal_irls_lla_solver(
        loss,
        penalty,
        Xb,
        yb,
        alpha_path=np.asarray([ALPHA], dtype=np.float64),
        max_lla_per_step=1,
        max_iter=2,
        tol=1e-12,
        lla_tol=1e-12,
        fit_intercept=False,
        sample_weight=wb,
        init_coef=init,
        fail_on_target_nonconvergence=True,
    )
    coef_np = np.asarray(_to_numpy(coef), dtype=np.float64)
    error = float(np.max(np.abs(coef_np - 2.0)))
    if error != 0.0:
        raise AssertionError(f"{backend}: boundary probe changed accepted coefficients: {error}")
    if float(intercept) != 0.0:
        raise AssertionError(f"{backend}: unexpected intercept {intercept!r}")
    if int(n_iter) != 2:
        raise AssertionError(f"{backend}: diagnostic probe changed n_iter: {n_iter!r}")
    if len(loss.calls) != 2:
        raise AssertionError(f"{backend}: expected solve + one probe, got {len(loss.calls)} IRLS calls")

    expected_device = "cpu" if backend == "numpy" else "cuda:0"
    for call in loss.calls:
        if call["X"] != [backend, expected_device]:
            raise AssertionError(f"{backend}: IRLS X ownership drifted: {call['X']!r}")
        if call["sample_weight"] != [backend, expected_device]:
            raise AssertionError(
                f"{backend}: IRLS weight ownership drifted: {call['sample_weight']!r}"
            )
    if loss.calls[0]["init_coef"] is not None:
        raise AssertionError(f"{backend}: flat solve should use canonical no-init IRLS")
    if loss.calls[1]["init_coef"] != [backend, expected_device]:
        raise AssertionError(
            f"{backend}: boundary-probe init ownership drifted: {loss.calls[1]['init_coef']!r}"
        )

    return {
        "backend": backend,
        "device": expected_device,
        "parameter_error": error,
        "n_iter": int(n_iter),
        "irls_calls": loss.calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dev/reviews/pr166_quantile_group_boundary_probe_gpu.json",
    )
    args = parser.parse_args()

    source_sha, source_clean = _source_state()
    if not source_clean:
        raise RuntimeError("Quantile Group boundary-probe gate requires a clean source")

    import cupy as cp
    import torch

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")

    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.8, -0.5, 0.4, -0.3], dtype=np.float64)
    weights = np.asarray([0.6, 0.9, 1.2, 1.6], dtype=np.float64)

    cases = [
        _run_case("numpy", X, y, weights),
        _run_case("cupy", X, y, weights, cp=cp, torch=torch),
        _run_case("torch", X, y, weights, cp=cp, torch=torch),
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": source_clean,
        "cases": cases,
        "environment": {
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
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
