#!/usr/bin/env python3
"""Exact-source physical CUDA gate for backend-native Gaussian residual bootstrap.

This validator is intentionally separate from hosted CPU CI. It requires a
clean worktree plus operational CuPy CUDA and Torch CUDA in one process, uses
one fixed backend-neutral residual-index schedule per case, and compares CuPy
and Torch bootstrap inference with the NumPy reference under identical draws.

The acceptance constants below are frozen before the first physical run for
PR #147. Do not loosen them after a failed run merely to obtain green status;
a justified tolerance change requires an explicit review, schema bump, and a
new physical artifact.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from statgpu.linear_model import PenalizedLinearRegression


SCHEMA_VERSION = 1
BOOTSTRAP_DRAWS = 24
BOOTSTRAP_RANDOM_STATE = 147001
ATOL_PARAMS = 2.0e-5
ATOL_BSE = 5.0e-4
ATOL_PVALUES = 1.0e-1
ATOL_CONF_INT = 1.0e-3


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "physical residual-bootstrap validation requires a clean worktree"
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

    device_id = 0
    torch_device = torch.device("cuda:0")
    return cp, torch, device_id, torch_device


def _device_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _data(seed=147, n=192, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.00, -0.72, 0.50, -0.35], dtype=np.float64)[:p]
    y = 0.45 + X @ beta + rng.normal(scale=0.22, size=n)
    return X.astype(np.float64), y.astype(np.float64)


def _to_numpy(value):
    if hasattr(value, "get"):
        return np.asarray(value.get())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _fit_case(X, y, *, penalty, l1_ratio, device):
    model = PenalizedLinearRegression(
        penalty=penalty,
        alpha=0.045,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        device=device,
        solver="fista",
        max_iter=1800,
        tol=1e-8,
        compute_inference=True,
        inference_method="bootstrap",
        cov_type="nonrobust",
    )
    model.n_bootstrap = BOOTSTRAP_DRAWS
    model.bootstrap_random_state = BOOTSTRAP_RANDOM_STATE
    model.fit(X, y)
    result = model._inference_result
    if result is None:
        raise AssertionError("residual bootstrap did not publish an inference result")
    metadata = dict(result.metadata)
    return {
        "params": np.asarray(result.params, dtype=np.float64),
        "coef": np.asarray(_to_numpy(model.coef_), dtype=np.float64),
        "intercept": float(model.intercept_),
        "bse": np.asarray(result.bse, dtype=np.float64),
        "pvalues": np.asarray(result.pvalues, dtype=np.float64),
        "conf_int": np.asarray(result.conf_int, dtype=np.float64),
        "requested_solver": str(model.solver),
        "selected_solver": str(getattr(model, "_selected_solver", "")),
        "requested_method": str(model.inference_requested_method_),
        "resolved_method": str(model.inference_resolved_method_),
        "reported_method": str(model.inference_method_),
        "target": str(model.inference_target_),
        "metadata": metadata,
    }


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _assert_contract(name, snap, *, backend, device):
    if snap["requested_method"] != "bootstrap":
        raise AssertionError(f"{name}: requested method drifted")
    if snap["resolved_method"] != "residual_bootstrap":
        raise AssertionError(f"{name}: resolved method drifted")
    if snap["reported_method"] != "residual_bootstrap":
        raise AssertionError(f"{name}: reported method drifted")
    if snap["target"] != "penalized_coefficient_distribution":
        raise AssertionError(f"{name}: inferential target drifted")
    if snap["requested_solver"] != "fista":
        raise AssertionError(f"{name}: public solver request drifted")
    if snap["selected_solver"] != "fista":
        raise AssertionError(
            f"{name}: selected solver drifted to {snap['selected_solver']!r}"
        )

    metadata = snap["metadata"]
    expected = {
        "resampling_scope": "unweighted_gaussian_residual",
        "resampling_schedule": "numpy_generator_control_plane",
        "numerical_backend": backend,
        "numerical_device": device,
        "reporting_backend": "numpy",
        "reporting_boundary": "post_numerical_inference",
        "n_bootstrap": BOOTSTRAP_DRAWS,
        "random_state": BOOTSTRAP_RANDOM_STATE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AssertionError(
                f"{name}: metadata[{key!r}]={metadata.get(key)!r}, expected {value!r}"
            )
    schedule_hash = str(metadata.get("resampling_schedule_sha256", ""))
    if len(schedule_hash) != 64:
        raise AssertionError(f"{name}: missing stable resampling schedule hash")
    child_solvers = metadata.get("child_selected_solvers")
    if child_solvers != ["fista"]:
        raise AssertionError(
            f"{name}: child solver provenance drifted: {child_solvers!r}"
        )


def _compare(name, reference, candidate):
    errors = {
        "params": _max_abs(reference["params"], candidate["params"]),
        "bse": _max_abs(reference["bse"], candidate["bse"]),
        "pvalues": _max_abs(reference["pvalues"], candidate["pvalues"]),
        "conf_int": _max_abs(reference["conf_int"], candidate["conf_int"]),
    }
    if errors["params"] > ATOL_PARAMS:
        raise AssertionError(
            f"{name}: params error {errors['params']:.3e} exceeds {ATOL_PARAMS:.3e}"
        )
    if errors["bse"] > ATOL_BSE:
        raise AssertionError(
            f"{name}: bse error {errors['bse']:.3e} exceeds {ATOL_BSE:.3e}"
        )
    if errors["pvalues"] > ATOL_PVALUES:
        raise AssertionError(
            f"{name}: pvalue error {errors['pvalues']:.3e} exceeds {ATOL_PVALUES:.3e}"
        )
    if errors["conf_int"] > ATOL_CONF_INT:
        raise AssertionError(
            f"{name}: conf_int error {errors['conf_int']:.3e} exceeds {ATOL_CONF_INT:.3e}"
        )
    ref_hash = reference["metadata"]["resampling_schedule_sha256"]
    got_hash = candidate["metadata"]["resampling_schedule_sha256"]
    if ref_hash != got_hash:
        raise AssertionError(
            f"{name}: backend did not consume the exact reference resampling schedule"
        )
    return errors


def _json_snapshot(snap):
    return {
        "params": snap["params"].tolist(),
        "coef": snap["coef"].tolist(),
        "intercept": snap["intercept"],
        "bse": snap["bse"].tolist(),
        "pvalues": snap["pvalues"].tolist(),
        "conf_int": snap["conf_int"].tolist(),
        "requested_solver": snap["requested_solver"],
        "selected_solver": snap["selected_solver"],
        "requested_method": snap["requested_method"],
        "resolved_method": snap["resolved_method"],
        "reported_method": snap["reported_method"],
        "target": snap["target"],
        "metadata": snap["metadata"],
    }


def run(output: Path):
    source_sha = _require_clean_source()
    cp, torch, device_id, torch_device = _require_gpu_backends()
    X_np, y_np = _data()

    cupy_properties = cp.cuda.runtime.getDeviceProperties(device_id)
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "torch": torch.__version__,
        "cuda_device_ordinal": device_id,
        "cupy_device_name": _device_name(cupy_properties["name"]),
        "torch_device_name": torch.cuda.get_device_name(torch_device),
    }

    cases = {}
    for penalty, l1_ratio in (("l1", 0.5), ("elasticnet", 0.40)):
        reference = _fit_case(
            X_np,
            y_np,
            penalty=penalty,
            l1_ratio=l1_ratio,
            device="cpu",
        )
        _assert_contract(
            f"{penalty}/numpy",
            reference,
            backend="numpy",
            device="cpu",
        )
        cases[f"{penalty}/numpy"] = {
            "snapshot": _json_snapshot(reference),
            "errors_vs_numpy": None,
        }

        with cp.cuda.Device(device_id):
            X_cp = cp.asarray(X_np)
            y_cp = cp.asarray(y_np)
            cupy_native = _fit_case(
                X_cp,
                y_cp,
                penalty=penalty,
                l1_ratio=l1_ratio,
                device="cuda",
            )
        _assert_contract(
            f"{penalty}/cupy",
            cupy_native,
            backend="cupy",
            device=f"cuda:{device_id}",
        )
        cases[f"{penalty}/cupy"] = {
            "snapshot": _json_snapshot(cupy_native),
            "errors_vs_numpy": _compare(
                f"{penalty}/cupy", reference, cupy_native
            ),
        }

        X_t = torch.as_tensor(X_np, dtype=torch.float64, device=torch_device)
        y_t = torch.as_tensor(y_np, dtype=torch.float64, device=torch_device)
        with torch.cuda.device(torch_device):
            torch_native = _fit_case(
                X_t,
                y_t,
                penalty=penalty,
                l1_ratio=l1_ratio,
                device="torch",
            )
        _assert_contract(
            f"{penalty}/torch",
            torch_native,
            backend="torch",
            device=f"cuda:{device_id}",
        )
        cases[f"{penalty}/torch"] = {
            "snapshot": _json_snapshot(torch_native),
            "errors_vs_numpy": _compare(
                f"{penalty}/torch", reference, torch_native
            ),
        }

        # Heterogeneous input containers must follow requested/executed fit
        # provenance rather than their original container type.
        with cp.cuda.Device(device_id):
            torch_to_cupy = _fit_case(
                X_t,
                y_t,
                penalty=penalty,
                l1_ratio=l1_ratio,
                device="cuda",
            )
        _assert_contract(
            f"{penalty}/torch_to_cupy",
            torch_to_cupy,
            backend="cupy",
            device=f"cuda:{device_id}",
        )
        cases[f"{penalty}/torch_to_cupy"] = {
            "snapshot": _json_snapshot(torch_to_cupy),
            "errors_vs_numpy": _compare(
                f"{penalty}/torch_to_cupy", reference, torch_to_cupy
            ),
        }

        with torch.cuda.device(torch_device):
            cupy_to_torch = _fit_case(
                X_cp,
                y_cp,
                penalty=penalty,
                l1_ratio=l1_ratio,
                device="torch",
            )
        _assert_contract(
            f"{penalty}/cupy_to_torch",
            cupy_to_torch,
            backend="torch",
            device=f"cuda:{device_id}",
        )
        cases[f"{penalty}/cupy_to_torch"] = {
            "snapshot": _json_snapshot(cupy_to_torch),
            "errors_vs_numpy": _compare(
                f"{penalty}/cupy_to_torch", reference, cupy_to_torch
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "validator": "dev/benchmarks/validate_gaussian_residual_bootstrap_gpu.py",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_random_state": BOOTSTRAP_RANDOM_STATE,
        "tolerances": {
            "params_abs": ATOL_PARAMS,
            "bse_abs": ATOL_BSE,
            "pvalues_abs": ATOL_PVALUES,
            "conf_int_abs": ATOL_CONF_INT,
        },
        "environment": environment,
        "child_provenance_gate": (
            "every GPU child must record the parent backend/concrete device "
            "and selected fista solver or the run raises"
        ),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pr147_gaussian_residual_bootstrap_gpu/"
            "pr147_gaussian_residual_bootstrap_gpu.json"
        ),
    )
    args = parser.parse_args()
    payload = run(args.output)
    print(
        json.dumps(
            {"status": payload["status"], "source_sha": payload["source_sha"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
