#!/usr/bin/env python3
"""Physical CuPy/Torch acceptance for issue #127 Gaussian inference.

The runner is intentionally strict: canonical evidence requires both GPU
backends, exact source identity, a tracked-clean tree, explicit validation tier,
and machine-auditable numerical-inference provenance.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from statgpu.linear_model import PenalizedGeneralizedLinearModel, RidgeCV
from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
)

SCHEMA_VERSION = 2
_REQUIRED_BACKENDS = ("cupy", "torch")
_VALIDATION_TIERS = ("local-minimal", "local-full", "remote-full")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _tracked_clean() -> bool:
    return _git("status", "--porcelain", "--untracked-files=no") == ""


def _validate_backend_list(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if values != _REQUIRED_BACKENDS:
        raise ValueError(
            "--backends must be exactly 'cupy,torch' in that order for canonical acceptance"
        )
    return values


def _gpu_model() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        return output[0].strip() if output else "unknown"
    except Exception:
        return "unknown"


def _backend_runtime(backend: str):
    if backend == "cupy":
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("CuPy imported but no CUDA device is available")
        device_id = int(cp.cuda.runtime.getDevice())
        return cp, f"cuda:{device_id}", cp.__version__

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch imported but CUDA is unavailable")
    device_id = int(torch.cuda.current_device())
    return torch, f"cuda:{device_id}", torch.__version__


def _as_backend(value, backend: str):
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(value, dtype=cp.float64)
    import torch

    return torch.as_tensor(
        np.asarray(value),
        dtype=torch.float64,
        device=f"cuda:{torch.cuda.current_device()}",
    )


def _assert_same_native_device(value, backend: str, expected_device: str):
    if backend == "cupy":
        import cupy as cp

        if not isinstance(value, cp.ndarray):
            raise AssertionError(f"expected CuPy ndarray, got {type(value)!r}")
        actual = f"cuda:{int(value.device.id)}"
    else:
        import torch

        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"expected Torch tensor, got {type(value)!r}")
        actual = str(value.device)
    if actual != expected_device:
        raise AssertionError(f"device mismatch: expected {expected_device}, got {actual}")


def _problem(seed: int = 127):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, 5))
    beta = np.asarray([0.8, -0.35, 0.2, 0.55, -0.1])
    y = -0.4 + X @ beta + rng.normal(scale=0.3, size=X.shape[0])
    weights = np.linspace(0.5, 1.75, X.shape[0])
    return X, y, weights


def _max_error(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _fit_pglm(X, y, *, device: str, cov_type: str, sample_weight=None):
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.05,
        fit_intercept=True,
        device=device,
        solver="fista",
        max_iter=5000,
        tol=1e-10,
        compute_inference=True,
        cov_type=cov_type,
        hac_maxlags=2,
    )
    model.fit(X, y, sample_weight=sample_weight)
    if model._inference_result is None:
        raise AssertionError("PGLM L2 inference result is missing")
    return model


def _consumer_case(backend: str, cov_type: str, weighted: bool):
    X, y, weights = _problem()
    sw = weights if weighted else None
    cpu = _fit_pglm(X, y, device="cpu", cov_type=cov_type, sample_weight=sw)

    X_gpu = _as_backend(X, backend)
    y_gpu = _as_backend(y, backend)
    sw_gpu = None if sw is None else _as_backend(sw, backend)
    device_arg = "cuda" if backend == "cupy" else "torch"
    gpu = _fit_pglm(
        X_gpu,
        y_gpu,
        device=device_arg,
        cov_type=cov_type,
        sample_weight=sw_gpu,
    )

    expected_backend = backend
    actual_backend = str(getattr(gpu, "_selected_backend_name", "")).lower()
    if actual_backend != expected_backend:
        raise AssertionError(
            f"fit backend mismatch: requested={backend}, executed={actual_backend}"
        )
    meta = dict(gpu._inference_result.metadata)
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"inference backend mismatch: {meta}")
    if not str(meta.get("numerical_device", "")).startswith("cuda:"):
        raise AssertionError(f"inference device is not concrete CUDA: {meta}")
    if meta.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"reporting boundary metadata missing: {meta}")

    errors = {
        "coef": _max_error(gpu.coef_, cpu.coef_),
        "bse": _max_error(gpu._bse, cpu._bse),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    limits = {"coef": 2e-6, "bse": 2e-5, "pvalue": 2e-4, "ci": 5e-5}
    for key, limit in limits.items():
        if not np.isfinite(errors[key]) or errors[key] > limit:
            raise AssertionError(
                f"{backend} {cov_type} weighted={weighted} {key} error "
                f"{errors[key]:.3e} exceeds {limit:.3e}"
            )
    return {
        "case": f"pglm_{cov_type}_{'weighted' if weighted else 'unweighted'}",
        "requested_backend": backend,
        "executed_backend": actual_backend,
        "executed_inference_backend": meta.get("numerical_backend"),
        "executed_inference_device": meta.get("numerical_device"),
        "reporting_backend": meta.get("reporting_backend"),
        "errors": errors,
        "limits": limits,
        "status": "success",
    }


def _functional_edge_cases(backend: str, concrete_device: str):
    # Rank-deficient native inference.
    x = np.arange(1.0, 9.0)
    design_np = np.column_stack([np.ones_like(x), x, x])
    params_np = np.asarray([0.5, 0.25, 0.25])
    resid_np = np.asarray([0.2, -0.1, 0.05, -0.15, 0.1, -0.1, 0.05, -0.05])
    design = _as_backend(design_np, backend)
    params = _as_backend(params_np, backend)
    resid = _as_backend(resid_np, backend)
    _assert_same_native_device(design, backend, concrete_device)
    result = compute_gaussian_inference(
        design,
        params,
        resid,
        float(np.sum(resid_np**2) / (len(x) - 3)),
        len(x) - 3,
        "nonrobust",
        backend=backend,
        device=concrete_device,
    )
    if result.metadata.get("numerical_device") != concrete_device:
        raise AssertionError("rank-deficient case lost concrete inference device")

    # Multi-target numerical state must stay native until the one final snapshot.
    X, y, _ = _problem(seed=128)
    coef = np.column_stack([
        np.asarray([0.8, -0.35, 0.2, 0.55, -0.1]),
        np.asarray([-0.2, 0.4, 0.1, -0.3, 0.25]),
    ])
    intercept = np.asarray([-0.4, 0.2])
    y2 = np.column_stack([y, intercept[1] + X @ coef[:, 1] + 0.05])
    state = build_gaussian_fit_state(
        _as_backend(X, backend),
        _as_backend(y2, backend),
        coef,
        intercept,
        True,
        backend=backend,
        device=concrete_device,
    )
    _assert_same_native_device(state.X_design, backend, concrete_device)
    multi = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "hc3",
        backend=backend,
        device=concrete_device,
    )
    if multi.metadata.get("numerical_device") != concrete_device:
        raise AssertionError("multi-target case lost concrete inference device")
    if tuple(multi.bse.shape) != tuple(state.params.shape):
        raise AssertionError("multi-target inference shape mismatch")

    return {
        "case": "functional_rank_and_multitarget",
        "requested_backend": backend,
        "executed_inference_backend": multi.metadata.get("numerical_backend"),
        "executed_inference_device": multi.metadata.get("numerical_device"),
        "rank_status": "success",
        "multi_target_status": "success",
        "status": "success",
    }


def _ridgecv_case(backend: str):
    X, y, _ = _problem(seed=129)
    alphas = np.asarray([0.01, 0.1, 1.0])
    cpu = RidgeCV(
        alphas=alphas,
        cv=3,
        device="cpu",
        compute_inference=True,
        random_state=129,
    ).fit(X, y)
    device_arg = "cuda" if backend == "cupy" else "torch"
    gpu = RidgeCV(
        alphas=alphas,
        cv=3,
        device=device_arg,
        compute_inference=True,
        random_state=129,
    ).fit(_as_backend(X, backend), _as_backend(y, backend))

    if float(gpu.alpha_) != float(cpu.alpha_):
        raise AssertionError(
            f"RidgeCV selected alpha differs: gpu={gpu.alpha_}, cpu={cpu.alpha_}"
        )
    estimator_backend = str(
        getattr(gpu.estimator_, "_selected_backend_name", "")
    ).lower()
    if estimator_backend != backend:
        raise AssertionError(
            f"RidgeCV final-refit backend mismatch: {estimator_backend}"
        )
    result = gpu.estimator_._inference_result
    if result is None:
        raise AssertionError("RidgeCV final-refit inference is missing")
    if result.metadata.get("numerical_backend") != backend:
        raise AssertionError(
            f"RidgeCV inference backend mismatch: {result.metadata}"
        )
    coef_error = _max_error(gpu.coef_, cpu.coef_)
    if coef_error > 2e-6:
        raise AssertionError(f"RidgeCV coefficient error {coef_error:.3e} exceeds 2e-6")
    return {
        "case": "ridgecv_final_refit_inference",
        "requested_backend": backend,
        "executed_backend": estimator_backend,
        "executed_inference_backend": result.metadata.get("numerical_backend"),
        "selected_alpha": float(gpu.alpha_),
        "coef_error": coef_error,
        "status": "success",
    }


def run_backend(backend: str):
    _, concrete_device, runtime_version = _backend_runtime(backend)
    cases = []
    for cov_type in ("nonrobust", "hc3", "hac"):
        cases.append(_consumer_case(backend, cov_type, weighted=False))
    cases.append(_consumer_case(backend, "nonrobust", weighted=True))
    cases.append(_functional_edge_cases(backend, concrete_device))
    cases.append(_ridgecv_case(backend))
    return {
        "backend": backend,
        "concrete_device": concrete_device,
        "runtime_version": runtime_version,
        "cases": cases,
        "status": "success",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--validation-tier", required=True, choices=_VALIDATION_TIERS
    )
    parser.add_argument("--backends", required=True)
    args = parser.parse_args(argv)

    backends = _validate_backend_list(args.backends)
    git_sha = _git("rev-parse", "HEAD")
    if git_sha != args.expected_sha:
        raise RuntimeError(
            f"exact-source gate failed: HEAD={git_sha}, expected={args.expected_sha}"
        )
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean before physical validation")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
        "validation_tier": args.validation_tier,
        "working_tree_clean": True,
        "python_version": platform.python_version(),
        "gpu_model": _gpu_model(),
        "required_backends": list(_REQUIRED_BACKENDS),
        "results": [],
        "status": "running",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for backend in backends:
            artifact["results"].append(run_backend(backend))
        artifact["status"] = "success"
    except Exception as exc:
        artifact["status"] = "failure"
        artifact["error_type"] = type(exc).__name__
        artifact["error"] = str(exc)
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        raise

    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
