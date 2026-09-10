"""Physical CUDA acceptance for the public nodewise-alpha inference contract.

Run from a clean checkout of the candidate commit. Both CuPy CUDA and Torch CUDA
are required; the validator fails instead of silently narrowing the backend
matrix. Results are written as schema-v1 JSON for exact-source auditability.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

from statgpu.linear_model import Lasso

SCHEMA_VERSION = 1
RTOL_M = 3e-6
ATOL_M = 3e-8
RTOL_REPORT = 8e-5
ATOL_REPORT = 3e-7


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def _provenance():
    return {
        "schema_version": SCHEMA_VERSION,
        "head_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _dataset(seed=2718, n=120, p=6):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 1))
    X = 0.3 * latent + rng.normal(size=(n, p))
    beta = np.array([1.2, -0.8, 0.55, 0.0, 0.0, 0.0])[:p]
    y = X @ beta + rng.normal(scale=0.35, size=n)
    return X.astype(np.float64), y.astype(np.float64)


def _backend_arrays(name, X, y, w=None):
    if name == "cupy":
        import cupy as cp

        return cp.asarray(X), cp.asarray(y), None if w is None else cp.asarray(w), "cuda"
    if name == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is required")
        device = torch.device("cuda:0")
        return (
            torch.as_tensor(X, dtype=torch.float64, device=device),
            torch.as_tensor(y, dtype=torch.float64, device=device),
            None if w is None else torch.as_tensor(w, dtype=torch.float64, device=device),
            "torch",
        )
    raise ValueError(name)


def _fit(name, X, y, *, nodewise_alpha=None, weight=None, simultaneous=False):
    Xb, yb, wb, device = _backend_arrays(name, X, y, weight)
    model = Lasso(
        alpha=0.05,
        nodewise_alpha=nodewise_alpha,
        device=device,
        solver="fista",
        max_iter=3000,
        tol=1e-7,
        compute_inference=True,
        inference_method="debiased",
        enable_simultaneous_inference=simultaneous,
        simultaneous_method="maxz_bootstrap",
        simultaneous_n_bootstrap=64,
        simultaneous_random_state=11,
        simultaneous_include_intercept=True,
    )
    model.fit(Xb, yb, sample_weight=wb)
    return model


def _assert_close(label, a, b, rtol, atol):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    err = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
    if not np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=False):
        raise AssertionError(f"{label} mismatch: max_abs={err:.3e}")
    return err


def _cpu_reference(X, y, *, nodewise_alpha=None, weight=None):
    return Lasso(
        alpha=0.05,
        nodewise_alpha=nodewise_alpha,
        device="cpu",
        solver="fista",
        max_iter=3000,
        tol=1e-7,
        compute_inference=True,
        inference_method="debiased",
    ).fit(X, y, sample_weight=weight)


def _run_backend(name):
    X, y = _dataset()
    cpu = _cpu_reference(X, y)
    gpu = _fit(name, X, y)
    expected_alpha = float(np.sqrt(2.0 * np.log(X.shape[1]) / X.shape[0]))
    if not np.isclose(gpu.nodewise_alpha_, expected_alpha, rtol=0, atol=1e-14):
        raise AssertionError("automatic nodewise_alpha does not match the declared unweighted rule")
    meta = dict(gpu._inference_result.metadata)
    if meta.get("nodewise_alpha_source") != "auto":
        raise AssertionError("automatic nodewise alpha provenance is missing")
    if meta.get("numerical_backend") != name:
        raise AssertionError(f"wrong numerical backend provenance: {meta.get('numerical_backend')!r}")

    errors = {
        "cpu_gpu_M": _assert_close("cpu_gpu_M", cpu._debiased_M_cpu, gpu._debiased_M_cpu, RTOL_M, ATOL_M),
        "cpu_gpu_params": _assert_close("cpu_gpu_params", cpu._params, gpu._params, RTOL_REPORT, ATOL_REPORT),
        "cpu_gpu_bse": _assert_close("cpu_gpu_bse", cpu._bse, gpu._bse, RTOL_REPORT, ATOL_REPORT),
        "cpu_gpu_p": _assert_close("cpu_gpu_p", cpu._pvalues, gpu._pvalues, 2e-4, 2e-7),
    }

    explicit = _fit(name, X, y, nodewise_alpha=gpu.nodewise_alpha_)
    _assert_close("auto_explicit_M", gpu._debiased_M_cpu, explicit._debiased_M_cpu, 1e-10, 1e-11)
    _assert_close("auto_explicit_params", gpu._params, explicit._params, 1e-10, 1e-11)
    if explicit._inference_result.metadata.get("nodewise_alpha_source") != "user":
        raise AssertionError("explicit nodewise alpha provenance is missing")

    y_scaled = _fit(name, X, 17.0 * y)
    _assert_close("response_scale_M", gpu._debiased_M_cpu, y_scaled._debiased_M_cpu, 1e-10, 1e-11)
    if not np.isclose(gpu.nodewise_alpha_, y_scaled.nodewise_alpha_, rtol=0, atol=1e-14):
        raise AssertionError("response scaling changed automatic nodewise alpha")

    rng = np.random.default_rng(99)
    w = rng.uniform(0.2, 2.0, size=X.shape[0])
    weighted = _fit(name, X, y, weight=w)
    weighted_scaled = _fit(name, X, y, weight=13.0 * w)
    _assert_close("weight_scale_M", weighted._debiased_M_cpu, weighted_scaled._debiased_M_cpu, 3e-7, 3e-9)
    if not np.isclose(weighted.nodewise_alpha_, weighted_scaled.nodewise_alpha_, rtol=0, atol=1e-13):
        raise AssertionError("global weight scaling changed automatic nodewise alpha")

    X1, y1 = _dataset(seed=19, n=80, p=1)
    p1 = _fit(name, X1, y1, nodewise_alpha=0.123)
    if p1.nodewise_alpha_ is not None:
        raise AssertionError("p=1 must not consume nodewise alpha")
    if p1._inference_result.metadata.get("precision_method") != "analytic_univariate":
        raise AssertionError("p=1 did not use analytic precision")

    simultaneous = _fit(name, X, y, simultaneous=True)
    if simultaneous._conf_int_simultaneous is None:
        raise AssertionError("simultaneous inference was not published")
    if not np.all(np.isfinite(np.asarray(simultaneous._conf_int_simultaneous, dtype=float))):
        raise AssertionError("simultaneous intervals are non-finite")
    sim_meta = dict(simultaneous._inference_result.metadata)
    if sim_meta.get("nodewise_alpha") != simultaneous.nodewise_alpha_:
        raise AssertionError("simultaneous result lost nodewise alpha provenance")
    if sim_meta.get("simultaneous_numerical_backend") != name:
        raise AssertionError("simultaneous inference did not remain backend-native")

    return {
        "backend": name,
        "status": "pass",
        "nodewise_alpha": float(gpu.nodewise_alpha_),
        "weighted_nodewise_alpha": float(weighted.nodewise_alpha_),
        "max_kkt_residual": float(meta["nodewise_max_kkt_residual"]),
        "errors": errors,
        "numerical_device": meta.get("numerical_device"),
        "simultaneous_device": sim_meta.get("simultaneous_numerical_device"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/nodewise_alpha_gpu_schema_v1.json")
    args = parser.parse_args()

    out = _provenance()
    out["status"] = "running"
    out["cases"] = []
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cupy as cp
        import torch

        out["cupy_version"] = cp.__version__
        out["torch_version"] = torch.__version__
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy CUDA device is required")
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA device is required")
        out["gpu_name_cupy"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        out["gpu_name_torch"] = torch.cuda.get_device_name(0)

        for backend in ("cupy", "torch"):
            out["cases"].append(_run_backend(backend))
        out["status"] = "success"
    except Exception as exc:
        out["status"] = "failure"
        out["exception_type"] = type(exc).__name__
        out["exception_message"] = str(exc)
        path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        raise

    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
