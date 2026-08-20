#!/usr/bin/env python3
"""Exact-head CuPy/Torch CUDA validation for the Fama-MacBeth t(2) tail fix."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _is_cupy_array, _is_torch_array, _to_numpy
from statgpu.inference._reference_distribution import two_sided_reference_inference

SCHEMA_VERSION = 1
_REQUIRED_BACKENDS = {"cupy", "torch"}
_EXTREME_STATISTIC = 1.0e154


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_acceptance_backends(backends):
    normalized = [value.strip() for value in backends if value.strip()]
    if len(normalized) != 2 or set(normalized) != _REQUIRED_BACKENDS:
        raise ValueError(
            "physical acceptance requires exactly both GPU backends: cupy,torch"
        )
    return normalized


def _expected_tail(statistic_value: float) -> float:
    root = np.hypot(float(statistic_value), np.sqrt(2.0))
    return (2.0 / root) / (root + float(statistic_value))


def _backend_case(backend: str):
    if backend == "cupy":
        import cupy as xp

        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy CUDA is unavailable")
        statistic = xp.asarray([_EXTREME_STATISTIC], dtype=xp.float64)
        device = None
    elif backend == "torch":
        import torch as xp

        if not xp.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable")
        statistic = xp.as_tensor(
            [_EXTREME_STATISTIC], dtype=xp.float64, device="cuda"
        )
        device = "cuda"
    else:
        raise ValueError(f"unsupported backend: {backend}")

    pvalues, critical = two_sided_reference_inference(
        statistic,
        distribution="t",
        alpha=0.05,
        backend=backend,
        xp=xp,
        df=2,
        device=device,
    )
    if backend == "cupy":
        native = _is_cupy_array(pvalues) and _is_cupy_array(critical)
    else:
        native = _is_torch_array(pvalues) and _is_torch_array(critical)
        native = native and str(pvalues.device).startswith("cuda")
        native = native and str(critical.device).startswith("cuda")
    if not native:
        raise AssertionError(f"{backend}: t(2) inference left the requested CUDA backend")

    observed = float(np.asarray(_to_numpy(pvalues), dtype=np.float64)[0])
    expected = _expected_tail(_EXTREME_STATISTIC)
    if not np.isfinite(observed) or observed <= 0.0:
        raise AssertionError(
            f"{backend}: representable extreme t(2) tail collapsed to {observed!r}"
        )
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=0.0)
    critical_value = float(np.asarray(_to_numpy(critical), dtype=np.float64))
    if not np.isfinite(critical_value) or critical_value <= 0.0:
        raise AssertionError(f"{backend}: invalid t(2) critical value {critical_value!r}")

    return {
        "status": "success",
        "executed_backend": backend,
        "statistic": float(_EXTREME_STATISTIC),
        "pvalue": observed,
        "expected_pvalue": expected,
        "critical_value": critical_value,
        "pvalue_nonzero": True,
        "backend_native": True,
    }


def _environment(backends):
    gpu_by_backend = {}
    cupy_version = None
    if "cupy" in backends:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        gpu_by_backend["cupy"] = name.decode() if isinstance(name, bytes) else name
        cupy_version = cp.__version__
    if "torch" in backends:
        import torch

        gpu_by_backend["torch"] = torch.cuda.get_device_name(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "cupy": cupy_version,
        "torch": _version("torch"),
        "gpu_by_backend": gpu_by_backend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    clean_before = _git_clean()
    if not clean_before:
        raise RuntimeError("physical acceptance requires a clean working tree")

    backends = _validate_acceptance_backends(args.backends.split(","))
    results = {backend: _backend_case(backend) for backend in backends}

    clean_after_checks = _git_clean()
    if not clean_after_checks:
        raise RuntimeError("working tree changed during physical validation")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": sha,
        "required_backends": sorted(_REQUIRED_BACKENDS),
        "validated_backends": backends,
        "working_tree_clean_before": clean_before,
        "working_tree_clean_after_checks": clean_after_checks,
        "status": "success",
        "validation_tier": "remote-full",
        "statistic": float(_EXTREME_STATISTIC),
        "environment": _environment(backends),
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Fama-MacBeth extreme t(2) CUDA validation: {args.out}")


if __name__ == "__main__":
    main()
