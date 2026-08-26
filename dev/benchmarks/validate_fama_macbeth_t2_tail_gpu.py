#!/usr/bin/env python3
"""Exact-head CuPy/Torch CUDA validation for the Fama-MacBeth t(2) tail fix."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from statgpu.backends import _is_cupy_array, _is_torch_array, _to_numpy
from statgpu.inference._reference_distribution import two_sided_reference_inference

SCHEMA_VERSION = 2
_REQUIRED_BACKENDS = {"cupy", "torch"}
_EXTREME_STATISTIC = 1.0e154


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


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


def _cuda_device_label(value, backend: str):
    if backend == "cupy":
        device = getattr(value, "device", None)
        device_id = getattr(device, "id", None)
        return None if device_id is None else f"cuda:{int(device_id)}"
    if backend == "torch":
        device = getattr(value, "device", None)
        return None if device is None else str(device)
    raise ValueError(f"unsupported backend: {backend}")


def _cuda_device_index(label: str) -> int:
    prefix, separator, suffix = str(label).partition(":")
    if prefix != "cuda" or separator != ":" or not suffix.isdigit():
        raise ValueError(f"invalid concrete CUDA device label: {label!r}")
    return int(suffix)


def _assert_cuda_native_and_same_device(statistic, pvalues, critical, backend: str):
    if backend == "cupy":
        native = all(
            _is_cupy_array(value) for value in (statistic, pvalues, critical)
        )
    elif backend == "torch":
        native = all(
            _is_torch_array(value) for value in (statistic, pvalues, critical)
        )
    else:
        raise ValueError(f"unsupported backend: {backend}")
    if not native:
        raise AssertionError(f"{backend}: t(2) inference left the requested CUDA backend")

    labels = [
        _cuda_device_label(value, backend)
        for value in (statistic, pvalues, critical)
    ]
    try:
        indices = [_cuda_device_index(label) for label in labels]
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"{backend}: t(2) inference produced a non-CUDA device trace {labels}"
        ) from exc
    if len(set(indices)) != 1:
        raise AssertionError(
            f"{backend}: t(2) inference crossed CUDA devices {labels}"
        )
    return f"cuda:{indices[0]}"


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
        device = str(statistic.device)
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
    execution_device = _assert_cuda_native_and_same_device(
        statistic, pvalues, critical, backend
    )

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
        "execution_device": execution_device,
        "statistic": float(_EXTREME_STATISTIC),
        "pvalue": observed,
        "expected_pvalue": expected,
        "critical_value": critical_value,
        "pvalue_nonzero": True,
        "backend_native": True,
        "same_cuda_device": True,
    }


def _environment(backends, results):
    gpu_by_backend = {}
    runtime_versions = {"numpy": np.__version__}
    if "cupy" in backends:
        import cupy as cp

        device_index = _cuda_device_index(results["cupy"]["execution_device"])
        props = cp.cuda.runtime.getDeviceProperties(device_index)
        name = props["name"]
        gpu_by_backend["cupy"] = name.decode() if isinstance(name, bytes) else name
        runtime_versions["cupy"] = cp.__version__
    if "torch" in backends:
        import torch

        device_index = _cuda_device_index(results["torch"]["execution_device"])
        gpu_by_backend["torch"] = torch.cuda.get_device_name(device_index)
        runtime_versions["torch"] = torch.__version__
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_versions": runtime_versions,
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
        "environment": _environment(backends, results),
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Fama-MacBeth extreme t(2) CUDA validation: {args.out}")


if __name__ == "__main__":
    main()
