#!/usr/bin/env python3
"""Exact-head physical CuPy validation for Panel device-affinity contracts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from statgpu import backends
from statgpu._config import _DeviceManager
from statgpu.backends._cupy import CuPyBackend
from statgpu.panel import PooledOLS


SCHEMA_VERSION = 1


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _target_device(device_count: int, current_device: int) -> int:
    device_count = int(device_count)
    current_device = int(current_device)
    if device_count <= 0:
        raise ValueError("physical CuPy validation requires at least one CUDA device")
    if not 0 <= current_device < device_count:
        raise ValueError("current CUDA device is outside the reported device range")
    if device_count == 1:
        return current_device
    return (current_device + 1) % device_count


def _runtime_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_device(value, expected: int, label: str):
    observed = int(value.device.id)
    if observed != int(expected):
        raise AssertionError(
            f"{label} landed on cuda:{observed}, expected cuda:{int(expected)}"
        )


def _probe_device_neutrality(cp, expected_device: int):
    before = int(cp.cuda.runtime.getDevice())
    if before != int(expected_device):
        raise AssertionError(f"unexpected current device before probe: cuda:{before}")

    if not CuPyBackend().is_available():
        raise RuntimeError("CuPyBackend availability probe unexpectedly failed")
    after_backend = int(cp.cuda.runtime.getDevice())
    if after_backend != before:
        raise AssertionError(
            f"CuPyBackend.is_available changed current device: {before} -> {after_backend}"
        )

    manager = _DeviceManager()
    if not manager._check_cupy():
        raise RuntimeError("global CuPy availability probe unexpectedly failed")
    after_global = int(cp.cuda.runtime.getDevice())
    if after_global != before:
        raise AssertionError(
            f"_DeviceManager._check_cupy changed current device: {before} -> {after_global}"
        )

    return {
        "before": f"cuda:{before}",
        "after_backend_probe": f"cuda:{after_backend}",
        "after_global_probe": f"cuda:{after_global}",
    }


def _fit_on_device(cp, target: int):
    rng = np.random.default_rng(20260821)
    X_np = rng.normal(size=(48, 2)).astype(np.float64)
    y_np = (0.4 + 0.7 * X_np[:, 0] - 0.25 * X_np[:, 1]).astype(np.float64)

    with cp.cuda.Device(target):
        X = cp.asarray(X_np)
        y = cp.asarray(y_np)
        _assert_device(X, target, "fit X")
        _assert_device(y, target, "fit y")

        probes = _probe_device_neutrality(cp, target)
        model = PooledOLS(cov_type="hc0", device="cuda").fit(X, y)
        if getattr(model, "_backend_name", None) != "cupy":
            raise AssertionError("PooledOLS did not persist CuPy execution provenance")
        params = getattr(model, "_params", None)
        if params is None or not type(params).__module__.startswith("cupy"):
            raise AssertionError("PooledOLS fit parameters left the CuPy backend")
        _assert_device(params, target, "PooledOLS params")
        after_fit = int(cp.cuda.runtime.getDevice())
        if after_fit != target:
            raise AssertionError(
                f"PooledOLS changed current device: cuda:{target} -> cuda:{after_fit}"
            )

    return X, probes, {
        "executed_backend": "cupy",
        "execution_device": f"cuda:{target}",
        "params_backend_native": True,
        "current_device_after_fit": f"cuda:{after_fit}",
    }


def _cross_current_creation_check(cp, ref, original: int, target: int):
    if int(cp.cuda.runtime.getDevice()) != int(original):
        raise AssertionError("current device was not restored before helper checks")

    ones = backends.xp_ones((3,), cp.float64, cp, ref)
    eye = backends.xp_eye(2, cp.float64, cp, ref)
    _assert_device(ones, target, "xp_ones")
    _assert_device(eye, target, "xp_eye")
    after = int(cp.cuda.runtime.getDevice())
    if after != int(original):
        raise AssertionError(
            f"reference-device helpers leaked current device: {original} -> {after}"
        )
    return {
        "cross_current_exercised": int(original) != int(target),
        "reference_device": f"cuda:{target}",
        "current_device_after_helpers": f"cuda:{after}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    clean_before = _git_clean()
    if not clean_before:
        raise RuntimeError("physical acceptance requires a clean working tree")

    import cupy as cp

    device_count = int(cp.cuda.runtime.getDeviceCount())
    original = int(cp.cuda.runtime.getDevice())
    target = _target_device(device_count, original)

    ref, probes, fit = _fit_on_device(cp, target)
    creation = _cross_current_creation_check(cp, ref, original, target)

    clean_after_checks = _git_clean()
    if not clean_after_checks:
        raise RuntimeError("working tree changed during physical validation")

    gpu_names = {
        f"cuda:{index}": (
            lambda name: name.decode() if isinstance(name, bytes) else str(name)
        )(cp.cuda.runtime.getDeviceProperties(index)["name"])
        for index in range(device_count)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "validation_tier": "remote-full",
        "git_sha": sha,
        "working_tree_clean_before": clean_before,
        "working_tree_clean_after_checks": clean_after_checks,
        "device_count": device_count,
        "original_device": f"cuda:{original}",
        "target_device": f"cuda:{target}",
        "availability_probe": probes,
        "fit": fit,
        "reference_creation": creation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cupy_distribution": _runtime_version("cupy-cuda12x")
            or _runtime_version("cupy-cuda11x")
            or _runtime_version("cupy"),
            "gpu_by_device": gpu_names,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Panel CuPy device-affinity validation: {args.out}")


if __name__ == "__main__":
    main()
