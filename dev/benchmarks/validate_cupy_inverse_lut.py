#!/usr/bin/env python3
"""Physical CuPy validation for Issue #120.

This is a correctness/provenance gate, not a performance benchmark.  It checks
the inverse-beta/inverse-gamma LUT paths on a real CuPy CUDA backend and
reproduces the confidence-interval calculation that blocked PR #119.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import special, stats

from statgpu.inference._distributions_backend import (
    CuPySpecialFunctions,
    get_distribution,
)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _working_tree_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _max_abs(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--rtol", type=float, default=5e-8)
    parser.add_argument("--atol", type=float, default=5e-9)
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    dirty = _working_tree_status()
    if dirty:
        raise RuntimeError(f"working tree is not clean:\n{dirty}")

    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy CUDA validation requested but no CUDA device is available")

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    sf = CuPySpecialFunctions(use_lut=True)

    beta_probability = cp.asarray([0.01, 0.05, 0.25, 0.50, 0.90], dtype=cp.float64)
    beta_actual = cp.asnumpy(sf.betaincinv(22.5, 0.5, beta_probability))
    beta_expected = special.betaincinv(22.5, 0.5, cp.asnumpy(beta_probability))
    np.testing.assert_allclose(
        beta_actual, beta_expected, rtol=args.rtol, atol=args.atol
    )

    # Call again to force the already-populated cache path.
    beta_cached = cp.asnumpy(sf.betaincinv(22.5, 0.5, beta_probability))
    np.testing.assert_allclose(
        beta_cached, beta_expected, rtol=args.rtol, atol=args.atol
    )

    gamma_probability = cp.asarray([0.01, 0.20, 0.50, 0.95], dtype=cp.float64)
    gamma_actual = cp.asnumpy(sf.gammaincinv(4.0, gamma_probability))
    gamma_expected = special.gammaincinv(4.0, cp.asnumpy(gamma_probability))
    np.testing.assert_allclose(
        gamma_actual, gamma_expected, rtol=args.rtol, atol=args.atol
    )
    gamma_cached = cp.asnumpy(sf.gammaincinv(4.0, gamma_probability))
    np.testing.assert_allclose(
        gamma_cached, gamma_expected, rtol=args.rtol, atol=args.atol
    )

    t_dist = get_distribution("t", backend="cupy")
    critical_actual = float(cp.asnumpy(t_dist.isf(cp.asarray(0.025), 45)))
    critical_expected = float(stats.t.isf(0.025, 45))
    np.testing.assert_allclose(
        critical_actual, critical_expected, rtol=args.rtol, atol=args.atol
    )
    if critical_actual <= 1.9:
        raise AssertionError(
            f"collapsed t critical value: {critical_actual} (expected {critical_expected})"
        )

    coef = np.asarray([0.36067077, 1.06308319, -0.61715928])
    bse = np.asarray([0.06958899, 0.08102734, 0.08449355])
    ci_actual = np.column_stack(
        [coef - critical_actual * bse, coef + critical_actual * bse]
    )
    ci_expected = np.column_stack(
        [coef - critical_expected * bse, coef + critical_expected * bse]
    )
    if not np.all(ci_actual[:, 1] > ci_actual[:, 0]):
        raise AssertionError(f"zero-width or reversed confidence interval: {ci_actual}")
    np.testing.assert_allclose(
        ci_actual, ci_expected, rtol=args.rtol, atol=args.atol
    )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
        "working_tree_clean": True,
        "status": "success",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu_name,
            "packages": {
                name: _version(name)
                for name in ("statgpu", "numpy", "scipy", "cupy")
            },
        },
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "checks": {
            "betaincinv_max_abs": _max_abs(beta_actual, beta_expected),
            "betaincinv_cached_max_abs": _max_abs(beta_cached, beta_expected),
            "gammaincinv_max_abs": _max_abs(gamma_actual, gamma_expected),
            "gammaincinv_cached_max_abs": _max_abs(gamma_cached, gamma_expected),
            "t_critical_actual": critical_actual,
            "t_critical_expected": critical_expected,
            "t_critical_abs_diff": abs(critical_actual - critical_expected),
            "panel_ci_max_abs": _max_abs(ci_actual, ci_expected),
            "panel_ci": ci_actual.tolist(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — CuPy inverse-LUT physical validation: {args.out}")


if __name__ == "__main__":
    main()
