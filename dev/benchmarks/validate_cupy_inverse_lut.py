#!/usr/bin/env python3
"""Physical CuPy validation for Issue #120.

This is a correctness/provenance gate, not a performance benchmark. It checks:

- raw inverse-beta/inverse-gamma LUT paths;
- public CuPy PPF/ISF surfaces that consume those inverse primitives;
- LUT and native-fallback regions for Student-t quantiles;
- backward-compatible R-style inverse-quantile aliases;
- the shared panel-inference consumer that originally exposed the defect.

The script requires an exact source SHA and a clean working tree.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import special, stats

from statgpu.inference._distributions_backend import (
    CuPySpecialFunctions,
    get_distribution,
)


# benchmark_distributions.py treats max absolute PPF/ISF error <1e-6 as PASS;
# the LUT-backed inverse special functions are expected to be approximately
# 1e-7 accurate rather than exact at symmetry points such as t.ppf(0.5).
_INVERSE_ABS_ACCURACY = 1e-6


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


def _public_inverse_check(cp, name, kwargs, scipy_dist, probability, rtol, atol):
    dist = get_distribution(name, backend="cupy", use_lut=True)

    actual_ppf = cp.asnumpy(dist.ppf(probability, **kwargs))
    expected_ppf = np.asarray(scipy_dist.ppf(cp.asnumpy(probability), **kwargs))
    actual_isf = cp.asnumpy(dist.isf(probability, **kwargs))
    expected_isf = np.asarray(scipy_dist.isf(cp.asnumpy(probability), **kwargs))

    np.testing.assert_allclose(actual_ppf, expected_ppf, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_isf, expected_isf, rtol=rtol, atol=atol)

    cdf_roundtrip = cp.asnumpy(dist.cdf(cp.asarray(actual_ppf), **kwargs))
    sf_roundtrip = cp.asnumpy(dist.sf(cp.asarray(actual_isf), **kwargs))
    probability_np = cp.asnumpy(probability)
    np.testing.assert_allclose(cdf_roundtrip, probability_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(sf_roundtrip, probability_np, rtol=rtol, atol=atol)

    return {
        "ppf_max_abs": _max_abs(actual_ppf, expected_ppf),
        "isf_max_abs": _max_abs(actual_isf, expected_isf),
        "cdf_ppf_roundtrip_max_abs": _max_abs(cdf_roundtrip, probability_np),
        "sf_isf_roundtrip_max_abs": _max_abs(sf_roundtrip, probability_np),
    }


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

    # ------------------------------------------------------------------
    # Raw inverse-special-function regression checks.
    # ------------------------------------------------------------------
    beta_probability = cp.asarray(
        [0.01, 0.05, 0.25, 0.50, 0.90], dtype=cp.float64
    )
    beta_actual = cp.asnumpy(sf.betaincinv(22.5, 0.5, beta_probability))
    beta_expected = special.betaincinv(
        22.5, 0.5, cp.asnumpy(beta_probability)
    )
    np.testing.assert_allclose(
        beta_actual, beta_expected, rtol=args.rtol, atol=args.atol
    )

    # Call again to force the already-populated cache path.
    beta_cached = cp.asnumpy(sf.betaincinv(22.5, 0.5, beta_probability))
    np.testing.assert_allclose(
        beta_cached, beta_expected, rtol=args.rtol, atol=args.atol
    )

    gamma_probability = cp.asarray(
        [0.01, 0.20, 0.50, 0.95], dtype=cp.float64
    )
    gamma_actual = cp.asnumpy(sf.gammaincinv(4.0, gamma_probability))
    gamma_expected = special.gammaincinv(
        4.0, cp.asnumpy(gamma_probability)
    )
    np.testing.assert_allclose(
        gamma_actual, gamma_expected, rtol=args.rtol, atol=args.atol
    )
    gamma_cached = cp.asnumpy(sf.gammaincinv(4.0, gamma_probability))
    np.testing.assert_allclose(
        gamma_cached, gamma_expected, rtol=args.rtol, atol=args.atol
    )

    # ------------------------------------------------------------------
    # Exact PR119 failure case.
    # ------------------------------------------------------------------
    t_dist = get_distribution("t", backend="cupy", use_lut=True)
    critical_actual = float(cp.asnumpy(t_dist.isf(cp.asarray(0.025), 45)))
    critical_expected = float(stats.t.isf(0.025, 45))
    np.testing.assert_allclose(
        critical_actual, critical_expected, rtol=args.rtol, atol=args.atol
    )
    if critical_actual <= 1.9:
        raise AssertionError(
            f"collapsed t critical value: {critical_actual} "
            f"(expected {critical_expected})"
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

    # ------------------------------------------------------------------
    # Public inverse-distribution blast radius.
    # The exact Student-t median is tracked separately below because the LUT
    # implementation has an established approximate inverse accuracy contract.
    # ------------------------------------------------------------------
    probability = cp.asarray(
        [0.01, 0.025, 0.20, 0.40, 0.60, 0.95, 0.975], dtype=cp.float64
    )
    public_checks = {
        "t": _public_inverse_check(
            cp, "t", {"df": 45}, stats.t, probability, args.rtol, args.atol
        ),
        "beta": _public_inverse_check(
            cp,
            "beta",
            {"a": 2.5, "b": 5.5},
            stats.beta,
            probability,
            args.rtol,
            args.atol,
        ),
        "f": _public_inverse_check(
            cp,
            "f",
            {"dfn": 5, "dfd": 24},
            stats.f,
            probability,
            args.rtol,
            args.atol,
        ),
        "gamma": _public_inverse_check(
            cp,
            "gamma",
            {"a": 4.0},
            stats.gamma,
            probability,
            args.rtol,
            args.atol,
        ),
        "chi2": _public_inverse_check(
            cp,
            "chi2",
            {"df": 8},
            stats.chi2,
            probability,
            args.rtol,
            args.atol,
        ),
    }

    # Student-t values on both sides of the inverse-beta LUT eligibility region.
    # Strict SciPy equality checks stay away from the exact zero median; median
    # residual and tail antisymmetry are checked against the documented 1e-6
    # inverse-quantile accuracy contract instead.
    t_boundary_checks = {}
    t_probability = cp.asarray([0.025, 0.25, 0.75, 0.975], dtype=cp.float64)
    for df in (1.0, 10.0, 45.0, 60.0, 80.0):
        actual_ppf = cp.asnumpy(t_dist.ppf(t_probability, df))
        expected_ppf = stats.t.ppf(cp.asnumpy(t_probability), df)
        actual_isf = cp.asnumpy(t_dist.isf(t_probability, df))
        expected_isf = stats.t.isf(cp.asnumpy(t_probability), df)
        np.testing.assert_allclose(
            actual_ppf, expected_ppf, rtol=args.rtol, atol=args.atol
        )
        np.testing.assert_allclose(
            actual_isf, expected_isf, rtol=args.rtol, atol=args.atol
        )

        median = float(cp.asnumpy(t_dist.ppf(cp.asarray(0.5), df)))
        if abs(median) >= _INVERSE_ABS_ACCURACY:
            raise AssertionError(
                f"t.ppf(0.5, df={df}) residual {median} exceeds "
                f"inverse accuracy contract {_INVERSE_ABS_ACCURACY}"
            )
        tails = cp.asnumpy(
            t_dist.ppf(cp.asarray([0.025, 0.975], dtype=cp.float64), df)
        )
        tail_symmetry_abs = abs(float(tails[0] + tails[1]))
        if tail_symmetry_abs >= _INVERSE_ABS_ACCURACY:
            raise AssertionError(
                f"t tail antisymmetry residual {tail_symmetry_abs} at df={df} "
                f"exceeds inverse accuracy contract {_INVERSE_ABS_ACCURACY}"
            )

        t_boundary_checks[str(df)] = {
            "ppf_max_abs": _max_abs(actual_ppf, expected_ppf),
            "isf_max_abs": _max_abs(actual_isf, expected_isf),
            "median_abs": abs(median),
            "tail_symmetry_abs": tail_symmetry_abs,
        }

    # ------------------------------------------------------------------
    # Backward-compatible inverse-quantile aliases. A CuPy q array forces
    # the module-level DistributionProxy to resolve the CuPy backend. Shape
    # parameters are passed by keyword because that is the current public
    # DistributionProxy contract.
    # ------------------------------------------------------------------
    from statgpu.linear_model.legacy import _distributions_legacy_gpu as legacy

    legacy_probability = cp.asarray(
        [0.025, 0.25, 0.75, 0.975], dtype=cp.float64
    )
    legacy_probability_np = cp.asnumpy(legacy_probability)
    legacy_specs = [
        (
            "qt_gpu",
            lambda q: legacy.qt_gpu(q, 45),
            stats.t.ppf(legacy_probability_np, df=45),
        ),
        (
            "qbeta_gpu",
            lambda q: legacy.qbeta_gpu(q, a=2.5, b=5.5),
            stats.beta.ppf(legacy_probability_np, a=2.5, b=5.5),
        ),
        (
            "qf_gpu",
            lambda q: legacy.qf_gpu(q, dfn=5, dfd=24),
            stats.f.ppf(legacy_probability_np, dfn=5, dfd=24),
        ),
        (
            "qgamma_gpu",
            lambda q: legacy.qgamma_gpu(q, a=4.0),
            stats.gamma.ppf(legacy_probability_np, a=4.0),
        ),
        (
            "qchisq_gpu",
            lambda q: legacy.qchisq_gpu(q, df=8),
            stats.chi2.ppf(legacy_probability_np, df=8),
        ),
    ]
    legacy_checks = {}
    for label, func, expected in legacy_specs:
        actual = cp.asnumpy(func(legacy_probability))
        np.testing.assert_allclose(
            actual, expected, rtol=args.rtol, atol=args.atol
        )
        legacy_checks[label] = {"max_abs": _max_abs(actual, expected)}

    # ------------------------------------------------------------------
    # Actual shared panel-inference consumer using real CuPy arrays.
    # ------------------------------------------------------------------
    from statgpu.panel._utils import compute_panel_inference

    rng = np.random.default_rng(120)
    n, k = 48, 3
    X_np = rng.normal(size=(n, k))
    params_np = np.asarray([0.35, 1.05, -0.60], dtype=np.float64)
    resid_np = rng.normal(scale=0.25, size=n)
    scale = float(np.sum(resid_np ** 2) / 45.0)
    panel_model = SimpleNamespace()
    compute_panel_inference(
        panel_model,
        cp.asarray(X_np),
        cp.asarray(resid_np),
        cp.asarray(params_np),
        scale,
        n,
        k,
        cp,
        "cupy",
        "nonrobust",
        0.05,
        dist_df=45,
    )
    panel_expected_ci = np.column_stack(
        [
            panel_model.coef_ - critical_expected * panel_model.bse_,
            panel_model.coef_ + critical_expected * panel_model.bse_,
        ]
    )
    if not np.all(panel_model.conf_int_[:, 1] > panel_model.conf_int_[:, 0]):
        raise AssertionError(
            f"shared panel inference produced invalid CI: {panel_model.conf_int_}"
        )
    np.testing.assert_allclose(
        panel_model.conf_int_,
        panel_expected_ci,
        rtol=args.rtol,
        atol=args.atol,
    )

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
        "working_tree_clean": True,
        "status": "success",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu_name,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "packages": {
                "statgpu": _version("statgpu"),
                "numpy": _version("numpy"),
                "scipy": _version("scipy"),
                "cupy": cp.__version__,
            },
        },
        "tolerances": {
            "rtol": args.rtol,
            "atol": args.atol,
            "inverse_abs_accuracy_contract": _INVERSE_ABS_ACCURACY,
        },
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
            "public_inverse_distributions": public_checks,
            "t_lut_fallback_boundary": t_boundary_checks,
            "legacy_inverse_aliases": legacy_checks,
            "shared_panel_ci_max_abs": _max_abs(
                panel_model.conf_int_, panel_expected_ci
            ),
            "shared_panel_ci": panel_model.conf_int_.tolist(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — CuPy inverse-LUT physical validation: {args.out}")


if __name__ == "__main__":
    main()
