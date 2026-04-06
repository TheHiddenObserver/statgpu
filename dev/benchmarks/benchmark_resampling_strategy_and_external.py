"""Benchmark resampling strategy backends and large-scale external comparisons.

Usage:
    python dev/benchmarks/benchmark_resampling_strategy_and_external.py [--strict-scalar] [--output-tag TAG]

Outputs:
    results/resampling_strategy_external_benchmark_YYYY-MM-DD[_strict_scalar][_TAG].json
"""

from __future__ import annotations

import json
import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statgpu.inference import bootstrap_statistic, permutation_test


try:
    from scipy.stats import bootstrap as scipy_bootstrap
    from scipy.stats import permutation_test as scipy_permutation_test
except Exception:  # pragma: no cover - optional dependency
    scipy_bootstrap = None
    scipy_permutation_test = None


def _maybe_import_cupy():
    try:
        import cupy as cp

        return cp
    except Exception:
        return None


def _bench(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeats: int = 3,
    synchronize: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()
        if synchronize is not None:
            synchronize()

    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "times_ms": [float(x) for x in arr.tolist()],
    }


def _corr_stat_maybe_vectorized(X_in, y_in, xp_module):
    x = X_in[:, 0]
    y_arr = xp_module.asarray(y_in)

    if y_arr.ndim == 1:
        return xp_module.corrcoef(x, y_arr)[0, 1]

    x_centered = x - xp_module.mean(x)
    y_centered = y_arr - xp_module.mean(y_arr, axis=1, keepdims=True)
    x_norm_sq = xp_module.sum(x_centered * x_centered)
    y_norm_sq = xp_module.sum(y_centered * y_centered, axis=1)
    denom = xp_module.sqrt(x_norm_sq * y_norm_sq)
    denom = xp_module.where(denom > 0.0, denom, np.inf if xp_module is np else xp_module.inf)
    numer = xp_module.sum(y_centered * x_centered.reshape(1, -1), axis=1)
    return numer / denom


def _run_strategy_timing(
    cp_module,
    rng: np.random.Generator,
    *,
    strict_scalar: bool = False,
) -> Dict[str, Any]:
    x_np = rng.normal(loc=1.0, scale=1.0, size=600)
    strata_boot = np.repeat(np.arange(6), 100)
    clusters_boot = np.repeat(np.arange(60), 10)

    X_np = rng.normal(size=(500, 1))
    y_np = 1.2 * X_np[:, 0] + rng.normal(scale=0.7, size=500)
    strata_perm = np.repeat(np.arange(5), 100)
    groups_perm = np.repeat(np.arange(50), 10)

    out: Dict[str, Any] = {
        "config": {
            "bootstrap_n": 600,
            "bootstrap_n_resamples": 1200,
            "permutation_n": 500,
            "permutation_n_resamples": 2000,
            "warmup": 1,
            "repeats": 3,
            "cupy_mode": "strict_scalar" if strict_scalar else "optimized_vectorized",
        },
        "bootstrap": {"numpy": {}, "cupy": None},
        "permutation": {"numpy": {}, "cupy": None},
    }

    # Bootstrap strategies - NumPy
    out["bootstrap"]["numpy"]["iid"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr),
            x_np,
            n_resamples=1200,
            strategy="iid",
            random_state=7,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )
    out["bootstrap"]["numpy"]["stratified"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr),
            x_np,
            n_resamples=1200,
            strategy="stratified",
            strata=strata_boot,
            random_state=7,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )
    out["bootstrap"]["numpy"]["cluster"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr),
            x_np,
            n_resamples=1200,
            strategy="cluster",
            clusters=clusters_boot,
            random_state=7,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )
    out["bootstrap"]["numpy"]["block"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr),
            x_np,
            n_resamples=1200,
            strategy="block",
            block_size=20,
            random_state=7,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )

    # Permutation strategies - NumPy
    out["permutation"]["numpy"]["iid"] = _bench(
        lambda: permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X_np,
            y_np,
            n_resamples=2000,
            strategy="iid",
            alternative="two-sided",
            random_state=11,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )
    out["permutation"]["numpy"]["stratified"] = _bench(
        lambda: permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X_np,
            y_np,
            n_resamples=2000,
            strategy="stratified",
            strata=strata_perm,
            alternative="two-sided",
            random_state=11,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )
    out["permutation"]["numpy"]["grouped"] = _bench(
        lambda: permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X_np,
            y_np,
            n_resamples=2000,
            strategy="grouped",
            groups=groups_perm,
            alternative="two-sided",
            random_state=11,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )

    if cp_module is not None:
        x_cp = cp_module.asarray(x_np)
        strata_boot_cp = cp_module.asarray(strata_boot)
        clusters_boot_cp = cp_module.asarray(clusters_boot)

        X_cp = cp_module.asarray(X_np)
        y_cp = cp_module.asarray(y_np)
        strata_perm_cp = cp_module.asarray(strata_perm)
        groups_perm_cp = cp_module.asarray(groups_perm)

        out["bootstrap"]["cupy"] = {}
        out["permutation"]["cupy"] = {}

        if strict_scalar:
            out["bootstrap"]["cupy"]["iid"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="iid",
                    random_state=7,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["bootstrap"]["cupy"]["stratified"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="stratified",
                    strata=strata_boot_cp,
                    random_state=7,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["bootstrap"]["cupy"]["cluster"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="cluster",
                    clusters=clusters_boot_cp,
                    random_state=7,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["bootstrap"]["cupy"]["block"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="block",
                    block_size=20,
                    random_state=7,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

            out["permutation"]["cupy"]["iid"] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: cp_module.corrcoef(X_in[:, 0], y_in)[0, 1],
                    X_cp,
                    y_cp,
                    n_resamples=2000,
                    strategy="iid",
                    alternative="two-sided",
                    random_state=11,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["permutation"]["cupy"]["stratified"] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: cp_module.corrcoef(X_in[:, 0], y_in)[0, 1],
                    X_cp,
                    y_cp,
                    n_resamples=2000,
                    strategy="stratified",
                    strata=strata_perm_cp,
                    alternative="two-sided",
                    random_state=11,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["permutation"]["cupy"]["grouped"] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: cp_module.corrcoef(X_in[:, 0], y_in)[0, 1],
                    X_cp,
                    y_cp,
                    n_resamples=2000,
                    strategy="grouped",
                    groups=groups_perm_cp,
                    alternative="two-sided",
                    random_state=11,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
        else:
            out["bootstrap"]["cupy"]["iid"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="iid",
                    random_state=7,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="mean",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["bootstrap"]["cupy"]["stratified"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="stratified",
                    strata=strata_boot_cp,
                    random_state=7,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="mean",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["bootstrap"]["cupy"]["cluster"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="cluster",
                    clusters=clusters_boot_cp,
                    random_state=7,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="mean",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["bootstrap"]["cupy"]["block"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1200,
                    strategy="block",
                    block_size=20,
                    random_state=7,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="mean",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

            out["permutation"]["cupy"]["iid"] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, cp_module),
                    X_cp,
                    y_cp,
                    n_resamples=2000,
                    strategy="iid",
                    alternative="two-sided",
                    random_state=11,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="pearson_corr",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["permutation"]["cupy"]["stratified"] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, cp_module),
                    X_cp,
                    y_cp,
                    n_resamples=2000,
                    strategy="stratified",
                    strata=strata_perm_cp,
                    alternative="two-sided",
                    random_state=11,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="pearson_corr",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
            out["permutation"]["cupy"]["grouped"] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, cp_module),
                    X_cp,
                    y_cp,
                    n_resamples=2000,
                    strategy="grouped",
                    groups=groups_perm_cp,
                    alternative="two-sided",
                    random_state=11,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="pearson_corr",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

    return out


def _run_large_scale_external(
    cp_module,
    rng: np.random.Generator,
    *,
    strict_scalar: bool = False,
) -> Dict[str, Any]:
    x_np = rng.normal(loc=1.0, scale=1.0, size=30_000)
    X_np = rng.normal(size=(18_000, 1))
    y_np = 1.7 * X_np[:, 0] + rng.normal(scale=0.6, size=18_000)

    # Keep external comparison feasible while preserving large-scale size.
    n_boot = 1200
    n_perm = 1000

    out: Dict[str, Any] = {
        "config": {
            "bootstrap_n": 30_000,
            "bootstrap_n_resamples": n_boot,
            "permutation_n": 18_000,
            "permutation_n_resamples": n_perm,
            "warmup": 1,
            "repeats": 3,
            "mode": "strict_scalar" if strict_scalar else "optimized_vectorized",
            "note": "Large-scale external comparison uses same data size and same resample counts across frameworks.",
        },
        "statgpu": {
            "bootstrap": {},
            "permutation": {},
        },
        "scipy": None,
        "ratios": {},
    }

    boot_numpy_key = "numpy_scalar" if strict_scalar else "numpy_vectorized"
    perm_numpy_key = "numpy_scalar" if strict_scalar else "numpy_vectorized"
    boot_cupy_key = "cupy_scalar" if strict_scalar else "cupy_vectorized"
    perm_cupy_key = "cupy_scalar" if strict_scalar else "cupy_vectorized"

    if strict_scalar:
        out["statgpu"]["bootstrap"][boot_numpy_key] = _bench(
            lambda: bootstrap_statistic(
                lambda arr: np.mean(arr),
                x_np,
                n_resamples=n_boot,
                strategy="iid",
                random_state=17,
                backend="numpy",
            ),
            warmup=1,
            repeats=3,
        )

        out["statgpu"]["permutation"][perm_numpy_key] = _bench(
            lambda: permutation_test(
                lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
                X_np,
                y_np,
                n_resamples=n_perm,
                strategy="iid",
                alternative="two-sided",
                random_state=23,
                backend="numpy",
            ),
            warmup=1,
            repeats=3,
        )
    else:
        out["statgpu"]["bootstrap"][boot_numpy_key] = _bench(
            lambda: bootstrap_statistic(
                lambda arr: np.mean(arr, axis=-1),
                x_np,
                n_resamples=n_boot,
                strategy="iid",
                random_state=17,
                backend="numpy",
                force_vectorized=True,
                statistic_hint="mean",
            ),
            warmup=1,
            repeats=3,
        )

        out["statgpu"]["permutation"][perm_numpy_key] = _bench(
            lambda: permutation_test(
                lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, np),
                X_np,
                y_np,
                n_resamples=n_perm,
                strategy="iid",
                alternative="two-sided",
                random_state=23,
                backend="numpy",
                force_vectorized=True,
            ),
            warmup=1,
            repeats=3,
        )

    if cp_module is not None:
        x_cp = cp_module.asarray(x_np)
        X_cp = cp_module.asarray(X_np)
        y_cp = cp_module.asarray(y_np)

        if strict_scalar:
            out["statgpu"]["bootstrap"][boot_cupy_key] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=n_boot,
                    strategy="iid",
                    random_state=17,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

            out["statgpu"]["permutation"][perm_cupy_key] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: cp_module.corrcoef(X_in[:, 0], y_in)[0, 1],
                    X_cp,
                    y_cp,
                    n_resamples=n_perm,
                    strategy="iid",
                    alternative="two-sided",
                    random_state=23,
                    backend="cupy",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
        else:
            out["statgpu"]["bootstrap"][boot_cupy_key] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr, axis=-1),
                    x_cp,
                    n_resamples=n_boot,
                    strategy="iid",
                    random_state=17,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="mean",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

            out["statgpu"]["permutation"][perm_cupy_key] = _bench(
                lambda: permutation_test(
                    lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, cp_module),
                    X_cp,
                    y_cp,
                    n_resamples=n_perm,
                    strategy="iid",
                    alternative="two-sided",
                    random_state=23,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="pearson_corr",
                ),
                warmup=1,
                repeats=3,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

    if scipy_bootstrap is not None and scipy_permutation_test is not None:
        out["scipy"] = {
            "bootstrap": _bench(
                lambda: scipy_bootstrap(
                    (x_np,),
                    np.mean,
                    n_resamples=n_boot,
                    method="percentile",
                    random_state=17,
                    vectorized=False,
                ),
                warmup=1,
                repeats=3,
            ),
            "permutation": _bench(
                lambda: scipy_permutation_test(
                    (X_np[:, 0], y_np),
                    lambda a, b: np.corrcoef(a, b)[0, 1],
                    permutation_type="pairings",
                    n_resamples=n_perm,
                    alternative="two-sided",
                    random_state=23,
                    vectorized=False,
                ),
                warmup=1,
                repeats=3,
            ),
        }

    b_np = out["statgpu"]["bootstrap"][boot_numpy_key]["mean_ms"]
    p_np = out["statgpu"]["permutation"][perm_numpy_key]["mean_ms"]
    out["ratios"]["statgpu_numpy_bootstrap_vs_permutation"] = float(b_np / p_np) if p_np > 0 else None

    b_cp_entry = out["statgpu"]["bootstrap"].get(boot_cupy_key)
    p_cp_entry = out["statgpu"]["permutation"].get(perm_cupy_key)
    if b_cp_entry is not None and p_cp_entry is not None:
        b_cp = b_cp_entry["mean_ms"]
        p_cp = p_cp_entry["mean_ms"]
        if strict_scalar:
            out["ratios"]["bootstrap_cpu_scalar_vs_gpu_scalar"] = float(b_np / b_cp) if b_cp > 0 else None
            out["ratios"]["permutation_cpu_scalar_vs_gpu_scalar"] = float(p_np / p_cp) if p_cp > 0 else None
        else:
            out["ratios"]["bootstrap_cpu_vectorized_vs_gpu_vectorized"] = float(b_np / b_cp) if b_cp > 0 else None
            out["ratios"]["permutation_cpu_vectorized_vs_gpu_vectorized"] = float(p_np / p_cp) if p_cp > 0 else None

    if out["scipy"] is not None:
        b_sp = out["scipy"]["bootstrap"]["mean_ms"]
        p_sp = out["scipy"]["permutation"]["mean_ms"]
        if strict_scalar:
            out["ratios"]["bootstrap_scipy_vs_statgpu_numpy_scalar"] = float(b_sp / b_np) if b_np > 0 else None
            out["ratios"]["permutation_scipy_vs_statgpu_numpy_scalar"] = float(p_sp / p_np) if p_np > 0 else None
        else:
            out["ratios"]["bootstrap_scipy_vs_statgpu_numpy_vectorized"] = float(b_sp / b_np) if b_np > 0 else None
            out["ratios"]["permutation_scipy_vs_statgpu_numpy_vectorized"] = float(p_sp / p_np) if p_np > 0 else None

        if b_cp_entry is not None and p_cp_entry is not None:
            b_cp = b_cp_entry["mean_ms"]
            p_cp = p_cp_entry["mean_ms"]
            if strict_scalar:
                out["ratios"]["bootstrap_scipy_vs_statgpu_cupy_scalar"] = float(b_sp / b_cp) if b_cp > 0 else None
                out["ratios"]["permutation_scipy_vs_statgpu_cupy_scalar"] = float(p_sp / p_cp) if p_cp > 0 else None
            else:
                out["ratios"]["bootstrap_scipy_vs_statgpu_cupy_vectorized"] = float(b_sp / b_cp) if b_cp > 0 else None
                out["ratios"]["permutation_scipy_vs_statgpu_cupy_vectorized"] = float(p_sp / p_cp) if p_cp > 0 else None

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark resampling strategy and external comparisons")
    parser.add_argument(
        "--strict-scalar",
        action="store_true",
        help="Use strict scalar callback mode for CuPy resampling benchmarks",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix tag for output filename",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(20260405)
    cp_module = _maybe_import_cupy()
    profile = "strict_scalar" if args.strict_scalar else "optimized_vectorized"

    payload = {
        "date": str(date.today()),
        "profile": profile,
        "environment": {
            "cupy_available": bool(cp_module is not None),
            "scipy_available": bool(scipy_bootstrap is not None and scipy_permutation_test is not None),
        },
        "strategy_timing": _run_strategy_timing(cp_module, rng, strict_scalar=args.strict_scalar),
        "large_scale_external": _run_large_scale_external(cp_module, rng, strict_scalar=args.strict_scalar),
    }

    suffix_parts = []
    if args.strict_scalar:
        suffix_parts.append("strict_scalar")
    tag = str(args.output_tag).strip()
    if tag:
        suffix_parts.append(tag)
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    out_path = Path("results") / f"resampling_strategy_external_benchmark_{date.today()}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
