"""Benchmark inference backends/frameworks and unified resampling runtime.

Usage:
    python dev/benchmarks/benchmark_inference_backends.py [--strict-scalar] [--output-tag TAG]

Outputs:
    results/inference_backend_benchmark_YYYY-MM-DD[_strict_scalar][_TAG].json
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

from statgpu.inference import adjust_pvalues, bootstrap_statistic, combine_pvalues, permutation_test


try:
    from statsmodels.stats.multitest import multipletests as sm_multipletests
except Exception:  # pragma: no cover - optional dependency
    sm_multipletests = None

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None


def _maybe_import_cupy():
    try:
        import cupy as cp
        # Probe runtime availability: CuPy may be importable while CUDA driver is unusable.
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
        return cp
    except Exception:
        return None


def _bench(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeats: int = 5,
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


def _method_to_statsmodels_name(method: str) -> str:
    mapping = {
        "bh": "fdr_bh",
        "by": "fdr_by",
        "holm": "holm",
        "bonferroni": "bonferroni",
    }
    return mapping[method]


def _run_fdr_benchmark(cp_module, rng: np.random.Generator) -> Dict[str, Any]:
    p = rng.uniform(1e-10, 1.0 - 1e-10, size=200_000)
    methods = ["bh", "by", "holm", "bonferroni"]

    out: Dict[str, Any] = {
        "config": {
            "n_pvalues": int(p.size),
            "warmup": 1,
            "repeats": 5,
        },
        "numpy": {},
        "cupy": None,
        "statsmodels": None,
        "consistency": {},
    }

    p_cp = None
    if cp_module is not None:
        p_cp = cp_module.asarray(p)

    for method in methods:
        # statgpu numpy
        out["numpy"][method] = _bench(
            lambda m=method: adjust_pvalues(p, method=m, backend="numpy"),
            warmup=1,
            repeats=5,
        )

        # statsmodels
        if sm_multipletests is not None:
            sm_method = _method_to_statsmodels_name(method)
            if out["statsmodels"] is None:
                out["statsmodels"] = {}
            out["statsmodels"][method] = _bench(
                lambda mm=sm_method: sm_multipletests(p, method=mm),
                warmup=1,
                repeats=5,
            )
            reject_sm, p_sm, _, _ = sm_multipletests(p, method=sm_method)
            reject_np, p_np = adjust_pvalues(p, method=method, backend="numpy")
            out["consistency"][method] = {
                "max_abs_padj_diff_vs_statsmodels": float(np.max(np.abs(p_np - p_sm))),
                "reject_mismatch_count_vs_statsmodels": int(np.sum(reject_np != reject_sm)),
            }

        # statgpu cupy
        if p_cp is not None:
            if out["cupy"] is None:
                out["cupy"] = {}
            out["cupy"][method] = _bench(
                lambda m=method: adjust_pvalues(p_cp, method=m, backend="cupy"),
                warmup=1,
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

    return out


def _as_numpy_array(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _scipy_fisher_axis(p_matrix: np.ndarray):
    if scipy_stats is None:
        raise RuntimeError("scipy is unavailable")

    try:
        stat, pval = scipy_stats.combine_pvalues(p_matrix, method="fisher", axis=1)
        return np.asarray(stat, dtype=float), np.asarray(pval, dtype=float)
    except TypeError:
        # Fallback for older SciPy versions that don't expose an axis argument.
        stat = np.empty(p_matrix.shape[0], dtype=float)
        pval = np.empty(p_matrix.shape[0], dtype=float)
        for i, row in enumerate(p_matrix):
            s_i, p_i = scipy_stats.combine_pvalues(row, method="fisher")
            stat[i] = float(s_i)
            pval[i] = float(p_i)
        return stat, pval


def _cauchy_reference_axis(p_matrix: np.ndarray, weights: np.ndarray):
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / np.sum(w)
    eps = np.finfo(np.float64).eps
    p_safe = np.clip(np.asarray(p_matrix, dtype=np.float64), eps, 1.0 - eps)
    stat = np.sum(w.reshape(1, -1) * np.tan((0.5 - p_safe) * np.pi), axis=1)
    pval = np.clip(0.5 - np.arctan(stat) / np.pi, 0.0, 1.0)
    return stat, pval


def _run_pvalue_combination_benchmark(cp_module, rng: np.random.Generator) -> Dict[str, Any]:
    p_matrix = rng.uniform(1e-10, 1.0 - 1e-10, size=(4000, 64))
    weights = rng.uniform(0.1, 2.0, size=64)

    out: Dict[str, Any] = {
        "config": {
            "n_groups": int(p_matrix.shape[0]),
            "group_size": int(p_matrix.shape[1]),
            "axis": 1,
            "warmup": 1,
            "repeats": 5,
        },
        "numpy": {},
        "cupy": None,
        "scipy": None,
        "consistency": {},
        "ratios": {},
    }

    out["numpy"]["fisher"] = _bench(
        lambda: combine_pvalues(
            p_matrix,
            method="fisher",
            axis=1,
            backend="numpy",
        ),
        warmup=1,
        repeats=5,
    )

    out["numpy"]["cauchy"] = _bench(
        lambda: combine_pvalues(
            p_matrix,
            method="cauchy",
            weights=weights,
            axis=1,
            backend="numpy",
        ),
        warmup=1,
        repeats=5,
    )

    out["numpy"]["acat_alias"] = _bench(
        lambda: combine_pvalues(
            p_matrix,
            method="acat",
            weights=weights,
            axis=1,
            backend="numpy",
        ),
        warmup=1,
        repeats=5,
    )

    stat_np_f, p_np_f = combine_pvalues(
        p_matrix,
        method="fisher",
        axis=1,
        backend="numpy",
    )
    stat_np_c, p_np_c = combine_pvalues(
        p_matrix,
        method="cauchy",
        weights=weights,
        axis=1,
        backend="numpy",
    )
    stat_np_a, p_np_a = combine_pvalues(
        p_matrix,
        method="acat",
        weights=weights,
        axis=1,
        backend="numpy",
    )

    out["consistency"]["cauchy_vs_acat_alias"] = {
        "max_abs_stat_diff": float(np.max(np.abs(_as_numpy_array(stat_np_c) - _as_numpy_array(stat_np_a)))),
        "max_abs_pvalue_diff": float(np.max(np.abs(_as_numpy_array(p_np_c) - _as_numpy_array(p_np_a)))),
    }

    stat_ref_c, p_ref_c = _cauchy_reference_axis(p_matrix, weights)
    out["consistency"]["cauchy_vs_numpy_reference"] = {
        "max_abs_stat_diff": float(np.max(np.abs(_as_numpy_array(stat_np_c) - stat_ref_c))),
        "max_abs_pvalue_diff": float(np.max(np.abs(_as_numpy_array(p_np_c) - p_ref_c))),
    }

    if scipy_stats is not None:
        out["scipy"] = {}
        out["scipy"]["fisher"] = _bench(
            lambda: _scipy_fisher_axis(p_matrix),
            warmup=1,
            repeats=5,
        )

        stat_sp_f, p_sp_f = _scipy_fisher_axis(p_matrix)
        out["consistency"]["fisher_vs_scipy"] = {
            "max_abs_stat_diff": float(np.max(np.abs(_as_numpy_array(stat_np_f) - stat_sp_f))),
            "max_abs_pvalue_diff": float(np.max(np.abs(_as_numpy_array(p_np_f) - p_sp_f))),
        }

    if cp_module is not None:
        p_cp = cp_module.asarray(p_matrix)
        w_cp = cp_module.asarray(weights)

        out["cupy"] = {}
        out["cupy"]["fisher"] = _bench(
            lambda: combine_pvalues(
                p_cp,
                method="fisher",
                axis=1,
                backend="cupy",
            ),
            warmup=1,
            repeats=5,
            synchronize=cp_module.cuda.runtime.deviceSynchronize,
        )
        out["cupy"]["cauchy"] = _bench(
            lambda: combine_pvalues(
                p_cp,
                method="cauchy",
                weights=w_cp,
                axis=1,
                backend="cupy",
            ),
            warmup=1,
            repeats=5,
            synchronize=cp_module.cuda.runtime.deviceSynchronize,
        )

        stat_cp_f, p_cp_f = combine_pvalues(
            p_cp,
            method="fisher",
            axis=1,
            backend="cupy",
        )
        stat_cp_c, p_cp_c = combine_pvalues(
            p_cp,
            method="cauchy",
            weights=w_cp,
            axis=1,
            backend="cupy",
        )

        out["consistency"]["fisher_cupy_vs_numpy"] = {
            "max_abs_stat_diff": float(np.max(np.abs(_as_numpy_array(stat_np_f) - _as_numpy_array(stat_cp_f)))),
            "max_abs_pvalue_diff": float(np.max(np.abs(_as_numpy_array(p_np_f) - _as_numpy_array(p_cp_f)))),
        }
        out["consistency"]["cauchy_cupy_vs_numpy"] = {
            "max_abs_stat_diff": float(np.max(np.abs(_as_numpy_array(stat_np_c) - _as_numpy_array(stat_cp_c)))),
            "max_abs_pvalue_diff": float(np.max(np.abs(_as_numpy_array(p_np_c) - _as_numpy_array(p_cp_c)))),
        }

    fisher_np_ms = out["numpy"]["fisher"]["mean_ms"]
    cauchy_np_ms = out["numpy"]["cauchy"]["mean_ms"]
    out["ratios"]["numpy_cauchy_vs_fisher"] = float(cauchy_np_ms / fisher_np_ms) if fisher_np_ms > 0 else None

    if out["scipy"] is not None:
        fisher_sp_ms = out["scipy"]["fisher"]["mean_ms"]
        out["ratios"]["fisher_scipy_vs_statgpu_numpy"] = float(fisher_sp_ms / fisher_np_ms) if fisher_np_ms > 0 else None

    if out["cupy"] is not None:
        fisher_cp_ms = out["cupy"]["fisher"]["mean_ms"]
        cauchy_cp_ms = out["cupy"]["cauchy"]["mean_ms"]
        out["ratios"]["fisher_numpy_vs_cupy"] = float(fisher_np_ms / fisher_cp_ms) if fisher_cp_ms > 0 else None
        out["ratios"]["cauchy_numpy_vs_cupy"] = float(cauchy_np_ms / cauchy_cp_ms) if cauchy_cp_ms > 0 else None
        out["ratios"]["cupy_cauchy_vs_fisher"] = float(cauchy_cp_ms / fisher_cp_ms) if fisher_cp_ms > 0 else None

    return out


def _run_resampling_benchmark(
    cp_module,
    rng: np.random.Generator,
    *,
    strict_scalar: bool = False,
) -> Dict[str, Any]:
    x_np = rng.normal(loc=1.0, scale=1.0, size=5000)
    X_np = rng.normal(size=(3000, 1))
    y_np = 1.7 * X_np[:, 0] + rng.normal(scale=0.6, size=3000)

    out: Dict[str, Any] = {
        "config": {
            "bootstrap_n": 5000,
            "bootstrap_n_resamples": 1500,
            "permutation_n": 3000,
            "permutation_n_resamples": 2000,
            "warmup": 1,
            "repeats": 5,
            "cupy_mode": "strict_scalar" if strict_scalar else "optimized_vectorized",
        },
        "bootstrap": {},
        "permutation": {},
        "gpu_sync_probe": None,
    }

    out["bootstrap"]["numpy"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr),
            x_np,
            n_resamples=1500,
            strategy="iid",
            random_state=7,
            backend="numpy",
        ),
        warmup=1,
        repeats=5,
    )

    out["permutation"]["numpy"] = _bench(
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
        repeats=5,
    )

    if cp_module is not None:
        x_cp = cp_module.asarray(x_np)
        X_cp = cp_module.asarray(X_np)
        y_cp = cp_module.asarray(y_np)

        if strict_scalar:
            out["bootstrap"]["cupy"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1500,
                    strategy="iid",
                    random_state=7,
                    backend="cupy",
                ),
                warmup=1,
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

            out["permutation"]["cupy"] = _bench(
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
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )
        else:
            out["bootstrap"]["cupy"] = _bench(
                lambda: bootstrap_statistic(
                    lambda arr: cp_module.mean(arr),
                    x_cp,
                    n_resamples=1500,
                    strategy="iid",
                    random_state=7,
                    backend="cupy",
                    force_vectorized=True,
                    statistic_hint="mean",
                ),
                warmup=1,
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

            out["permutation"]["cupy"] = _bench(
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
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            )

        # Probe for host-sync overhead from scalar extraction in tight loops.
        def _sync_heavy_probe():
            acc = 0.0
            for _ in range(800):
                idx = cp_module.random.randint(0, x_cp.shape[0], size=1024, dtype=cp_module.int64)
                acc += float(cp_module.mean(x_cp[idx]).item())
            return acc

        def _sync_light_probe():
            acc = cp_module.asarray(0.0)
            for _ in range(800):
                idx = cp_module.random.randint(0, x_cp.shape[0], size=1024, dtype=cp_module.int64)
                acc = acc + cp_module.mean(x_cp[idx])
            return acc

        out["gpu_sync_probe"] = {
            "sync_heavy_ms": _bench(
                _sync_heavy_probe,
                warmup=1,
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            ),
            "sync_light_ms": _bench(
                _sync_light_probe,
                warmup=1,
                repeats=5,
                synchronize=cp_module.cuda.runtime.deviceSynchronize,
            ),
        }

        b_np = out["bootstrap"]["numpy"]["mean_ms"]
        b_cp = out["bootstrap"]["cupy"]["mean_ms"]
        p_np = out["permutation"]["numpy"]["mean_ms"]
        p_cp = out["permutation"]["cupy"]["mean_ms"]
        out["ratios"] = {
            "bootstrap_cpu_vs_gpu": float(b_np / b_cp) if b_cp > 0 else None,
            "permutation_cpu_vs_gpu": float(p_np / p_cp) if p_cp > 0 else None,
        }

    return out


def _corr_stat_maybe_vectorized(X_in, y_in, xp_module):
    """Pearson correlation for 1D y or batched y rows."""
    x = X_in[:, 0]
    y_arr = xp_module.asarray(y_in)

    if y_arr.ndim == 1:
        return xp_module.corrcoef(x, y_arr)[0, 1]

    x_centered = x - xp_module.mean(x)
    y_centered = y_arr - xp_module.mean(y_arr, axis=1, keepdims=True)
    x_norm_sq = xp_module.sum(x_centered * x_centered)
    y_norm_sq = xp_module.sum(y_centered * y_centered, axis=1)
    denom = xp_module.sqrt(x_norm_sq * y_norm_sq)

    inf_value = np.inf if xp_module is np else xp_module.inf
    denom = xp_module.where(denom > 0.0, denom, inf_value)
    numer = xp_module.sum(y_centered * x_centered.reshape(1, -1), axis=1)
    return numer / denom


def _run_resampling_large_scale_benchmark(cp_module, rng: np.random.Generator) -> Dict[str, Any]:
    x_np = rng.normal(loc=1.0, scale=1.0, size=30_000)
    X_np = rng.normal(size=(18_000, 1))
    y_np = 1.7 * X_np[:, 0] + rng.normal(scale=0.6, size=18_000)

    out: Dict[str, Any] = {
        "config": {
            "bootstrap_n": 30_000,
            "bootstrap_n_resamples": 2_500,
            "permutation_n": 18_000,
            "permutation_n_resamples": 2_000,
            "warmup": 1,
            "repeats": 3,
            "note": "Large-scale workload plus vectorized callback path for CuPy iid.",
        },
        "bootstrap": {},
        "permutation": {},
    }

    out["bootstrap"]["numpy_scalar"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr),
            x_np,
            n_resamples=2500,
            strategy="iid",
            random_state=17,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )

    out["bootstrap"]["numpy_vectorized"] = _bench(
        lambda: bootstrap_statistic(
            lambda arr: np.mean(arr, axis=-1),
            x_np,
            n_resamples=2500,
            strategy="iid",
            random_state=17,
            backend="numpy",
            force_vectorized=True,
        ),
        warmup=1,
        repeats=3,
    )

    out["permutation"]["numpy_scalar"] = _bench(
        lambda: permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X_np,
            y_np,
            n_resamples=2000,
            strategy="iid",
            alternative="two-sided",
            random_state=23,
            backend="numpy",
        ),
        warmup=1,
        repeats=3,
    )

    out["permutation"]["numpy_vectorized"] = _bench(
        lambda: permutation_test(
            lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, np),
            X_np,
            y_np,
            n_resamples=2000,
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

        out["bootstrap"]["cupy_scalar"] = _bench(
            lambda: bootstrap_statistic(
                lambda arr: cp_module.mean(arr),
                x_cp,
                n_resamples=2500,
                strategy="iid",
                random_state=17,
                backend="cupy",
            ),
            warmup=1,
            repeats=3,
            synchronize=cp_module.cuda.runtime.deviceSynchronize,
        )

        out["bootstrap"]["cupy_vectorized"] = _bench(
            lambda: bootstrap_statistic(
                lambda arr: cp_module.mean(arr, axis=-1),
                x_cp,
                n_resamples=2500,
                strategy="iid",
                random_state=17,
                backend="cupy",
                force_vectorized=True,
            ),
            warmup=1,
            repeats=3,
            synchronize=cp_module.cuda.runtime.deviceSynchronize,
        )

        out["permutation"]["cupy_scalar"] = _bench(
            lambda: permutation_test(
                lambda X_in, y_in: cp_module.corrcoef(X_in[:, 0], y_in)[0, 1],
                X_cp,
                y_cp,
                n_resamples=2000,
                strategy="iid",
                alternative="two-sided",
                random_state=23,
                backend="cupy",
            ),
            warmup=1,
            repeats=3,
            synchronize=cp_module.cuda.runtime.deviceSynchronize,
        )

        out["permutation"]["cupy_vectorized"] = _bench(
            lambda: permutation_test(
                lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, cp_module),
                X_cp,
                y_cp,
                n_resamples=2000,
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

        b_np = out["bootstrap"]["numpy_scalar"]["mean_ms"]
        b_np_vec = out["bootstrap"]["numpy_vectorized"]["mean_ms"]
        b_cp_scalar = out["bootstrap"]["cupy_scalar"]["mean_ms"]
        b_cp_vec = out["bootstrap"]["cupy_vectorized"]["mean_ms"]
        p_np = out["permutation"]["numpy_scalar"]["mean_ms"]
        p_np_vec = out["permutation"]["numpy_vectorized"]["mean_ms"]
        p_cp_scalar = out["permutation"]["cupy_scalar"]["mean_ms"]
        p_cp_vec = out["permutation"]["cupy_vectorized"]["mean_ms"]

        out["ratios"] = {
            "bootstrap_cpu_vs_gpu_scalar": float(b_np / b_cp_scalar) if b_cp_scalar > 0 else None,
            "bootstrap_cpu_vs_gpu_vectorized": float(b_np / b_cp_vec) if b_cp_vec > 0 else None,
            "bootstrap_cpu_vectorized_vs_gpu_vectorized": float(b_np_vec / b_cp_vec) if b_cp_vec > 0 else None,
            "bootstrap_cpu_scalar_vs_cpu_vectorized": float(b_np / b_np_vec) if b_np_vec > 0 else None,
            "permutation_cpu_vs_gpu_scalar": float(p_np / p_cp_scalar) if p_cp_scalar > 0 else None,
            "permutation_cpu_vs_gpu_vectorized": float(p_np / p_cp_vec) if p_cp_vec > 0 else None,
            "permutation_cpu_vectorized_vs_gpu_vectorized": float(p_np_vec / p_cp_vec) if p_cp_vec > 0 else None,
            "permutation_cpu_scalar_vs_cpu_vectorized": float(p_np / p_np_vec) if p_np_vec > 0 else None,
            "bootstrap_gpu_scalar_vs_vectorized": float(b_cp_scalar / b_cp_vec) if b_cp_vec > 0 else None,
            "permutation_gpu_scalar_vs_vectorized": float(p_cp_scalar / p_cp_vec) if p_cp_vec > 0 else None,
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference backends")
    parser.add_argument(
        "--strict-scalar",
        action="store_true",
        help="Use strict scalar callback mode for CuPy resampling benchmark",
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
            "statsmodels_available": bool(sm_multipletests is not None),
            "scipy_available": bool(scipy_stats is not None),
        },
        "fdr": _run_fdr_benchmark(cp_module, rng),
        "pvalue_combination": _run_pvalue_combination_benchmark(cp_module, rng),
        "resampling": _run_resampling_benchmark(
            cp_module,
            rng,
            strict_scalar=args.strict_scalar,
        ),
        "resampling_large_scale": _run_resampling_large_scale_benchmark(cp_module, rng),
    }

    suffix_parts = []
    if args.strict_scalar:
        suffix_parts.append("strict_scalar")
    tag = str(args.output_tag).strip()
    if tag:
        suffix_parts.append(tag)
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    out_path = Path("results") / f"inference_backend_benchmark_{date.today()}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
