"""Benchmark and precision comparison: statgpu KDE vs scipy.stats.gaussian_kde."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import cuda_available
from statgpu.inference import fit_kde


def _maybe_sync_cuda() -> None:
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _parse_bandwidth(text: str):
    s = str(text).strip().lower()
    if s in ("scott", "silverman"):
        return s
    return float(text)


def _build_data(*, seed: int, n_samples: int, n_eval: int, dim: int, weighted: bool):
    rng = np.random.default_rng(seed)

    if dim == 1:
        n1 = n_samples // 2
        n2 = n_samples - n1
        samples = np.concatenate(
            [
                rng.normal(loc=-1.0, scale=0.8, size=n1),
                rng.normal(loc=1.2, scale=0.6, size=n2),
            ]
        )
        points = np.linspace(-5.0, 5.0, n_eval)
    else:
        idx = np.arange(dim)
        cov = 0.5 ** np.abs(np.subtract.outer(idx, idx))
        chol = np.linalg.cholesky(cov)
        samples = rng.normal(size=(n_samples, dim)) @ chol.T
        points = rng.normal(size=(n_eval, dim)) @ chol.T

    if weighted:
        w = rng.uniform(0.1, 2.0, size=n_samples)
        w = w / w.sum()
    else:
        w = None

    return samples.astype(np.float64), points.astype(np.float64), w


def _run_repeats(fn, *, repeats: int, warmup: int, sync_cuda: bool = False):
    for _ in range(max(0, warmup)):
        fn()
        if sync_cuda:
            _maybe_sync_cuda()

    times = []
    last_out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        last_out = fn()
        if sync_cuda:
            _maybe_sync_cuda()
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "time_ms_mean": float(np.mean(times)),
        "time_ms_std": float(np.std(times, ddof=0)),
        "time_ms_min": float(np.min(times)),
        "time_ms_max": float(np.max(times)),
        "times_ms": [float(t) for t in times],
        "output": last_out,
    }


def _l2_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom <= 1e-15:
        denom = 1.0
    return float(np.linalg.norm(a - b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark statgpu KDE against SciPy")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--bandwidth", type=str, default="scott")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if int(args.dim) <= 0:
        raise ValueError("--dim must be >= 1")

    scipy_stats = __import__("scipy.stats", fromlist=["gaussian_kde"])

    bandwidth = _parse_bandwidth(args.bandwidth)
    samples, points, weights = _build_data(
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        n_eval=int(args.n_eval),
        dim=int(args.dim),
        weighted=bool(args.weighted),
    )

    def run_scipy():
        data_ref = samples if samples.ndim == 1 else samples.T
        points_ref = points if points.ndim == 1 else points.T
        kde = scipy_stats.gaussian_kde(data_ref, bw_method=bandwidth, weights=weights)
        return np.asarray(kde(points_ref), dtype=np.float64)

    def run_statgpu_numpy():
        kde = fit_kde(samples, bandwidth=bandwidth, weights=weights, backend="numpy")
        return np.asarray(kde(points), dtype=np.float64)

    scipy_res = _run_repeats(run_scipy, repeats=int(args.repeats), warmup=int(args.warmup))
    statgpu_np_res = _run_repeats(run_statgpu_numpy, repeats=int(args.repeats), warmup=int(args.warmup))

    scipy_out = np.asarray(scipy_res["output"], dtype=np.float64)
    np_out = np.asarray(statgpu_np_res["output"], dtype=np.float64)

    metrics = {
        "precision": {
            "numpy_vs_scipy": {
                "max_abs_diff": float(np.max(np.abs(np_out - scipy_out))),
                "mean_abs_diff": float(np.mean(np.abs(np_out - scipy_out))),
                "l2_rel_diff": _l2_rel(np_out, scipy_out),
            }
        },
        "timing_ms": {
            "scipy": {k: v for k, v in scipy_res.items() if k != "output"},
            "statgpu_numpy": {k: v for k, v in statgpu_np_res.items() if k != "output"},
            "ratios": {
                "statgpu_numpy_over_scipy": float(statgpu_np_res["time_ms_mean"] / scipy_res["time_ms_mean"]),
            },
        },
    }

    gpu_payload = None
    if cuda_available():
        import cupy as cp

        samples_cp = cp.asarray(samples)
        points_cp = cp.asarray(points)
        weights_cp = None if weights is None else cp.asarray(weights)

        def run_statgpu_cupy():
            kde = fit_kde(samples_cp, bandwidth=bandwidth, weights=weights_cp, backend="cupy")
            return cp.asnumpy(kde(points_cp))

        statgpu_cp_res = _run_repeats(
            run_statgpu_cupy,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            sync_cuda=True,
        )

        cp_out = np.asarray(statgpu_cp_res["output"], dtype=np.float64)
        metrics["precision"]["cupy_vs_scipy"] = {
            "max_abs_diff": float(np.max(np.abs(cp_out - scipy_out))),
            "mean_abs_diff": float(np.mean(np.abs(cp_out - scipy_out))),
            "l2_rel_diff": _l2_rel(cp_out, scipy_out),
        }
        metrics["precision"]["cupy_vs_numpy"] = {
            "max_abs_diff": float(np.max(np.abs(cp_out - np_out))),
            "mean_abs_diff": float(np.mean(np.abs(cp_out - np_out))),
            "l2_rel_diff": _l2_rel(cp_out, np_out),
        }
        metrics["timing_ms"]["statgpu_cupy"] = {k: v for k, v in statgpu_cp_res.items() if k != "output"}
        metrics["timing_ms"]["ratios"]["statgpu_cupy_over_scipy"] = float(
            statgpu_cp_res["time_ms_mean"] / scipy_res["time_ms_mean"]
        )
        metrics["timing_ms"]["ratios"]["statgpu_cupy_over_numpy"] = float(
            statgpu_cp_res["time_ms_mean"] / statgpu_np_res["time_ms_mean"]
        )
        gpu_payload = statgpu_cp_res

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "cuda_available": bool(cuda_available()),
        },
        "config": {
            "n_samples": int(args.n_samples),
            "n_eval": int(args.n_eval),
            "dim": int(args.dim),
            "bandwidth": args.bandwidth,
            "weighted": bool(args.weighted),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "seed": int(args.seed),
        },
        "metrics": metrics,
    }

    if args.json_out:
        out_path = Path(args.json_out)
    else:
        out_path = Path("results") / f"kde_vs_scipy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "output": str(out_path),
        "precision": metrics["precision"],
        "timing_ratios": metrics["timing_ms"]["ratios"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
