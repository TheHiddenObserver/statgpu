"""Benchmark and precision comparison: statgpu kernel regression vs statsmodels KernelReg."""

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
from statgpu.nonparametric import fit_kernel_regression


def _maybe_sync_cuda() -> None:
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


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


def _normalize_regression_name(name: str) -> str:
    s = str(name).strip().lower()
    if s in ("nw", "nadaraya_watson", "nadaraya-watson"):
        return "nw"
    if s in ("local_linear", "local-linear", "ll"):
        return "local_linear"
    raise ValueError("regression must be one of: nw, local_linear")


def _normalize_kernel_name(name: str) -> str:
    s = str(name).strip().lower()
    if s != "gaussian":
        raise ValueError("for statsmodels comparability, kernel must be 'gaussian'")
    return s


def _build_data(*, seed: int, n_samples: int, n_eval: int, dim: int, noise_scale: float):
    rng = np.random.default_rng(seed)

    if dim == 1:
        x = rng.uniform(-3.0, 3.0, size=n_samples)
        y_clean = np.sin(1.3 * x) + 0.2 * x
        y = y_clean + rng.normal(scale=noise_scale, size=n_samples)
        points = np.linspace(-2.9, 2.9, n_eval)
        truth = np.sin(1.3 * points) + 0.2 * points
        samples = x.reshape(-1, 1)
        eval_points = points.reshape(-1, 1)
    else:
        x = rng.normal(size=(n_samples, dim))
        coef = np.linspace(0.9, -0.4, dim)
        nonlinear = np.sin(x[:, 0])
        y_clean = x @ coef + 0.25 * nonlinear
        y = y_clean + rng.normal(scale=noise_scale, size=n_samples)

        points = rng.normal(size=(n_eval, dim))
        truth = points @ coef + 0.25 * np.sin(points[:, 0])
        samples = x
        eval_points = points

    return (
        np.asarray(samples, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(eval_points, dtype=np.float64),
        np.asarray(truth, dtype=np.float64),
    )


def _bandwidth_factor_from_abs(samples_2d: np.ndarray, bandwidth_abs: float) -> tuple[float, float]:
    if samples_2d.ndim != 2:
        raise ValueError("samples_2d must be 2D")

    sd = np.std(samples_2d, axis=0, ddof=1)
    sd = np.asarray(sd, dtype=np.float64)
    scale = float(np.mean(sd))
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = 1.0

    factor = float(bandwidth_abs / scale)
    if (not np.isfinite(factor)) or factor <= 0.0:
        raise ValueError("bandwidth factor must be positive and finite")
    return factor, scale


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark statgpu kernel regression against statsmodels KernelReg"
    )
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--n-eval", type=int, default=1200)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--noise-scale", type=float, default=0.12)
    parser.add_argument("--bandwidth-abs", type=float, default=0.45)
    parser.add_argument("--kernel", type=str, default="gaussian")
    parser.add_argument("--regression", type=str, default="nw")
    parser.add_argument(
        "--kernel-metric",
        type=str,
        default="diagonal",
        choices=["full", "diagonal"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    dim = int(args.dim)
    if dim <= 0:
        raise ValueError("--dim must be >= 1")

    regression = _normalize_regression_name(args.regression)
    kernel = _normalize_kernel_name(args.kernel)
    kernel_metric = str(args.kernel_metric).strip().lower()

    try:
        from statsmodels.nonparametric.kernel_regression import KernelReg
    except Exception as exc:
        raise RuntimeError(
            "statsmodels.nonparametric.kernel_regression.KernelReg is required"
        ) from exc

    samples, targets, points, truth = _build_data(
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        n_eval=int(args.n_eval),
        dim=dim,
        noise_scale=float(args.noise_scale),
    )

    bandwidth_abs = float(args.bandwidth_abs)
    if (not np.isfinite(bandwidth_abs)) or bandwidth_abs <= 0.0:
        raise ValueError("--bandwidth-abs must be positive")

    factor, sample_scale = _bandwidth_factor_from_abs(samples, bandwidth_abs)

    var_type = "c" * dim
    reg_type = "lc" if regression == "nw" else "ll"
    bw_vec = np.full(dim, bandwidth_abs, dtype=np.float64)
    bandwidth_per_feature = bw_vec if kernel_metric == "diagonal" else None

    def run_statsmodels():
        model = KernelReg(
            endog=targets,
            exog=samples,
            var_type=var_type,
            reg_type=reg_type,
            bw=bw_vec,
        )
        pred, _ = model.fit(points)
        return np.asarray(pred, dtype=np.float64)

    def run_statgpu_numpy():
        model = fit_kernel_regression(
            samples,
            targets,
            bandwidth=factor,
            bandwidth_per_feature=bandwidth_per_feature,
            kernel=kernel,
            regression=regression,
            kernel_metric=kernel_metric,
            backend="numpy",
        )
        return np.asarray(model.predict(points), dtype=np.float64)

    sm_res = _run_repeats(run_statsmodels, repeats=int(args.repeats), warmup=int(args.warmup))
    sg_np_res = _run_repeats(run_statgpu_numpy, repeats=int(args.repeats), warmup=int(args.warmup))

    sm_out = np.asarray(sm_res["output"], dtype=np.float64).reshape(-1)
    np_out = np.asarray(sg_np_res["output"], dtype=np.float64).reshape(-1)

    metrics = {
        "precision": {
            "numpy_vs_statsmodels": {
                "max_abs_diff": float(np.max(np.abs(np_out - sm_out))),
                "mean_abs_diff": float(np.mean(np.abs(np_out - sm_out))),
                "l2_rel_diff": _l2_rel(np_out, sm_out),
            }
        },
        "fit_quality": {
            "rmse_vs_truth": {
                "statsmodels": _rmse(sm_out, truth),
                "statgpu_numpy": _rmse(np_out, truth),
            }
        },
        "timing_ms": {
            "statsmodels": {k: v for k, v in sm_res.items() if k != "output"},
            "statgpu_numpy": {k: v for k, v in sg_np_res.items() if k != "output"},
            "ratios": {
                "statgpu_numpy_over_statsmodels": float(
                    sg_np_res["time_ms_mean"] / sm_res["time_ms_mean"]
                ),
            },
        },
    }

    if cuda_available():
        try:
            import cupy as cp

            samples_cp = cp.asarray(samples)
            targets_cp = cp.asarray(targets)
            points_cp = cp.asarray(points)

            def run_statgpu_cupy():
                model = fit_kernel_regression(
                    samples_cp,
                    targets_cp,
                    bandwidth=factor,
                    bandwidth_per_feature=bandwidth_per_feature,
                    kernel=kernel,
                    regression=regression,
                    kernel_metric=kernel_metric,
                    backend="cupy",
                )
                return cp.asnumpy(model.predict(points_cp))

            sg_cp_res = _run_repeats(
                run_statgpu_cupy,
                repeats=int(args.repeats),
                warmup=int(args.warmup),
                sync_cuda=True,
            )

            cp_out = np.asarray(sg_cp_res["output"], dtype=np.float64).reshape(-1)
            metrics["precision"]["cupy_vs_statsmodels"] = {
                "max_abs_diff": float(np.max(np.abs(cp_out - sm_out))),
                "mean_abs_diff": float(np.mean(np.abs(cp_out - sm_out))),
                "l2_rel_diff": _l2_rel(cp_out, sm_out),
            }
            metrics["precision"]["cupy_vs_numpy"] = {
                "max_abs_diff": float(np.max(np.abs(cp_out - np_out))),
                "mean_abs_diff": float(np.mean(np.abs(cp_out - np_out))),
                "l2_rel_diff": _l2_rel(cp_out, np_out),
            }
            metrics["fit_quality"]["rmse_vs_truth"]["statgpu_cupy"] = _rmse(cp_out, truth)
            metrics["timing_ms"]["statgpu_cupy"] = {
                k: v for k, v in sg_cp_res.items() if k != "output"
            }
            metrics["timing_ms"]["ratios"]["statgpu_cupy_over_statsmodels"] = float(
                sg_cp_res["time_ms_mean"] / sm_res["time_ms_mean"]
            )
            metrics["timing_ms"]["ratios"]["statgpu_cupy_over_numpy"] = float(
                sg_cp_res["time_ms_mean"] / sg_np_res["time_ms_mean"]
            )
        except Exception as exc:
            metrics["timing_ms"]["statgpu_cupy"] = {
                "error": f"cupy benchmark skipped: {exc}",
            }

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "cuda_available": bool(cuda_available()),
            "statsmodels_available": True,
        },
        "config": {
            "n_samples": int(args.n_samples),
            "n_eval": int(args.n_eval),
            "dim": dim,
            "noise_scale": float(args.noise_scale),
            "bandwidth_abs": bandwidth_abs,
            "bandwidth_factor_for_statgpu": float(factor),
            "sample_scale_for_factor": float(sample_scale),
            "kernel": kernel,
            "regression": regression,
            "kernel_metric": kernel_metric,
            "bandwidth_per_feature_abs": [float(v) for v in bw_vec],
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "seed": int(args.seed),
        },
        "metrics": metrics,
    }

    if args.json_out:
        out_path = Path(args.json_out)
    else:
        out_path = (
            Path("results")
            / f"kernel_regression_vs_statsmodels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "output": str(out_path),
        "precision": metrics["precision"],
        "fit_quality": metrics["fit_quality"],
        "timing_ratios": metrics["timing_ms"]["ratios"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
