"""Benchmark and precision comparison: statgpu nonparametric vs R.

This script compares:
1) KDE: statgpu.fit_kde vs R density()
2) Kernel regression (NW): statgpu.fit_kernel_regression vs R ksmooth()
3) Kernel regression (local linear): statgpu local_linear vs R KernSmooth::locpoly
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

from statgpu._config import cuda_available
from statgpu.nonparametric import (
    fit_kde,
    fit_kernel_regression,
    kde_confidence_interval,
    kde_bootstrap_confidence_interval,
)


def _l2_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom <= 1e-15:
        denom = 1.0
    return float(np.linalg.norm(a - b) / denom)


def _maybe_sync_cuda() -> None:
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _to_numpy_array(x) -> np.ndarray:
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return np.asarray(cp.asnumpy(x), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float64)


def _run_statgpu_repeats(fn, repeats: int, warmup: int, *, sync_cuda: bool = False) -> tuple[dict, np.ndarray]:
    for _ in range(max(0, int(warmup))):
        fn()
        if sync_cuda:
            _maybe_sync_cuda()

    times_ms: list[float] = []
    out = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        out = fn()
        if sync_cuda:
            _maybe_sync_cuda()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    assert out is not None
    return {
        "time_ms_mean": float(np.mean(times_ms)),
        "time_ms_std": float(np.std(times_ms, ddof=0)),
        "time_ms_min": float(np.min(times_ms)),
        "time_ms_max": float(np.max(times_ms)),
        "times_ms": [float(t) for t in times_ms],
    }, np.asarray(out, dtype=np.float64)


def _run_statgpu_kde_ci_repeats(
    x,
    points,
    *,
    bandwidth_factor: float,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
    ci_method: str,
    backend: str,
    repeats: int,
    warmup: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    use_cupy = str(backend).strip().lower() == "cupy"

    def _one(seed_offset: int):
        method_name = str(ci_method).strip().lower()
        if method_name == "bootstrap":
            result = kde_bootstrap_confidence_interval(
                x,
                points,
                bandwidth=bandwidth_factor,
                kernel="gaussian",
                backend=backend,
                n_resamples=int(n_resamples),
                confidence_level=float(confidence_level),
                random_state=int(random_state + seed_offset),
                method="percentile",
            )
        elif method_name == "normal":
            result = kde_confidence_interval(
                x,
                points,
                bandwidth=bandwidth_factor,
                kernel="gaussian",
                backend=backend,
                n_resamples=int(n_resamples),
                confidence_level=float(confidence_level),
                random_state=int(random_state + seed_offset),
                method="normal",
            )
        else:
            raise ValueError("ci_method must be one of: 'normal', 'bootstrap'")
        return {
            "estimate": np.asarray(result.estimate, dtype=np.float64),
            "lower": np.asarray(result.lower, dtype=np.float64),
            "upper": np.asarray(result.upper, dtype=np.float64),
        }

    for i in range(max(0, int(warmup))):
        _one(i)
        if use_cupy:
            _maybe_sync_cuda()

    times_ms: list[float] = []
    out = None
    for i in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        out = _one(10_000 + i)
        if use_cupy:
            _maybe_sync_cuda()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    assert out is not None
    return {
        "time_ms_mean": float(np.mean(times_ms)),
        "time_ms_std": float(np.std(times_ms, ddof=0)),
        "time_ms_min": float(np.min(times_ms)),
        "time_ms_max": float(np.max(times_ms)),
        "times_ms": [float(t) for t in times_ms],
    }, out


def _run_scipy_kde_ci_repeats(
    x: np.ndarray,
    points: np.ndarray,
    *,
    bandwidth_factor: float,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
    ci_method: str,
    repeats: int,
    warmup: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    scipy_stats = __import__("scipy.stats", fromlist=["gaussian_kde"])

    def _one(seed_offset: int):
        method_name = str(ci_method).strip().lower()
        rng = np.random.default_rng(int(random_state + seed_offset))
        ref = scipy_stats.gaussian_kde(x, bw_method=float(bandwidth_factor))
        estimate = np.asarray(ref(points), dtype=np.float64)

        if method_name == "bootstrap":
            n = int(x.shape[0])
            boot = np.empty((int(n_resamples), int(points.shape[0])), dtype=np.float64)
            for b in range(int(n_resamples)):
                idx = rng.integers(0, n, size=n)
                ref_b = scipy_stats.gaussian_kde(x[idx], bw_method=float(bandwidth_factor))
                boot[b, :] = np.asarray(ref_b(points), dtype=np.float64)

            alpha = 1.0 - float(confidence_level)
            lower = np.quantile(boot, alpha / 2.0, axis=0)
            upper = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
        elif method_name == "normal":
            n = int(x.shape[0])
            cov11 = float(np.asarray(ref.covariance, dtype=np.float64).reshape(1, 1)[0, 0])
            h = float(np.sqrt(max(cov11, np.finfo(np.float64).tiny)))
            r_kernel = 1.0 / (2.0 * np.sqrt(np.pi))
            var = np.maximum(estimate, 0.0) * (r_kernel / (float(n) * h))
            se = np.sqrt(np.maximum(var, 0.0))
            z = float(NormalDist().inv_cdf(0.5 + 0.5 * float(confidence_level)))
            lower = np.maximum(estimate - z * se, 0.0)
            upper = estimate + z * se
        else:
            raise ValueError("ci_method must be one of: 'normal', 'bootstrap'")

        return {
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
        }

    for i in range(max(0, int(warmup))):
        _one(i)

    times_ms: list[float] = []
    out = None
    for i in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        out = _one(20_000 + i)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    assert out is not None
    return {
        "time_ms_mean": float(np.mean(times_ms)),
        "time_ms_std": float(np.std(times_ms, ddof=0)),
        "time_ms_min": float(np.min(times_ms)),
        "time_ms_max": float(np.max(times_ms)),
        "times_ms": [float(t) for t in times_ms],
    }, out


def _run_r(payload: dict) -> dict:
    def _parse_r_float(text: str, *, default: float = 0.0) -> float:
        s = str(text).strip()
        if s == "" or s.upper() == "NA":
            return float(default)
        return float(s)

    r_code = r"""
      args <- commandArgs(trailingOnly = TRUE)
    x_file <- args[1]
    y_file <- args[2]
    points_file <- args[3]
    points_ci_file <- args[4]
    bw_density <- as.numeric(args[5])
    bw_ksmooth <- as.numeric(args[6])
    bw_locpoly <- as.numeric(args[7])
    repeats <- as.integer(args[8])
    warmup <- as.integer(args[9])
    pred_file <- args[10]
    timing_file <- args[11]
    ci_resamples <- as.integer(args[12])
    ci_level <- as.numeric(args[13])
    ci_file <- args[14]
    ci_method <- tolower(trimws(args[15]))

    x <- as.numeric(scan(x_file, what = numeric(), quiet = TRUE))
    y <- as.numeric(scan(y_file, what = numeric(), quiet = TRUE))
    points <- as.numeric(scan(points_file, what = numeric(), quiet = TRUE))
    points_ci <- as.numeric(scan(points_ci_file, what = numeric(), quiet = TRUE))

      time_repeats <- function(expr_fun, repeats, warmup) {
        for (i in seq_len(max(0L, warmup))) {
          expr_fun()
        }
        times <- numeric(max(1L, repeats))
        out <- NULL
        for (i in seq_len(max(1L, repeats))) {
          tm <- system.time({ out <- expr_fun() })["elapsed"]
          times[i] <- as.numeric(tm) * 1000.0
        }
        list(times_ms = times, output = out)
      }

      kde_eval <- function() {
                d <- density(x, bw = bw_density, kernel = "gaussian", n = length(points),
                     from = min(points), to = max(points))
        as.numeric(d$y)
      }

      nw_eval <- function() {
                k <- ksmooth(x, y, kernel = "normal", bandwidth = bw_ksmooth, x.points = points)
        as.numeric(k$y)
      }

      ll_eval <- function() {
                if (requireNamespace("KernSmooth", quietly = TRUE)) {
                    lp <- KernSmooth::locpoly(
                        x, y,
                        drv = 0L,
                        degree = 1L,
                        bandwidth = bw_locpoly,
                        gridsize = max(401L, length(points)),
                        range.x = c(min(points), max(points))
                    )
                    as.numeric(approx(lp$x, lp$y, xout = points, rule = 2)$y)
                } else {
                    k <- ksmooth(x, y, kernel = "normal", bandwidth = bw_ksmooth, x.points = points)
                    as.numeric(k$y)
                }
      }

      r_kde <- time_repeats(kde_eval, repeats, warmup)
      r_nw <- time_repeats(nw_eval, repeats, warmup)
      r_ll <- time_repeats(ll_eval, repeats, warmup)

              kde_ci_eval <- function() {
                        if (ci_method == "bootstrap") {
                            n <- length(x)
                            boot <- matrix(0.0, nrow = max(1L, ci_resamples), ncol = length(points_ci))
                            for (i in seq_len(max(1L, ci_resamples))) {
                                idx <- sample.int(n, size = n, replace = TRUE)
                                d <- density(x[idx], bw = bw_density, kernel = "gaussian", n = length(points_ci),
                                             from = min(points_ci), to = max(points_ci))
                                boot[i, ] <- as.numeric(d$y)
                            }
                            base <- density(x, bw = bw_density, kernel = "gaussian", n = length(points_ci),
                                            from = min(points_ci), to = max(points_ci))
                            base <- as.numeric(base$y)
                            alpha <- 1.0 - ci_level
                            lower <- apply(boot, 2, function(v) as.numeric(quantile(v, probs = alpha / 2.0, names = FALSE, type = 7)))
                            upper <- apply(boot, 2, function(v) as.numeric(quantile(v, probs = 1.0 - alpha / 2.0, names = FALSE, type = 7)))
                            return(list(estimate = as.numeric(base), lower = as.numeric(lower), upper = as.numeric(upper)))
                        }
                        if (ci_method == "normal") {
                            base <- density(x, bw = bw_density, kernel = "gaussian", n = length(points_ci),
                                            from = min(points_ci), to = max(points_ci))
                            est <- as.numeric(base$y)
                            n <- length(x)
                            rk <- 1.0 / (2.0 * sqrt(pi))
                            se <- sqrt(pmax(est, 0.0) * rk / (n * bw_density))
                            z <- qnorm(0.5 + 0.5 * ci_level)
                            lower <- pmax(est - z * se, 0.0)
                            upper <- est + z * se
                            return(list(estimate = as.numeric(est), lower = as.numeric(lower), upper = as.numeric(upper)))
                        }
                        stop("ci_method must be one of: normal, bootstrap")
                    }
            ci_tm <- system.time({ r_kde_ci <- kde_ci_eval() })["elapsed"]

            summarize_times <- function(method, times) {
                data.frame(
                    method = method,
                    time_ms_mean = as.numeric(mean(times)),
                    time_ms_std = as.numeric(sd(times)),
                    time_ms_min = as.numeric(min(times)),
                    time_ms_max = as.numeric(max(times)),
                    stringsAsFactors = FALSE
                )
            }

            pred_df <- data.frame(
                kde = as.numeric(r_kde$output),
                nw = as.numeric(r_nw$output),
                local_linear = as.numeric(r_ll$output)
            )
            write.csv(pred_df, file = pred_file, row.names = FALSE)

            timing_df <- rbind(
                summarize_times("kde", r_kde$times_ms),
                summarize_times("nw", r_nw$times_ms),
                summarize_times("local_linear", r_ll$times_ms),
                summarize_times("kde_ci", c(as.numeric(ci_tm) * 1000.0))
            )
            write.csv(timing_df, file = timing_file, row.names = FALSE)

            ci_df <- data.frame(
                estimate = as.numeric(r_kde_ci$estimate),
                lower = as.numeric(r_kde_ci$lower),
                upper = as.numeric(r_kde_ci$upper)
            )
            write.csv(ci_df, file = ci_file, row.names = FALSE)
    """

    with tempfile.TemporaryDirectory(prefix="statgpu_vs_r_") as td:
        td_path = Path(td)
        x_file = td_path / "x.csv"
        y_file = td_path / "y.csv"
        points_file = td_path / "points.csv"
        points_ci_file = td_path / "points_ci.csv"
        pred_file = td_path / "pred.csv"
        timing_file = td_path / "timing.csv"
        ci_file = td_path / "ci.csv"
        r_file = td_path / "run_compare.R"

        np.savetxt(x_file, np.asarray(payload["x"], dtype=np.float64), fmt="%.17g")
        np.savetxt(y_file, np.asarray(payload["y"], dtype=np.float64), fmt="%.17g")
        np.savetxt(points_file, np.asarray(payload["points"], dtype=np.float64), fmt="%.17g")
        np.savetxt(points_ci_file, np.asarray(payload["points_ci"], dtype=np.float64), fmt="%.17g")
        r_file.write_text(r_code, encoding="utf-8")

        proc = subprocess.run(
            [
            "Rscript",
            str(r_file),
            str(x_file),
            str(y_file),
            str(points_file),
            str(points_ci_file),
                str(float(payload["r_bw_density"])),
                str(float(payload["r_bw_ksmooth"])),
                str(float(payload["r_bw_locpoly"])),
            str(int(payload["repeats"])),
            str(int(payload["warmup"])),
            str(pred_file),
            str(timing_file),
            str(int(payload["ci_resamples"])),
            str(float(payload["ci_confidence_level"])),
            str(ci_file),
            str(payload["ci_method"]),
            ],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Rscript execution failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        pred_df = np.genfromtxt(pred_file, delimiter=",", names=True, dtype=np.float64)
        ci_df = np.genfromtxt(ci_file, delimiter=",", names=True, dtype=np.float64)

        timing_map = {}
        with timing_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = str(row["method"]).strip()
                timing_map[method] = {
                    "time_ms_mean": _parse_r_float(row["time_ms_mean"]),
                    "time_ms_std": _parse_r_float(row["time_ms_std"], default=0.0),
                    "time_ms_min": _parse_r_float(row["time_ms_min"]),
                    "time_ms_max": _parse_r_float(row["time_ms_max"]),
                }

        return {
            "kde": {
                "timing_ms": timing_map["kde"],
                "pred": np.asarray(pred_df["kde"], dtype=np.float64).tolist(),
            },
            "nw": {
                "timing_ms": timing_map["nw"],
                "pred": np.asarray(pred_df["nw"], dtype=np.float64).tolist(),
            },
            "local_linear": {
                "timing_ms": timing_map["local_linear"],
                "pred": np.asarray(pred_df["local_linear"], dtype=np.float64).tolist(),
            },
            "kde_ci": {
                "timing_ms": timing_map["kde_ci"],
                "estimate": np.asarray(ci_df["estimate"], dtype=np.float64).tolist(),
                "lower": np.asarray(ci_df["lower"], dtype=np.float64).tolist(),
                "upper": np.asarray(ci_df["upper"], dtype=np.float64).tolist(),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare statgpu nonparametric methods against R")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=1200)
    parser.add_argument(
        "--bandwidth-abs",
        type=float,
        default=0.45,
        help="Canonical absolute bandwidth scale used across methods before conversion.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--ci-resamples", type=int, default=120)
    parser.add_argument("--ci-confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--statgpu-backend",
        type=str,
        default="numpy",
        choices=["numpy", "cupy"],
        help="Backend used for statgpu timings in this script.",
    )
    parser.add_argument(
        "--ci-method",
        type=str,
        default="normal",
        choices=["normal", "bootstrap"],
        help="CI method for KDE intervals; bootstrap is available as fallback option.",
    )
    parser.add_argument("--ci-repeats", type=int, default=1)
    parser.add_argument("--ci-warmup", type=int, default=0)
    parser.add_argument("--n-ci-eval", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))

    n_samples = int(args.n_samples)
    n_eval = int(args.n_eval)
    bw_abs = float(args.bandwidth_abs)
    if bw_abs <= 0.0:
        raise ValueError("--bandwidth-abs must be positive")

    x = np.concatenate(
        [
            rng.normal(loc=-1.0, scale=0.8, size=n_samples // 2),
            rng.normal(loc=1.2, scale=0.6, size=n_samples - n_samples // 2),
        ]
    ).astype(np.float64)

    noise = rng.normal(scale=0.25, size=n_samples)
    y = (np.sin(1.4 * x) + 0.25 * x + noise).astype(np.float64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    points = np.linspace(x_min, x_max, n_eval, dtype=np.float64)
    n_ci_eval = int(args.n_ci_eval)
    if n_ci_eval <= 0:
        raise ValueError("--n-ci-eval must be a positive integer")
    points_ci = np.linspace(x_min, x_max, n_ci_eval, dtype=np.float64)
    statgpu_backend = str(args.statgpu_backend).strip().lower()
    if statgpu_backend == "cupy" and (not cuda_available()):
        raise RuntimeError("--statgpu-backend cupy requested but CUDA is not available")

    if statgpu_backend == "cupy":
        import cupy as cp

        x_stat = cp.asarray(x)
        y_stat = cp.asarray(y)
        points_stat = cp.asarray(points)
        points_ci_stat = cp.asarray(points_ci)
        sync_cuda = True
    else:
        x_stat = x
        y_stat = y
        points_stat = points
        points_ci_stat = points_ci
        sync_cuda = False

    # Canonical bandwidth mapping:
    # - statgpu expects a factor relative to sample scale in 1D
    # - R density/locpoly use bandwidth on the x scale
    # - R ksmooth(normal) uses a window width where sd ~= 0.3706506 * bandwidth
    x_sd = float(np.std(x, ddof=1))
    if (not np.isfinite(x_sd)) or x_sd <= 0.0:
        raise ValueError("sample standard deviation must be finite and positive")

    statgpu_bw_factor = float(bw_abs / x_sd)
    r_bw_density = float(bw_abs)
    r_bw_locpoly = float(bw_abs)
    ksmooth_sd_factor = 0.3706506
    r_bw_ksmooth = float(bw_abs / ksmooth_sd_factor)

    stat_kde_timing, stat_kde_pred = _run_statgpu_repeats(
        lambda: _to_numpy_array(
            fit_kde(x_stat, bandwidth=statgpu_bw_factor, kernel="gaussian", backend=statgpu_backend)(
                points_stat
            )
        ),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        sync_cuda=sync_cuda,
    )

    stat_nw_timing, stat_nw_pred = _run_statgpu_repeats(
        lambda: _to_numpy_array(
            fit_kernel_regression(
                x_stat,
                y_stat,
                bandwidth=statgpu_bw_factor,
                kernel="gaussian",
                regression="nw",
                backend=statgpu_backend,
            ).predict(points_stat)
        ),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        sync_cuda=sync_cuda,
    )

    stat_ll_timing, stat_ll_pred = _run_statgpu_repeats(
        lambda: _to_numpy_array(
            fit_kernel_regression(
                x_stat,
                y_stat,
                bandwidth=statgpu_bw_factor,
                kernel="gaussian",
                regression="local_linear",
                backend=statgpu_backend,
            ).predict(points_stat)
        ),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        sync_cuda=sync_cuda,
    )

    stat_ci_timing, stat_ci_out = _run_statgpu_kde_ci_repeats(
        x_stat,
        points_ci_stat,
        bandwidth_factor=statgpu_bw_factor,
        n_resamples=int(args.ci_resamples),
        confidence_level=float(args.ci_confidence_level),
        random_state=int(args.seed),
        ci_method=str(args.ci_method),
        backend=statgpu_backend,
        repeats=int(args.ci_repeats),
        warmup=int(args.ci_warmup),
    )

    scipy_ci_timing = None
    scipy_ci_out = None
    scipy_ci_error = None
    try:
        scipy_ci_timing, scipy_ci_out = _run_scipy_kde_ci_repeats(
            x,
            points_ci,
            bandwidth_factor=statgpu_bw_factor,
            n_resamples=int(args.ci_resamples),
            confidence_level=float(args.ci_confidence_level),
            random_state=int(args.seed),
            ci_method=str(args.ci_method),
            repeats=int(args.ci_repeats),
            warmup=int(args.ci_warmup),
        )
    except Exception as exc:
        scipy_ci_error = str(exc)

    r_payload = {
        "x": x.tolist(),
        "y": y.tolist(),
        "points": points.tolist(),
        "points_ci": points_ci.tolist(),
        "r_bw_density": r_bw_density,
        "r_bw_ksmooth": r_bw_ksmooth,
        "r_bw_locpoly": r_bw_locpoly,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "ci_resamples": int(args.ci_resamples),
        "ci_confidence_level": float(args.ci_confidence_level),
        "ci_method": str(args.ci_method),
    }
    r_out = _run_r(r_payload)

    r_kde_pred = np.asarray(r_out["kde"]["pred"], dtype=np.float64)
    r_nw_pred = np.asarray(r_out["nw"]["pred"], dtype=np.float64)
    r_ll_pred = np.asarray(r_out["local_linear"]["pred"], dtype=np.float64)
    r_ci_est = np.asarray(r_out["kde_ci"]["estimate"], dtype=np.float64)
    r_ci_low = np.asarray(r_out["kde_ci"]["lower"], dtype=np.float64)
    r_ci_up = np.asarray(r_out["kde_ci"]["upper"], dtype=np.float64)

    stat_ci_est = np.asarray(stat_ci_out["estimate"], dtype=np.float64)
    stat_ci_low = np.asarray(stat_ci_out["lower"], dtype=np.float64)
    stat_ci_up = np.asarray(stat_ci_out["upper"], dtype=np.float64)

    metrics = {
        "kde": {
            "precision": {
                "max_abs_diff": float(np.max(np.abs(stat_kde_pred - r_kde_pred))),
                "mean_abs_diff": float(np.mean(np.abs(stat_kde_pred - r_kde_pred))),
                "l2_rel_diff": _l2_rel(stat_kde_pred, r_kde_pred),
            },
            "timing_ms": {
                "statgpu": stat_kde_timing,
                "r": r_out["kde"]["timing_ms"],
                "ratio_statgpu_over_r": float(
                    stat_kde_timing["time_ms_mean"] / max(float(r_out["kde"]["timing_ms"]["time_ms_mean"]), 1e-12)
                ),
            },
        },
        "kernel_regression_nw": {
            "precision": {
                "max_abs_diff": float(np.max(np.abs(stat_nw_pred - r_nw_pred))),
                "mean_abs_diff": float(np.mean(np.abs(stat_nw_pred - r_nw_pred))),
                "l2_rel_diff": _l2_rel(stat_nw_pred, r_nw_pred),
            },
            "timing_ms": {
                "statgpu": stat_nw_timing,
                "r": r_out["nw"]["timing_ms"],
                "ratio_statgpu_over_r": float(
                    stat_nw_timing["time_ms_mean"] / max(float(r_out["nw"]["timing_ms"]["time_ms_mean"]), 1e-12)
                ),
            },
        },
        "kernel_regression_local_linear": {
            "precision": {
                "max_abs_diff": float(np.max(np.abs(stat_ll_pred - r_ll_pred))),
                "mean_abs_diff": float(np.mean(np.abs(stat_ll_pred - r_ll_pred))),
                "l2_rel_diff": _l2_rel(stat_ll_pred, r_ll_pred),
            },
            "timing_ms": {
                "statgpu": stat_ll_timing,
                "r": r_out["local_linear"]["timing_ms"],
                "ratio_statgpu_over_r": float(
                    stat_ll_timing["time_ms_mean"]
                    / max(float(r_out["local_linear"]["timing_ms"]["time_ms_mean"]), 1e-12)
                ),
            },
        },
        "kde_confidence_interval": {
            "config": {
                "n_eval": int(n_ci_eval),
                "n_resamples": int(args.ci_resamples),
                "confidence_level": float(args.ci_confidence_level),
                "method": str(args.ci_method),
            },
            "precision_vs_r": {
                "estimate": {
                    "max_abs_diff": float(np.max(np.abs(stat_ci_est - r_ci_est))),
                    "mean_abs_diff": float(np.mean(np.abs(stat_ci_est - r_ci_est))),
                    "l2_rel_diff": _l2_rel(stat_ci_est, r_ci_est),
                },
                "lower": {
                    "max_abs_diff": float(np.max(np.abs(stat_ci_low - r_ci_low))),
                    "mean_abs_diff": float(np.mean(np.abs(stat_ci_low - r_ci_low))),
                    "l2_rel_diff": _l2_rel(stat_ci_low, r_ci_low),
                },
                "upper": {
                    "max_abs_diff": float(np.max(np.abs(stat_ci_up - r_ci_up))),
                    "mean_abs_diff": float(np.mean(np.abs(stat_ci_up - r_ci_up))),
                    "l2_rel_diff": _l2_rel(stat_ci_up, r_ci_up),
                },
            },
            "timing_ms": {
                "statgpu": stat_ci_timing,
                "r": r_out["kde_ci"]["timing_ms"],
                "ratio_statgpu_over_r": float(
                    stat_ci_timing["time_ms_mean"]
                    / max(float(r_out["kde_ci"]["timing_ms"]["time_ms_mean"]), 1e-12)
                ),
            },
        },
    }

    if scipy_ci_out is not None and scipy_ci_timing is not None:
        scipy_ci_est = np.asarray(scipy_ci_out["estimate"], dtype=np.float64)
        scipy_ci_low = np.asarray(scipy_ci_out["lower"], dtype=np.float64)
        scipy_ci_up = np.asarray(scipy_ci_out["upper"], dtype=np.float64)
        metrics["kde_confidence_interval"]["precision_vs_scipy"] = {
            "estimate": {
                "max_abs_diff": float(np.max(np.abs(stat_ci_est - scipy_ci_est))),
                "mean_abs_diff": float(np.mean(np.abs(stat_ci_est - scipy_ci_est))),
                "l2_rel_diff": _l2_rel(stat_ci_est, scipy_ci_est),
            },
            "lower": {
                "max_abs_diff": float(np.max(np.abs(stat_ci_low - scipy_ci_low))),
                "mean_abs_diff": float(np.mean(np.abs(stat_ci_low - scipy_ci_low))),
                "l2_rel_diff": _l2_rel(stat_ci_low, scipy_ci_low),
            },
            "upper": {
                "max_abs_diff": float(np.max(np.abs(stat_ci_up - scipy_ci_up))),
                "mean_abs_diff": float(np.mean(np.abs(stat_ci_up - scipy_ci_up))),
                "l2_rel_diff": _l2_rel(stat_ci_up, scipy_ci_up),
            },
        }
        metrics["kde_confidence_interval"]["timing_ms"]["scipy"] = scipy_ci_timing
        metrics["kde_confidence_interval"]["timing_ms"]["ratio_statgpu_over_scipy"] = float(
            stat_ci_timing["time_ms_mean"] / max(float(scipy_ci_timing["time_ms_mean"]), 1e-12)
        )
    else:
        metrics["kde_confidence_interval"]["precision_vs_scipy"] = {
            "skipped": True,
            "reason": scipy_ci_error or "SciPy not available",
        }

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_samples": n_samples,
            "n_eval": n_eval,
            "bandwidth_abs": bw_abs,
            "x_sd": x_sd,
            "statgpu_bandwidth_factor": statgpu_bw_factor,
            "statgpu_backend": statgpu_backend,
            "r_bw_density": r_bw_density,
            "r_bw_ksmooth": r_bw_ksmooth,
            "r_bw_locpoly": r_bw_locpoly,
            "ksmooth_sd_factor": ksmooth_sd_factor,
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "ci_resamples": int(args.ci_resamples),
            "ci_confidence_level": float(args.ci_confidence_level),
            "ci_method": str(args.ci_method),
            "ci_repeats": int(args.ci_repeats),
            "ci_warmup": int(args.ci_warmup),
            "n_ci_eval": int(n_ci_eval),
            "seed": int(args.seed),
            "kde_r_method": "density(kernel='gaussian')",
            "nw_r_method": "ksmooth(kernel='normal')",
            "local_linear_r_method": "KernSmooth::locpoly(degree=1)",
        },
        "metrics": metrics,
    }

    if args.json_out:
        out_path = Path(args.json_out)
    else:
        out_path = Path("results") / f"nonparametric_vs_r_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(out_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
