"""
Benchmark runtime comparison for LassoCV implementations.

Compares:
- sklearn LassoCV (CPU)
- statgpu LassoCV (CPU)
- statgpu LassoCV (GPU, if CUDA is available)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# Ensure local imports work when running from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _maybe_sync_cuda() -> None:
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _cuda_available() -> bool:
    try:
        from statgpu._config import cuda_available

        return bool(cuda_available())
    except Exception:
        return False


def _make_correlated_data(
    *,
    seed: int,
    n_samples: int,
    n_features: int,
    n_signal: int,
    rho: float,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    idx = np.arange(int(n_features))
    cov = float(rho) ** np.abs(np.subtract.outer(idx, idx))
    chol = np.linalg.cholesky(cov)

    X = rng.normal(size=(int(n_samples), int(n_features))) @ chol.T

    beta = np.zeros(int(n_features), dtype=np.float64)
    beta[: int(n_signal)] = np.linspace(2.5, 0.8, int(n_signal))
    y = X @ beta + rng.normal(scale=float(noise_scale), size=int(n_samples))

    return (
        X.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        beta.astype(np.float64, copy=False),
    )


def _make_shared_alpha_grid(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_alphas: int,
    alpha_min_ratio: float,
) -> np.ndarray:
    """Build one alpha grid and share it across all implementations for fair CV comparison."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must have the same number of rows as X")

    n_samples = int(X_arr.shape[0])
    y_centered = y_arr - np.mean(y_arr)
    corr = np.abs(X_arr.T @ y_centered) / float(max(1, n_samples))

    alpha_max = float(np.max(corr)) if corr.size else 1.0
    alpha_max = max(alpha_max, 1e-8)

    n_alpha = max(1, int(n_alphas))
    if n_alpha == 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-8)
    return np.geomspace(alpha_max, alpha_min, num=n_alpha).astype(np.float64)


def _make_shared_cv_splits(
    *,
    n_samples: int,
    n_splits: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic KFold-like splits (shuffle=False) shared across implementations."""
    n = int(n_samples)
    k = max(2, min(int(n_splits), n))

    indices = np.arange(n, dtype=np.int64)
    fold_sizes = np.full(k, n // k, dtype=np.int64)
    fold_sizes[: n % k] += 1

    folds: List[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        current = stop

        if train_idx.size == 0 or val_idx.size == 0:
            continue
        folds.append((train_idx, val_idx))

    if not folds:
        all_idx = np.arange(n, dtype=np.int64)
        return [(all_idx, all_idx)]

    return folds


@dataclass
class RunResult:
    seed: int
    time_ms: float
    alpha: Optional[float]
    n_iter: Optional[int]
    train_mse: Optional[float]
    test_mse: Optional[float]
    test_mse_noiseless: Optional[float]
    coef_l2_rel: Optional[float]
    coef_l1_rel: Optional[float]
    support_precision: Optional[float]
    support_recall: Optional[float]
    support_f1: Optional[float]
    support_jaccard: Optional[float]


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _extract_n_iter(n_iter_raw) -> Optional[int]:
    if n_iter_raw is None:
        return None
    arr = np.asarray(n_iter_raw)
    if arr.ndim == 0:
        return int(arr.item())
    if arr.size == 0:
        return None
    return int(np.max(arr))


def _support_metrics(
    coef: np.ndarray,
    beta_true: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    coef_mask = np.abs(coef) > float(threshold)
    true_mask = np.abs(beta_true) > float(threshold)

    tp = int(np.sum(coef_mask & true_mask))
    fp = int(np.sum(coef_mask & (~true_mask)))
    fn = int(np.sum((~coef_mask) & true_mask))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if precision + recall > 0.0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    union = int(np.sum(coef_mask | true_mask))
    jaccard = float(tp / union) if union > 0 else 0.0

    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "support_jaccard": jaccard,
    }


def _evaluate_model_accuracy(
    model,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    beta_true: np.ndarray,
    coef_support_threshold: float,
) -> Dict[str, float]:
    pred_train = _to_numpy(model.predict(X_train)).reshape(-1)
    pred_test = _to_numpy(model.predict(X_test)).reshape(-1)

    y_test_noiseless = X_test @ beta_true

    coef = _to_numpy(getattr(model, "coef_", np.zeros(X_train.shape[1], dtype=np.float64))).reshape(-1)
    if coef.shape[0] != beta_true.shape[0]:
        coef = np.resize(coef, beta_true.shape[0])

    l2_denom = float(np.linalg.norm(beta_true))
    l1_denom = float(np.sum(np.abs(beta_true)))

    l2_denom = l2_denom if l2_denom > 1e-12 else 1.0
    l1_denom = l1_denom if l1_denom > 1e-12 else 1.0

    out = {
        "train_mse": float(np.mean((y_train - pred_train) ** 2)),
        "test_mse": float(np.mean((y_test - pred_test) ** 2)),
        "test_mse_noiseless": float(np.mean((y_test_noiseless - pred_test) ** 2)),
        "coef_l2_rel": float(np.linalg.norm(coef - beta_true) / l2_denom),
        "coef_l1_rel": float(np.sum(np.abs(coef - beta_true)) / l1_denom),
    }
    out.update(_support_metrics(coef, beta_true, threshold=float(coef_support_threshold)))
    return out


def _benchmark_fit(
    model_builder: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    *,
    warmup: int,
    repeats: int,
    sync_cuda: bool,
    eval_data: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    for _ in range(max(0, int(warmup))):
        model = model_builder()
        if sync_cuda:
            _maybe_sync_cuda()
        model.fit(X, y)
        if sync_cuda:
            _maybe_sync_cuda()

    times_ms: List[float] = []
    alpha: Optional[float] = None
    n_iter: Optional[int] = None
    accuracy: Dict[str, float] = {}
    model = None

    for _ in range(max(1, int(repeats))):
        model = model_builder()
        if sync_cuda:
            _maybe_sync_cuda()

        t0 = time.perf_counter()
        model.fit(X, y)
        if sync_cuda:
            _maybe_sync_cuda()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

        alpha_raw = getattr(model, "alpha_", None)
        alpha = float(alpha_raw) if alpha_raw is not None else None

        n_iter_raw = getattr(model, "n_iter_", None)
        n_iter = _extract_n_iter(n_iter_raw)

    if eval_data is not None and model is not None:
        accuracy = _evaluate_model_accuracy(
            model,
            X_train=np.asarray(eval_data["X_train"], dtype=np.float64),
            y_train=np.asarray(eval_data["y_train"], dtype=np.float64).reshape(-1),
            X_test=np.asarray(eval_data["X_test"], dtype=np.float64),
            y_test=np.asarray(eval_data["y_test"], dtype=np.float64).reshape(-1),
            beta_true=np.asarray(eval_data["beta_true"], dtype=np.float64).reshape(-1),
            coef_support_threshold=float(eval_data["coef_support_threshold"]),
        )

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "time_ms_mean": float(np.mean(arr)),
        "time_ms_std": float(np.std(arr, ddof=0)),
        "time_ms": [float(v) for v in arr.tolist()],
        "alpha": alpha,
        "n_iter": n_iter,
        **accuracy,
    }


def _aggregate_runs(runs: List[RunResult]) -> Dict[str, float]:
    if not runs:
        return {"n_runs": 0}

    out = {
        "n_runs": int(len(runs)),
    }

    metric_fields = [
        "time_ms",
        "train_mse",
        "test_mse",
        "test_mse_noiseless",
        "coef_l2_rel",
        "coef_l1_rel",
        "support_precision",
        "support_recall",
        "support_f1",
        "support_jaccard",
    ]

    for field_name in metric_fields:
        vals = [float(getattr(r, field_name)) for r in runs if getattr(r, field_name) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{field_name}_mean"] = float(np.mean(arr))
        out[f"{field_name}_std"] = float(np.std(arr, ddof=0))
        out[f"{field_name}_min"] = float(np.min(arr))
        out[f"{field_name}_max"] = float(np.max(arr))

    return out


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    requested = {str(m).strip().lower() for m in args.methods}

    SkLassoCV = None
    StatgpuLassoCV = None

    sklearn_error: Optional[str] = None
    statgpu_error: Optional[str] = None

    if "sklearn" in requested:
        try:
            from sklearn.linear_model import LassoCV as _SkLassoCV

            SkLassoCV = _SkLassoCV
        except Exception as exc:
            sklearn_error = str(exc)

    if "statgpu_cpu" in requested or "statgpu_gpu" in requested:
        try:
            from statgpu.linear_model._lasso import LassoCV as _StatgpuLassoCV

            StatgpuLassoCV = _StatgpuLassoCV
        except Exception as exc:
            statgpu_error = str(exc)

    has_cuda = _cuda_available() if "statgpu_gpu" in requested else False

    methods: Dict[str, Optional[Dict[str, object]]] = {
        "sklearn_lassocv_cpu": ({"runs": []} if ("sklearn" in requested and SkLassoCV is not None) else None),
        "statgpu_lassocv_cpu": ({"runs": []} if ("statgpu_cpu" in requested and StatgpuLassoCV is not None) else None),
        "statgpu_lassocv_gpu": ({"runs": []} if ("statgpu_gpu" in requested and StatgpuLassoCV is not None and has_cuda) else None),
    }

    for seed in args.seeds:
        X, y, beta_true = _make_correlated_data(
            seed=int(seed),
            n_samples=int(args.n_samples),
            n_features=int(args.n_features),
            n_signal=int(args.n_signal),
            rho=float(args.rho),
            noise_scale=float(args.noise_scale),
        )

        X_test, y_test, beta_test = _make_correlated_data(
            seed=int(seed) + 1_000_003,
            n_samples=int(args.n_test_samples),
            n_features=int(args.n_features),
            n_signal=int(args.n_signal),
            rho=float(args.rho),
            noise_scale=float(args.noise_scale),
        )
        if beta_test.shape != beta_true.shape:
            beta_true_eval = beta_true
        else:
            beta_true_eval = beta_test

        eval_data = {
            "X_train": X,
            "y_train": y,
            "X_test": X_test,
            "y_test": y_test,
            "beta_true": beta_true_eval,
            "coef_support_threshold": float(args.coef_support_threshold),
        }

        shared_alphas = None
        if bool(args.shared_alpha_grid):
            shared_alphas = _make_shared_alpha_grid(
                X,
                y,
                n_alphas=int(args.n_alphas),
                alpha_min_ratio=float(args.alpha_min_ratio),
            )

        shared_cv_splits = None
        if bool(args.shared_cv_splits):
            shared_cv_splits = _make_shared_cv_splits(
                n_samples=int(X.shape[0]),
                n_splits=int(args.cv),
            )

        if methods["sklearn_lassocv_cpu"] is not None and SkLassoCV is not None:
            sk = _benchmark_fit(
                lambda: SkLassoCV(
                    n_alphas=int(args.n_alphas),
                    alphas=shared_alphas,
                    cv=shared_cv_splits if shared_cv_splits is not None else int(args.cv),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    random_state=int(seed),
                    n_jobs=int(args.n_jobs),
                ),
                X,
                y,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                sync_cuda=False,
                eval_data=eval_data,
            )
            methods["sklearn_lassocv_cpu"]["runs"].append(
                asdict(
                    RunResult(
                        seed=int(seed),
                        time_ms=float(sk["time_ms_mean"]),
                        alpha=sk.get("alpha"),
                        n_iter=sk.get("n_iter"),
                        train_mse=sk.get("train_mse"),
                        test_mse=sk.get("test_mse"),
                        test_mse_noiseless=sk.get("test_mse_noiseless"),
                        coef_l2_rel=sk.get("coef_l2_rel"),
                        coef_l1_rel=sk.get("coef_l1_rel"),
                        support_precision=sk.get("support_precision"),
                        support_recall=sk.get("support_recall"),
                        support_f1=sk.get("support_f1"),
                        support_jaccard=sk.get("support_jaccard"),
                    )
                )
            )

        if methods["statgpu_lassocv_cpu"] is not None and StatgpuLassoCV is not None:
            sg_cpu = _benchmark_fit(
                lambda: StatgpuLassoCV(
                    alphas=shared_alphas,
                    cv=int(args.cv),
                    cv_splits=shared_cv_splits,
                    n_alphas=int(args.n_alphas),
                    alpha_min_ratio=float(args.alpha_min_ratio),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    fit_intercept=True,
                    compute_inference=False,
                    device="cpu",
                    n_jobs=int(args.n_jobs),
                    random_state=int(seed),
                    cpu_solver=str(args.cpu_solver),
                    method=str(args.statgpu_lassocv_method),
                    cd_kkt_check_every=args.statgpu_cd_kkt_check_every,
                ),
                X,
                y,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                sync_cuda=False,
                eval_data=eval_data,
            )
            methods["statgpu_lassocv_cpu"]["runs"].append(
                asdict(
                    RunResult(
                        seed=int(seed),
                        time_ms=float(sg_cpu["time_ms_mean"]),
                        alpha=sg_cpu.get("alpha"),
                        n_iter=sg_cpu.get("n_iter"),
                        train_mse=sg_cpu.get("train_mse"),
                        test_mse=sg_cpu.get("test_mse"),
                        test_mse_noiseless=sg_cpu.get("test_mse_noiseless"),
                        coef_l2_rel=sg_cpu.get("coef_l2_rel"),
                        coef_l1_rel=sg_cpu.get("coef_l1_rel"),
                        support_precision=sg_cpu.get("support_precision"),
                        support_recall=sg_cpu.get("support_recall"),
                        support_f1=sg_cpu.get("support_f1"),
                        support_jaccard=sg_cpu.get("support_jaccard"),
                    )
                )
            )

        if methods["statgpu_lassocv_gpu"] is not None and StatgpuLassoCV is not None:
            sg_gpu = _benchmark_fit(
                lambda: StatgpuLassoCV(
                    alphas=shared_alphas,
                    cv=int(args.cv),
                    cv_splits=shared_cv_splits,
                    n_alphas=int(args.n_alphas),
                    alpha_min_ratio=float(args.alpha_min_ratio),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    fit_intercept=True,
                    compute_inference=False,
                    device="cuda",
                    n_jobs=int(args.n_jobs),
                    random_state=int(seed),
                    cpu_solver=str(args.cpu_solver),
                    method=str(args.statgpu_lassocv_method),
                    cd_kkt_check_every=args.statgpu_cd_kkt_check_every,
                ),
                X,
                y,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                sync_cuda=True,
                eval_data=eval_data,
            )
            methods["statgpu_lassocv_gpu"]["runs"].append(
                asdict(
                    RunResult(
                        seed=int(seed),
                        time_ms=float(sg_gpu["time_ms_mean"]),
                        alpha=sg_gpu.get("alpha"),
                        n_iter=sg_gpu.get("n_iter"),
                        train_mse=sg_gpu.get("train_mse"),
                        test_mse=sg_gpu.get("test_mse"),
                        test_mse_noiseless=sg_gpu.get("test_mse_noiseless"),
                        coef_l2_rel=sg_gpu.get("coef_l2_rel"),
                        coef_l1_rel=sg_gpu.get("coef_l1_rel"),
                        support_precision=sg_gpu.get("support_precision"),
                        support_recall=sg_gpu.get("support_recall"),
                        support_f1=sg_gpu.get("support_f1"),
                        support_jaccard=sg_gpu.get("support_jaccard"),
                    )
                )
            )

    for payload in methods.values():
        if payload is None:
            continue
        payload["aggregate"] = _aggregate_runs(
            [
                RunResult(
                    seed=int(run["seed"]),
                    time_ms=float(run["time_ms"]),
                    alpha=run.get("alpha"),
                    n_iter=run.get("n_iter"),
                    train_mse=run.get("train_mse"),
                    test_mse=run.get("test_mse"),
                    test_mse_noiseless=run.get("test_mse_noiseless"),
                    coef_l2_rel=run.get("coef_l2_rel"),
                    coef_l1_rel=run.get("coef_l1_rel"),
                    support_precision=run.get("support_precision"),
                    support_recall=run.get("support_recall"),
                    support_f1=run.get("support_f1"),
                    support_jaccard=run.get("support_jaccard"),
                )
                for run in payload["runs"]
            ]
        )

    pairwise: Dict[str, float] = {}
    sk_payload = methods.get("sklearn_lassocv_cpu")
    cpu_payload = methods.get("statgpu_lassocv_cpu")
    gpu_payload = methods.get("statgpu_lassocv_gpu")

    if sk_payload is not None and cpu_payload is not None:
        sk_base = sk_payload["aggregate"].get("time_ms_mean")
        cpu_base = cpu_payload["aggregate"].get("time_ms_mean")
        if isinstance(sk_base, (int, float)) and sk_base > 0:
            pairwise["statgpu_cpu_over_sklearn"] = cpu_base / sk_base
            sk_test = sk_payload["aggregate"].get("test_mse_mean")
            cpu_test = cpu_payload["aggregate"].get("test_mse_mean")
            if isinstance(sk_test, (int, float)) and sk_test > 0 and isinstance(cpu_test, (int, float)):
                pairwise["statgpu_cpu_test_mse_over_sklearn"] = cpu_test / sk_test

            sk_coef = sk_payload["aggregate"].get("coef_l2_rel_mean")
            cpu_coef = cpu_payload["aggregate"].get("coef_l2_rel_mean")
            if isinstance(sk_coef, (int, float)) and sk_coef > 0 and isinstance(cpu_coef, (int, float)):
                pairwise["statgpu_cpu_coef_l2_rel_over_sklearn"] = cpu_coef / sk_coef

            sk_f1 = sk_payload["aggregate"].get("support_f1_mean")
            cpu_f1 = cpu_payload["aggregate"].get("support_f1_mean")
            if isinstance(sk_f1, (int, float)) and isinstance(cpu_f1, (int, float)):
                pairwise["statgpu_cpu_support_f1_minus_sklearn"] = cpu_f1 - sk_f1

            if gpu_payload is not None:
                gpu_base = gpu_payload["aggregate"].get("time_ms_mean")
                if isinstance(gpu_base, (int, float)):
                    pairwise["statgpu_gpu_over_sklearn"] = gpu_base / sk_base

                gpu_test = gpu_payload["aggregate"].get("test_mse_mean")
                if isinstance(sk_test, (int, float)) and sk_test > 0 and isinstance(gpu_test, (int, float)):
                    pairwise["statgpu_gpu_test_mse_over_sklearn"] = gpu_test / sk_test

                gpu_coef = gpu_payload["aggregate"].get("coef_l2_rel_mean")
                if isinstance(sk_coef, (int, float)) and sk_coef > 0 and isinstance(gpu_coef, (int, float)):
                    pairwise["statgpu_gpu_coef_l2_rel_over_sklearn"] = gpu_coef / sk_coef

                gpu_f1 = gpu_payload["aggregate"].get("support_f1_mean")
                if isinstance(sk_f1, (int, float)) and isinstance(gpu_f1, (int, float)):
                    pairwise["statgpu_gpu_support_f1_minus_sklearn"] = gpu_f1 - sk_f1

        if gpu_payload is not None and isinstance(cpu_base, (int, float)) and cpu_base > 0:
            gpu_base = gpu_payload["aggregate"].get("time_ms_mean")
            if isinstance(gpu_base, (int, float)):
                pairwise["statgpu_gpu_over_statgpu_cpu"] = gpu_base / cpu_base

            cpu_test = cpu_payload["aggregate"].get("test_mse_mean")
            gpu_test = gpu_payload["aggregate"].get("test_mse_mean")
            if isinstance(cpu_test, (int, float)) and cpu_test > 0 and isinstance(gpu_test, (int, float)):
                pairwise["statgpu_gpu_test_mse_over_statgpu_cpu"] = gpu_test / cpu_test

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "cuda_available": bool(has_cuda),
            "sklearn_available": bool(SkLassoCV is not None),
            "statgpu_available": bool(StatgpuLassoCV is not None),
        },
        "config": {
            "seeds": [int(s) for s in args.seeds],
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "n_signal": int(args.n_signal),
            "n_test_samples": int(args.n_test_samples),
            "noise_scale": float(args.noise_scale),
            "rho": float(args.rho),
            "cv": int(args.cv),
            "n_alphas": int(args.n_alphas),
            "alpha_min_ratio": float(args.alpha_min_ratio),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "n_jobs": int(args.n_jobs),
            "cpu_solver": str(args.cpu_solver),
            "statgpu_lassocv_method": str(args.statgpu_lassocv_method),
            "statgpu_cd_kkt_check_every": (
                None
                if args.statgpu_cd_kkt_check_every is None
                else int(args.statgpu_cd_kkt_check_every)
            ),
            "coef_support_threshold": float(args.coef_support_threshold),
            "shared_alpha_grid": bool(args.shared_alpha_grid),
            "shared_cv_splits": bool(args.shared_cv_splits),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "methods": methods,
        "pairwise": pairwise,
        "errors": {
            "sklearn": sklearn_error,
            "statgpu": statgpu_error,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LassoCV runtime across implementations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260406, 20260407, 20260408])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_features", type=int, default=100)
    parser.add_argument("--n_signal", type=int, default=12)
    parser.add_argument("--n_test_samples", type=int, default=2000)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.3)

    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_alphas", type=int, default=12)
    parser.add_argument("--alpha_min_ratio", type=float, default=1e-3)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--cpu_solver", type=str, default="coordinate_descent", choices=["coordinate_descent", "fista"])
    parser.add_argument(
        "--statgpu_lassocv_method",
        type=str,
        default="standard",
        choices=["standard", "glmnet"],
        help="LassoCV optimization profile used by statgpu backends.",
    )
    parser.add_argument(
        "--statgpu_cd_kkt_check_every",
        type=int,
        default=None,
        help="Optional KKT full-scan cadence for statgpu coordinate descent in LassoCV.",
    )
    parser.add_argument("--coef_support_threshold", type=float, default=1e-8)
    parser.add_argument(
        "--shared_alpha_grid",
        action="store_true",
        help="Force sklearn/statgpu to use the exact same alpha candidate grid per seed.",
    )
    parser.add_argument(
        "--shared_cv_splits",
        action="store_true",
        help="Force sklearn/statgpu to use the exact same CV splits per seed.",
    )

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["sklearn", "statgpu_cpu", "statgpu_gpu"],
        choices=["sklearn", "statgpu_cpu", "statgpu_gpu"],
        help="Subset of methods to run",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path. Defaults to results/lassocv_runtime_compare_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_benchmark(args)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("results") / f"lassocv_runtime_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    summary = {
        "output": str(output_path),
        "environment": out["environment"],
        "aggregates": {
            method_name: (None if payload is None else payload.get("aggregate"))
            for method_name, payload in out["methods"].items()
        },
        "pairwise": out["pairwise"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
