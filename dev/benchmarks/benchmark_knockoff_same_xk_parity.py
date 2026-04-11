"""Parity benchmark: statgpu vs knockpy using the exact same knockoff matrix Xk."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict

import numpy as np

from statgpu.feature_selection import model_x_knockoff_filter


@dataclass
class BenchmarkConfig:
    seed: int = 20260409
    n_samples: int = 400
    n_features: int = 80
    n_signal: int = 12
    rho: float = 0.30
    noise_scale: float = 1.0
    q: float = 0.10
    method: str = "lasso_coef_diff"
    lasso_cv_impl: str = "sklearn"


def _make_correlated_data(cfg: BenchmarkConfig):
    rng = np.random.default_rng(int(cfg.seed))
    Z = rng.normal(size=(int(cfg.n_samples), int(cfg.n_features)))
    common = rng.normal(size=(int(cfg.n_samples), 1))
    X = np.sqrt(max(0.0, 1.0 - float(cfg.rho))) * Z + np.sqrt(max(0.0, float(cfg.rho))) * common

    beta = np.zeros(int(cfg.n_features), dtype=np.float64)
    n_signal = min(int(cfg.n_signal), int(cfg.n_features))
    signs = rng.choice([-1.0, 1.0], size=n_signal)
    beta[:n_signal] = signs * rng.uniform(0.9, 2.2, size=n_signal)

    y = X @ beta + rng.normal(scale=float(cfg.noise_scale), size=int(cfg.n_samples))
    truth = np.where(beta != 0.0)[0].astype(np.int64)
    return X.astype(np.float64), y.astype(np.float64), truth


def _selection_metrics(selected, truth) -> Dict[str, float]:
    sel = set(int(i) for i in np.asarray(selected).reshape(-1).tolist())
    sig = set(int(i) for i in np.asarray(truth).reshape(-1).tolist())

    tp = len(sel.intersection(sig))
    fp = len(sel.difference(sig))
    fn = len(sig.difference(sel))

    n_sel = len(sel)
    precision = float(tp / max(1, n_sel))
    recall = float(tp / max(1, len(sig)))
    fdp = float(fp / max(1, n_sel))
    f1 = float((2.0 * precision * recall) / max(1e-12, precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard_truth = float(tp / max(1, len(sel.union(sig))))

    return {
        "n_selected": int(n_sel),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "fdp": fdp,
        "f1": f1,
        "jaccard_truth": jaccard_truth,
    }


def _jaccard(a, b) -> float:
    sa = set(int(i) for i in np.asarray(a).reshape(-1).tolist())
    sb = set(int(i) for i in np.asarray(b).reshape(-1).tolist())
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    return float(len(sa.intersection(sb)) / max(1, len(sa.union(sb))))


def _build_shared_xk(X: np.ndarray, seed: int) -> np.ndarray:
    knockoffs = importlib.import_module("knockpy.knockoffs")
    utilities = importlib.import_module("knockpy.utilities")

    Sigma, _ = utilities.estimate_covariance(X, tol=1e-2, shrinkage="ledoitwolf")
    mu = np.mean(X, axis=0)

    sampler = knockoffs.GaussianSampler(
        X=X,
        mu=mu,
        Sigma=Sigma,
        method="mvr",
    )

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        Xk = sampler.sample_knockoffs()
    finally:
        np.random.set_state(state)

    return np.asarray(Xk, dtype=np.float64)


def _run_statgpu_same_xk(X, Xk, y, cfg: BenchmarkConfig):
    t0 = time.perf_counter()
    result = model_x_knockoff_filter(
        X,
        y,
        q=float(cfg.q),
        method=str(cfg.method),
        fdr_control="knockoff_plus",
        random_state=int(cfg.seed),
        backend="numpy",
        Xk=Xk,
        compat_mode="knockpy",
        lasso_cv_impl=str(cfg.lasso_cv_impl),
    )
    elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
    return {
        "time_ms": elapsed_ms,
        "W": np.asarray(result.W, dtype=np.float64),
        "selected": np.asarray(result.selected_features, dtype=np.int64),
        "threshold": float(result.threshold),
        "estimated_fdr": float(result.estimated_fdr),
        "metadata": dict(result.metadata),
    }


def _run_knockpy_same_xk(X, Xk, y, cfg: BenchmarkConfig):
    knockoff_filter_mod = importlib.import_module("knockpy.knockoff_filter")
    KnockoffFilter = knockoff_filter_mod.KnockoffFilter

    state = np.random.get_state()
    np.random.seed(int(cfg.seed))
    try:
        t0 = time.perf_counter()
        kf = KnockoffFilter(ksampler="gaussian", fstat="lasso")
        rejections = kf.forward(
            X=np.asarray(X, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64).reshape(-1),
            Xk=np.asarray(Xk, dtype=np.float64),
            fdr=float(cfg.q),
            shrinkage="ledoitwolf",
        )
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
    finally:
        np.random.set_state(state)

    rej = np.asarray(rejections).reshape(-1)
    selected = np.where(rej > 0)[0].astype(np.int64)

    return {
        "time_ms": elapsed_ms,
        "W": np.asarray(kf.W, dtype=np.float64),
        "selected": selected,
        "threshold": float(kf.threshold) if getattr(kf, "threshold", None) is not None else None,
        "estimated_fdr": None,
    }


def run(cfg: BenchmarkConfig, output_path: Path):
    try:
        importlib.import_module("knockpy")
    except Exception as exc:
        raise RuntimeError(
            "This parity benchmark requires knockpy. Install with: pip install knockpy"
        ) from exc

    X, y, truth = _make_correlated_data(cfg)
    Xk = _build_shared_xk(X, seed=int(cfg.seed))

    statgpu_out = _run_statgpu_same_xk(X, Xk, y, cfg)
    knockpy_out = _run_knockpy_same_xk(X, Xk, y, cfg)

    W_sg = statgpu_out["W"]
    W_kp = knockpy_out["W"]
    if W_sg.shape != W_kp.shape:
        raise RuntimeError(f"W shape mismatch: statgpu={W_sg.shape}, knockpy={W_kp.shape}")

    w_corr = float(np.corrcoef(W_sg, W_kp)[0, 1]) if np.std(W_sg) > 0 and np.std(W_kp) > 0 else 1.0
    w_mae = float(np.mean(np.abs(W_sg - W_kp)))
    w_max_abs = float(np.max(np.abs(W_sg - W_kp)))

    thr_sg = statgpu_out["threshold"]
    thr_kp = knockpy_out["threshold"]
    if thr_kp is None or not np.isfinite(float(thr_kp)) or not np.isfinite(float(thr_sg)):
        thr_abs_diff = None
    else:
        thr_abs_diff = float(abs(float(thr_sg) - float(thr_kp)))

    parity = {
        "w_corr": w_corr,
        "w_mae": w_mae,
        "w_max_abs": w_max_abs,
        "threshold_abs_diff": thr_abs_diff,
        "selected_jaccard": _jaccard(statgpu_out["selected"], knockpy_out["selected"]),
        "selected_overlap": int(
            len(set(int(i) for i in statgpu_out["selected"]).intersection(set(int(i) for i in knockpy_out["selected"])))
        ),
    }

    out = {
        "date": str(date.today()),
        "config": {
            "seed": int(cfg.seed),
            "n_samples": int(cfg.n_samples),
            "n_features": int(cfg.n_features),
            "n_signal": int(cfg.n_signal),
            "rho": float(cfg.rho),
            "noise_scale": float(cfg.noise_scale),
            "q": float(cfg.q),
            "method": str(cfg.method),
            "lasso_cv_impl": str(cfg.lasso_cv_impl),
            "compat_mode": "knockpy",
            "shared_xk": True,
        },
        "statgpu": {
            "time_ms": float(statgpu_out["time_ms"]),
            "threshold": float(statgpu_out["threshold"]),
            "metrics": _selection_metrics(statgpu_out["selected"], truth),
            "metadata": statgpu_out["metadata"],
        },
        "knockpy": {
            "time_ms": float(knockpy_out["time_ms"]),
            "threshold": None if knockpy_out["threshold"] is None else float(knockpy_out["threshold"]),
            "metrics": _selection_metrics(knockpy_out["selected"], truth),
        },
        "parity": parity,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote parity benchmark to: {output_path}")
    print(
        "Parity summary: "
        f"w_corr={parity['w_corr']:.6f}, "
        f"w_mae={parity['w_mae']:.6f}, "
        f"selected_jaccard={parity['selected_jaccard']:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare statgpu and knockpy on the same knockoff matrix Xk")
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-features", type=int, default=80)
    parser.add_argument("--n-signal", type=int, default=12)
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=0.10)
    parser.add_argument("--method", type=str, default="lasso_coef_diff")
    parser.add_argument("--lasso-cv-impl", type=str, default="sklearn", choices=["sklearn", "statgpu"])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        n_features=int(args.n_features),
        n_signal=int(args.n_signal),
        rho=float(args.rho),
        noise_scale=float(args.noise_scale),
        q=float(args.q),
        method=str(args.method),
        lasso_cv_impl=str(args.lasso_cv_impl),
    )

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("results") / f"benchmark_knockoff_same_xk_parity_{date.today()}.json"

    run(cfg, out_path)
