"""Structured benchmark for fixed-X knockoff skeleton."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from statgpu.feature_selection import fixed_x_knockoff_filter


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _make_data(seed: int, n: int, p: int, n_signal: int, noise_scale: float):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    beta = np.zeros(p)
    beta[:n_signal] = rng.uniform(0.8, 2.0, size=n_signal) * rng.choice([-1.0, 1.0], size=n_signal)
    y = X @ beta + rng.normal(scale=noise_scale, size=n)
    return X, y, np.where(beta != 0)[0]


def _summarize_selection(selected, true_signal):
    sel = set(int(i) for i in np.asarray(selected).tolist())
    sig = set(int(i) for i in np.asarray(true_signal).tolist())

    tp = len(sel.intersection(sig))
    fp = len(sel.difference(sig))
    n_sel = len(sel)

    empirical_fdp = float(fp / max(1, n_sel))
    recall = float(tp / max(1, len(sig)))
    return {
        "n_selected": int(n_sel),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "empirical_fdp": empirical_fdp,
        "recall": recall,
    }


def _run_once(X, y, true_signal, q: float, backend: str, seed: int):
    t0 = time.perf_counter()
    result = fixed_x_knockoff_filter(
        X,
        y,
        q=q,
        method="corr_diff",
        fdr_control="knockoff_plus",
        random_state=seed,
        backend=backend,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    summary = _summarize_selection(result.selected_features, true_signal)
    threshold = float(result.threshold)
    threshold_out = threshold if np.isfinite(threshold) else None
    return {
        "time_ms": float(elapsed_ms),
        "threshold": threshold_out,
        "estimated_fdr": float(result.estimated_fdr),
        "backend": result.backend,
        **summary,
    }


def run_benchmark():
    cfg = {
        "seed": 20260406,
        "n_samples": 400,
        "n_features": 80,
        "n_signal": 12,
        "noise_scale": 1.0,
        "q_grid": [0.05, 0.10, 0.20],
    }

    X, y, true_signal = _make_data(
        seed=cfg["seed"],
        n=cfg["n_samples"],
        p=cfg["n_features"],
        n_signal=cfg["n_signal"],
        noise_scale=cfg["noise_scale"],
    )

    out = {
        "date": str(date.today()),
        "environment": {
            "cuda_available": bool(_cuda_available()),
        },
        "config": cfg,
        "runs": {
            "numpy": {},
            "cupy": None,
        },
    }

    for q in cfg["q_grid"]:
        key = f"q={q:.2f}"
        out["runs"]["numpy"][key] = _run_once(X, y, true_signal, q=q, backend="numpy", seed=cfg["seed"])

    if out["environment"]["cuda_available"]:
        import cupy as cp

        X_cp = cp.asarray(X)
        y_cp = cp.asarray(y)
        out["runs"]["cupy"] = {}
        for q in cfg["q_grid"]:
            key = f"q={q:.2f}"
            out["runs"]["cupy"][key] = _run_once(
                X_cp,
                y_cp,
                true_signal,
                q=q,
                backend="cupy",
                seed=cfg["seed"],
            )

    output_path = Path("results") / f"benchmark_knockoff_fixedx_{date.today()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote benchmark result to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
