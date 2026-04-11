"""Structured benchmark: fixed-X/model-X knockoff vs baseline selectors."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from statgpu.feature_selection import knockoff_filter
from statgpu.linear_model import Lasso


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _knockpy_available() -> bool:
    try:
        import knockpy  # noqa: F401

        return True
    except Exception:
        return False


def _make_correlated_data(
    *,
    seed: int,
    n_samples: int,
    n_features: int,
    n_signal: int,
    noise_scale: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n_samples, n_features))
    common = rng.normal(size=(n_samples, 1))
    X = np.sqrt(max(0.0, 1.0 - rho)) * Z + np.sqrt(max(0.0, rho)) * common

    beta = np.zeros(n_features)
    signs = rng.choice([-1.0, 1.0], size=n_signal)
    beta[:n_signal] = signs * rng.uniform(0.9, 2.2, size=n_signal)
    y = X @ beta + rng.normal(scale=noise_scale, size=n_samples)

    true_signal = np.where(beta != 0.0)[0]
    return X, y, true_signal


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    k_eff = min(int(k), int(scores.shape[0]))
    if k_eff == 0:
        return np.asarray([], dtype=np.int64)

    idx = np.argsort(np.asarray(scores))[-k_eff:]
    return np.sort(idx.astype(np.int64))


def _selection_metrics(selected, true_signal) -> Dict[str, float]:
    sel = set(int(i) for i in np.asarray(selected).reshape(-1).tolist())
    sig = set(int(i) for i in np.asarray(true_signal).reshape(-1).tolist())

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


def _run_knockoff(
    X,
    y,
    true_signal,
    *,
    knockoff_type: str,
    seed: int,
    q: float,
    backend: str,
    method: str,
    compat_mode: str,
    lasso_cv_impl: str,
):
    t0 = time.perf_counter()
    result = knockoff_filter(
        X,
        y,
        knockoff_type=knockoff_type,
        q=q,
        method=method,
        fdr_control="knockoff_plus",
        random_state=seed,
        backend=backend,
        compat_mode=compat_mode,
        lasso_cv_impl=lasso_cv_impl,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metrics = _selection_metrics(result.selected_features, true_signal)
    threshold = float(result.threshold)
    threshold_out = threshold if np.isfinite(threshold) else None

    out = {
        "time_ms": float(elapsed_ms),
        "knockoff_type": result.knockoff_type,
        "backend": result.backend,
        "estimated_fdr": float(result.estimated_fdr),
        "threshold": threshold_out,
        "compat_mode": str(result.metadata.get("compat_mode", compat_mode)),
        "lasso_cv_impl": str(result.metadata.get("lasso_cv_impl", lasso_cv_impl)),
        **metrics,
    }
    if result.knockoff_type == "model_x":
        out["modelx_n_draws"] = int(result.metadata.get("n_modelx_draws", 0))
        out["modelx_covariance_shrinkage"] = float(result.metadata.get("covariance_shrinkage", 0.0))
    return out


def _run_marginal_corr_topk(X, y, true_signal, *, k_topk: int):
    t0 = time.perf_counter()
    y_centered = y - np.mean(y)
    scores = np.abs(X.T @ y_centered)
    selected = _topk_indices(scores, k=k_topk)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "time_ms": float(elapsed_ms),
        **_selection_metrics(selected, true_signal),
    }


def _run_statgpu_lasso_topk(X, y, true_signal, *, alpha: float, k_topk: int):
    t0 = time.perf_counter()
    model = Lasso(
        alpha=float(alpha),
        device="cpu",
        compute_inference=False,
        max_iter=3000,
        tol=1e-4,
        cpu_solver="coordinate_descent",
    )
    model.fit(X, y)
    scores = np.abs(np.asarray(model.coef_))
    selected = _topk_indices(scores, k=k_topk)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "time_ms": float(elapsed_ms),
        "alpha": float(alpha),
        **_selection_metrics(selected, true_signal),
    }


def _run_sklearn_lasso_cv(X, y, true_signal):
    from sklearn.linear_model import LassoCV

    t0 = time.perf_counter()
    model = LassoCV(cv=5, random_state=0, max_iter=5000)
    model.fit(X, y)
    selected = np.where(np.abs(model.coef_) > 1e-10)[0].astype(np.int64)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "time_ms": float(elapsed_ms),
        "alpha": float(model.alpha_),
        **_selection_metrics(selected, true_signal),
    }


def _run_knockpy_gaussian_lasso(X, y, true_signal, *, q: float, seed: int):
    try:
        import knockpy
        from knockpy.knockoff_filter import KnockoffFilter

        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float).reshape(-1)

        np.random.seed(int(seed))
        t0 = time.perf_counter()
        kfilter = KnockoffFilter(ksampler="gaussian", fstat="lasso")
        rejections = kfilter.forward(
            X=X_np,
            y=y_np,
            fdr=float(q),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rej_arr = np.asarray(rejections).reshape(-1)
        if rej_arr.shape[0] == X_np.shape[1]:
            selected = np.where(rej_arr > 0)[0].astype(np.int64)
        else:
            selected = np.asarray([], dtype=np.int64)

        threshold = getattr(kfilter, "threshold", None)
        threshold_out = float(threshold) if threshold is not None else None

        return {
            "time_ms": float(elapsed_ms),
            "knockoff_type": "model_x",
            "backend": "numpy",
            "estimated_fdr": None,
            "threshold": threshold_out,
            "framework": "knockpy",
            "knockpy_version": str(getattr(knockpy, "__version__", "unknown")),
            "knockpy_ksampler": "gaussian",
            "knockpy_fstat": "lasso",
            **_selection_metrics(selected, true_signal),
        }
    except Exception as exc:
        return {
            "time_ms": None,
            "knockoff_type": "model_x",
            "backend": "numpy",
            "estimated_fdr": None,
            "threshold": None,
            "framework": "knockpy",
            "error": str(exc),
            **_selection_metrics(np.asarray([], dtype=np.int64), true_signal),
        }


def _aggregate_runs(runs: List[Dict]) -> Dict[str, float]:
    if len(runs) == 0:
        return {"n_runs": 0}

    keys = [
        "time_ms",
        "n_selected",
        "precision",
        "recall",
        "fdp",
        "f1",
        "jaccard_truth",
        "estimated_fdr",
    ]

    out = {"n_runs": int(len(runs))}
    for k in keys:
        vals = [float(r[k]) for r in runs if k in r and r[k] is not None]
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_std"] = float(np.std(arr, ddof=0))

    threshold_nonnull = [1.0 for r in runs if r.get("threshold", None) is not None]
    out["threshold_nonnull_rate"] = float(len(threshold_nonnull) / len(runs))
    return out


def run_benchmark():
    cfg = {
        "seeds": [20260406, 20260407, 20260408],
        "n_samples": 400,
        "n_features": 80,
        "n_signal": 12,
        "noise_scale": 1.0,
        "rho": 0.30,
        "q": 0.10,
        "knockoff_method": os.environ.get("STATGPU_KNOCKOFF_METHOD", "ols_coef_diff"),
        "knockoff_compat_mode": os.environ.get("STATGPU_KNOCKOFF_COMPAT_MODE", "statgpu"),
        "knockoff_lasso_cv_impl": os.environ.get("STATGPU_KNOCKOFF_LASSO_CV_IMPL", "auto"),
        "k_topk": 12,
        "lasso_alpha": 0.08,
    }

    cuda_ok = _cuda_available()
    skl_ok = _sklearn_available()
    knockpy_ok = _knockpy_available()

    out = {
        "date": str(date.today()),
        "environment": {
            "cuda_available": bool(cuda_ok),
            "sklearn_available": bool(skl_ok),
            "knockpy_available": bool(knockpy_ok),
        },
        "config": cfg,
        "methods": {
            "knockoff_fixedx_numpy": {"runs": []},
            "knockoff_modelx_numpy": {"runs": []},
            "marginal_corr_topk": {"runs": []},
            "statgpu_lasso_topk": {"runs": []},
            "sklearn_lasso_cv": {"runs": []} if skl_ok else None,
            "knockpy_gaussian_lasso": {"runs": []} if knockpy_ok else None,
            "knockoff_fixedx_cupy": {"runs": []} if cuda_ok else None,
            "knockoff_modelx_cupy": {"runs": []} if cuda_ok else None,
        },
        "pairwise": {},
    }

    for seed in cfg["seeds"]:
        X, y, true_signal = _make_correlated_data(
            seed=int(seed),
            n_samples=int(cfg["n_samples"]),
            n_features=int(cfg["n_features"]),
            n_signal=int(cfg["n_signal"]),
            noise_scale=float(cfg["noise_scale"]),
            rho=float(cfg["rho"]),
        )

        out["methods"]["knockoff_fixedx_numpy"]["runs"].append(
            {
                "seed": int(seed),
                **_run_knockoff(
                    X,
                    y,
                    true_signal,
                    knockoff_type="fixed_x",
                    seed=int(seed),
                    q=float(cfg["q"]),
                    backend="numpy",
                    method=str(cfg["knockoff_method"]),
                    compat_mode=str(cfg["knockoff_compat_mode"]),
                    lasso_cv_impl=str(cfg["knockoff_lasso_cv_impl"]),
                ),
            }
        )
        out["methods"]["knockoff_modelx_numpy"]["runs"].append(
            {
                "seed": int(seed),
                **_run_knockoff(
                    X,
                    y,
                    true_signal,
                    knockoff_type="model_x",
                    seed=int(seed),
                    q=float(cfg["q"]),
                    backend="numpy",
                    method=str(cfg["knockoff_method"]),
                    compat_mode=str(cfg["knockoff_compat_mode"]),
                    lasso_cv_impl=str(cfg["knockoff_lasso_cv_impl"]),
                ),
            }
        )
        out["methods"]["marginal_corr_topk"]["runs"].append(
            {"seed": int(seed), **_run_marginal_corr_topk(X, y, true_signal, k_topk=int(cfg["k_topk"]))}
        )
        out["methods"]["statgpu_lasso_topk"]["runs"].append(
            {
                "seed": int(seed),
                **_run_statgpu_lasso_topk(
                    X,
                    y,
                    true_signal,
                    alpha=float(cfg["lasso_alpha"]),
                    k_topk=int(cfg["k_topk"]),
                ),
            }
        )

        if skl_ok and out["methods"]["sklearn_lasso_cv"] is not None:
            out["methods"]["sklearn_lasso_cv"]["runs"].append(
                {"seed": int(seed), **_run_sklearn_lasso_cv(X, y, true_signal)}
            )

        if knockpy_ok and out["methods"]["knockpy_gaussian_lasso"] is not None:
            out["methods"]["knockpy_gaussian_lasso"]["runs"].append(
                {
                    "seed": int(seed),
                    **_run_knockpy_gaussian_lasso(
                        X,
                        y,
                        true_signal,
                        q=float(cfg["q"]),
                        seed=int(seed),
                    ),
                }
            )

        if cuda_ok and out["methods"]["knockoff_fixedx_cupy"] is not None:
            import cupy as cp

            X_cp = cp.asarray(X)
            y_cp = cp.asarray(y)
            out["methods"]["knockoff_fixedx_cupy"]["runs"].append(
                {
                    "seed": int(seed),
                    **_run_knockoff(
                        X_cp,
                        y_cp,
                        true_signal,
                        knockoff_type="fixed_x",
                        seed=int(seed),
                        q=float(cfg["q"]),
                        backend="cupy",
                        method=str(cfg["knockoff_method"]),
                        compat_mode=str(cfg["knockoff_compat_mode"]),
                        lasso_cv_impl=str(cfg["knockoff_lasso_cv_impl"]),
                    ),
                }
            )
            out["methods"]["knockoff_modelx_cupy"]["runs"].append(
                {
                    "seed": int(seed),
                    **_run_knockoff(
                        X_cp,
                        y_cp,
                        true_signal,
                        knockoff_type="model_x",
                        seed=int(seed),
                        q=float(cfg["q"]),
                        backend="cupy",
                        method=str(cfg["knockoff_method"]),
                        compat_mode=str(cfg["knockoff_compat_mode"]),
                        lasso_cv_impl=str(cfg["knockoff_lasso_cv_impl"]),
                    ),
                }
            )

    for method_name, payload in out["methods"].items():
        if payload is None:
            continue
        payload["aggregate"] = _aggregate_runs(payload["runs"])

    numpy_runs = out["methods"]["knockoff_fixedx_numpy"]["runs"]
    cupy_payload = out["methods"]["knockoff_fixedx_cupy"]
    if cupy_payload is not None and len(cupy_payload["runs"]) == len(numpy_runs):
        diffs = []
        for r_np, r_cp in zip(numpy_runs, cupy_payload["runs"]):
            diffs.append(abs(float(r_np["estimated_fdr"]) - float(r_cp["estimated_fdr"])))
        out["pairwise"]["knockoff_numpy_vs_cupy"] = {
            "estimated_fdr_abs_diff_mean": float(np.mean(diffs)) if len(diffs) else 0.0,
            "estimated_fdr_abs_diff_max": float(np.max(diffs)) if len(diffs) else 0.0,
        }

    numpy_runs_modelx = out["methods"]["knockoff_modelx_numpy"]["runs"]
    cupy_payload_modelx = out["methods"]["knockoff_modelx_cupy"]
    if cupy_payload_modelx is not None and len(cupy_payload_modelx["runs"]) == len(numpy_runs_modelx):
        diffs = []
        for r_np, r_cp in zip(numpy_runs_modelx, cupy_payload_modelx["runs"]):
            diffs.append(abs(float(r_np["estimated_fdr"]) - float(r_cp["estimated_fdr"])))
        out["pairwise"]["knockoff_modelx_numpy_vs_cupy"] = {
            "estimated_fdr_abs_diff_mean": float(np.mean(diffs)) if len(diffs) else 0.0,
            "estimated_fdr_abs_diff_max": float(np.max(diffs)) if len(diffs) else 0.0,
        }

    knockpy_payload = out["methods"]["knockpy_gaussian_lasso"]
    if knockpy_payload is not None and len(knockpy_payload["runs"]) == len(numpy_runs_modelx):
        precision_diffs = []
        recall_diffs = []
        fdp_diffs = []
        time_ratios = []
        for r_sg, r_kp in zip(numpy_runs_modelx, knockpy_payload["runs"]):
            if "precision" in r_kp:
                precision_diffs.append(float(r_sg["precision"]) - float(r_kp["precision"]))
                recall_diffs.append(float(r_sg["recall"]) - float(r_kp["recall"]))
                fdp_diffs.append(float(r_sg["fdp"]) - float(r_kp["fdp"]))
            t_sg = r_sg.get("time_ms", None)
            t_kp = r_kp.get("time_ms", None)
            if t_sg is not None and t_kp is not None and float(t_sg) > 0.0:
                time_ratios.append(float(t_kp) / float(t_sg))

        out["pairwise"]["knockoff_modelx_numpy_vs_knockpy"] = {
            "precision_diff_mean": float(np.mean(precision_diffs)) if len(precision_diffs) else None,
            "recall_diff_mean": float(np.mean(recall_diffs)) if len(recall_diffs) else None,
            "fdp_diff_mean": float(np.mean(fdp_diffs)) if len(fdp_diffs) else None,
            "time_ratio_knockpy_over_statgpu_mean": float(np.mean(time_ratios)) if len(time_ratios) else None,
        }

    output_path = Path("results") / f"benchmark_knockoff_vs_baselines_{date.today()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote benchmark result to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
