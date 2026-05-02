"""
Benchmark and validate statgpu unsupervised estimators.

Examples
--------
Local smoke:
    python dev/benchmarks/benchmark_unsupervised.py --n 1000 --p 20 --devices cpu

Remote GPU:
    python dev/benchmarks/benchmark_unsupervised.py \
      --n 50000 --p 1000 --k 10 --devices cpu,cuda,torch \
      --json-out results/unsupervised_remote.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.unsupervised import KMeans, PCA


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _time_call(fn, repeats: int = 1, warmup: int = 0):
    for _ in range(max(0, int(warmup))):
        fn()
        _synchronize_all()
    times = []
    out = None
    for _ in range(max(1, int(repeats))):
        _synchronize_all()
        t0 = time.perf_counter()
        out = fn()
        _synchronize_all()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, float(np.median(times)), times


def _synchronize_all():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _time_call_once(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        try:
            import cupy as cp

            cp.cuda.Device(0).use()
            return True
        except Exception:
            return False
    if device == "torch":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return False


def _as_device_input(X, device: str):
    if device == "cuda":
        import cupy as cp

        return cp.asarray(X, dtype=cp.float64)
    if device == "torch":
        import torch

        return torch.as_tensor(X, dtype=torch.float64, device="cuda")
    return X


def make_data(seed: int, n: int, p: int, k: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, p))
    labels = rng.integers(0, k, size=n)
    X = centers[labels] + rng.normal(scale=0.6, size=(n, p))
    return X.astype(np.float64, copy=False)


def _projector(components):
    return components.T @ components


def bench_pca(
    X,
    devices: List[str],
    n_components: int,
    solver: str,
    preload_device_data: bool = False,
    repeats: int = 1,
    warmup: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cpu_ref: Optional[PCA] = None

    for device in devices:
        if not _device_available(device):
            rows.append({"method": "PCA", "device": device, "status": "skipped"})
            continue

        X_fit = _as_device_input(X, device) if preload_device_data else X
        model, fit_ms, fit_ms_all = _time_call(
            lambda device=device: PCA(
                n_components=n_components,
                svd_solver=solver,
                device=device,
            ).fit(X_fit),
            repeats=repeats,
            warmup=warmup,
        )
        if device == "cpu":
            cpu_ref = model

        row: Dict[str, Any] = {
            "method": "PCA",
            "device": device,
            "status": "ok",
            "fit_ms": fit_ms,
            "fit_ms_all": fit_ms_all,
            "explained_variance_sum": float(np.sum(_to_numpy(model.explained_variance_))),
        }
        if cpu_ref is not None and device != "cpu":
            row["max_abs_explained_variance_diff_vs_cpu"] = float(
                np.max(
                    np.abs(
                        _to_numpy(model.explained_variance_)
                        - _to_numpy(cpu_ref.explained_variance_)
                    )
                )
            )
            row["max_abs_projector_diff_vs_cpu"] = float(
                np.max(
                    np.abs(
                        _projector(_to_numpy(model.components_))
                        - _projector(_to_numpy(cpu_ref.components_))
                    )
                )
            )
        rows.append(row)

    try:
        from sklearn.decomposition import PCA as SklearnPCA

        sk, fit_ms, fit_ms_all = _time_call(
            lambda: SklearnPCA(n_components=n_components, svd_solver="full").fit(X)
            ,
            repeats=repeats,
            warmup=warmup,
        )
        row = {
            "method": "PCA",
            "device": "sklearn",
            "status": "ok",
            "fit_ms": fit_ms,
            "fit_ms_all": fit_ms_all,
        }
        if cpu_ref is not None:
            row["max_abs_explained_variance_diff_vs_cpu"] = float(
                np.max(np.abs(sk.explained_variance_ - _to_numpy(cpu_ref.explained_variance_)))
            )
            row["max_abs_projector_diff_vs_cpu"] = float(
                np.max(np.abs(_projector(sk.components_) - _projector(_to_numpy(cpu_ref.components_))))
            )
        rows.append(row)
    except Exception as exc:
        rows.append({"method": "PCA", "device": "sklearn", "status": "skipped", "notes": str(exc)})

    return rows


def _match_centers(a, b):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return np.nan
    distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(distances)
    return float(np.max(np.sqrt(distances[rows, cols])))


def bench_kmeans(
    X,
    devices: List[str],
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    preload_device_data: bool = False,
    repeats: int = 1,
    warmup: int = 0,
):
    rows: List[Dict[str, Any]] = []
    cpu_ref: Optional[KMeans] = None

    for device in devices:
        if not _device_available(device):
            rows.append({"method": "KMeans", "device": device, "status": "skipped"})
            continue

        X_fit = _as_device_input(X, device) if preload_device_data else X
        model, fit_ms, fit_ms_all = _time_call(
            lambda device=device: KMeans(
                n_clusters=k,
                n_init=n_init,
                max_iter=max_iter,
                random_state=seed,
                device=device,
            ).fit(X_fit),
            repeats=repeats,
            warmup=warmup,
        )
        if device == "cpu":
            cpu_ref = model

        row: Dict[str, Any] = {
            "method": "KMeans",
            "device": device,
            "status": "ok",
            "fit_ms": fit_ms,
            "fit_ms_all": fit_ms_all,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        }
        if cpu_ref is not None and device != "cpu":
            row["abs_inertia_diff_vs_cpu"] = float(abs(model.inertia_ - cpu_ref.inertia_))
            row["max_center_distance_vs_cpu"] = _match_centers(
                _to_numpy(model.cluster_centers_),
                _to_numpy(cpu_ref.cluster_centers_),
            )
        rows.append(row)

    try:
        from sklearn.cluster import KMeans as SklearnKMeans

        if cpu_ref is not None:
            sk_init = _to_numpy(cpu_ref.cluster_centers_)
            sk_n_init = 1
        else:
            sk_init = "k-means++"
            sk_n_init = n_init
        sk, fit_ms, fit_ms_all = _time_call(
            lambda: SklearnKMeans(
                n_clusters=k,
                init=sk_init,
                n_init=sk_n_init,
                max_iter=max_iter,
                random_state=seed,
                algorithm="lloyd",
            ).fit(X),
            repeats=repeats,
            warmup=warmup,
        )
        row = {
            "method": "KMeans",
            "device": "sklearn",
            "status": "ok",
            "fit_ms": fit_ms,
            "fit_ms_all": fit_ms_all,
            "inertia": float(sk.inertia_),
            "n_iter": int(sk.n_iter_),
        }
        if cpu_ref is not None:
            row["abs_inertia_diff_vs_cpu"] = float(abs(sk.inertia_ - cpu_ref.inertia_))
            row["max_center_distance_vs_cpu"] = _match_centers(
                sk.cluster_centers_,
                _to_numpy(cpu_ref.cluster_centers_),
            )
        rows.append(row)
    except Exception as exc:
        rows.append({"method": "KMeans", "device": "sklearn", "status": "skipped", "notes": str(exc)})

    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark statgpu PCA and KMeans.")
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--p", type=int, default=32)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--pca-components", type=int, default=8)
    p.add_argument("--pca-solver", choices=["auto", "full", "covariance", "randomized"], default="auto")
    p.add_argument("--kmeans-n-init", type=int, default=2)
    p.add_argument("--kmeans-max-iter", type=int, default=100)
    p.add_argument("--devices", type=str, default="cpu,cuda,torch")
    p.add_argument("--methods", type=str, default="pca,kmeans", help="Comma-separated subset: pca,kmeans")
    p.add_argument("--preload-device-data", action="store_true", help="Move CUDA/Torch inputs to GPU before timing")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup-runs", type=int, default=0)
    p.add_argument("--json-out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    X = make_data(args.seed, args.n, args.p, args.k)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}

    rows = []
    if "pca" in methods:
        rows.extend(
            bench_pca(
                X,
                devices,
                args.pca_components,
                args.pca_solver,
                args.preload_device_data,
                args.repeats,
                args.warmup_runs,
            )
        )
    if "kmeans" in methods:
        rows.extend(
            bench_kmeans(
                X,
                devices,
                args.k,
                args.kmeans_n_init,
                args.kmeans_max_iter,
                args.seed,
                args.preload_device_data,
                args.repeats,
                args.warmup_runs,
            )
        )

    result = {
        "seed": args.seed,
        "n": args.n,
        "p": args.p,
        "k": args.k,
        "devices": devices,
        "methods": sorted(methods),
        "preload_device_data": bool(args.preload_device_data),
        "repeats": int(args.repeats),
        "warmup_runs": int(args.warmup_runs),
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
