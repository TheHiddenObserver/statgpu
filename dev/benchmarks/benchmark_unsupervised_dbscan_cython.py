"""DBSCAN CPU Cython fast-path validation benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import statgpu.unsupervised._dbscan as dbscan_mod
from statgpu.unsupervised import DBSCAN


def make_blobs(seed: int, n: int, p: int, k: int, scale: float = 0.25):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 6.0, size=(k, p))
    labels = rng.integers(0, k, size=n)
    return (centers[labels] + rng.normal(0.0, scale, size=(n, p))).astype(np.float64)


def make_variable(seed: int, n: int, p: int, k: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 8.0, size=(k, p))
    labels = rng.integers(0, k, size=n)
    scales = np.linspace(0.2, 1.0, k)
    X = np.empty((n, p), dtype=np.float64)
    for i in range(n):
        X[i] = centers[labels[i]] + rng.normal(0.0, scales[labels[i]], size=p)
    return X


def sync():
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def available(device: str):
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


def as_device(X, device: str):
    if device == "cuda":
        import cupy as cp

        return cp.asarray(X, dtype=cp.float64)
    if device == "torch":
        import torch

        return torch.as_tensor(X, dtype=torch.float64, device="cuda")
    return X


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def time_rep(fn, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
        sync()
    vals = []
    out = None
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return out, float(np.median(vals)), vals


def fit_cpu(X, eps, min_samples, use_cython: bool):
    old = dbscan_mod.dbscan_dense_pairwise
    if not use_cython:
        dbscan_mod.dbscan_dense_pairwise = None
    try:
        return DBSCAN(eps=eps, min_samples=min_samples, device="cpu").fit(X)
    finally:
        dbscan_mod.dbscan_dense_pairwise = old


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", default="")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--devices", default="cuda,torch")
    args = p.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    cases = [
        ("compact_medium", make_blobs(args.seed + 1, 1200, 8, 4), 1.0, 5),
        ("compact_2500", make_blobs(args.seed + 2, 2500, 8, 5), 1.0, 5),
        ("compact_large", make_blobs(args.seed + 3, 5000, 8, 5), 1.0, 5),
        ("variable", make_variable(args.seed + 4, 5000, 8, 6), 1.0, 5),
        ("all_noise", make_blobs(args.seed + 5, 5000, 8, 5), 0.02, 5),
    ]

    rows = []
    extension_loaded = dbscan_mod.dbscan_dense_pairwise is not None
    for case, X, eps, min_samples in cases:
        labels_ref = None
        for framework, use_cython in (("statgpu_cpu_cython", True), ("statgpu_cpu_fallback", False)):
            row = {
                "method": "DBSCAN",
                "case": case,
                "framework": framework,
                "backend": "cpu",
                "n": int(X.shape[0]),
                "p": int(X.shape[1]),
                "eps": eps,
                "min_samples": int(min_samples),
                "extension_loaded": bool(extension_loaded),
            }
            if use_cython and not extension_loaded:
                row.update({"status": "skipped", "notes": "compiled extension unavailable"})
                rows.append(row)
                continue
            model, ms, all_ms = time_rep(
                lambda use_cython=use_cython: fit_cpu(X, eps, min_samples, use_cython),
                args.repeats,
                args.warmup_runs,
            )
            labels = to_numpy(model.labels_)
            row.update(
                {
                    "status": "ok",
                    "fit_ms": ms,
                    "fit_ms_all": all_ms,
                    "n_clusters": int(len(set(labels) - {-1})),
                    "n_noise": int(np.sum(labels == -1)),
                    "n_core": int(len(to_numpy(model.core_sample_indices_))),
                }
            )
            if labels_ref is None:
                labels_ref = labels
            else:
                row["ari_vs_cython"] = float(adjusted_rand_score(labels_ref, labels))
                row["noise_mask_match_vs_cython"] = bool(np.array_equal(labels_ref == -1, labels == -1))
            rows.append(row)

        sk_model, ms, all_ms = time_rep(lambda: SkDBSCAN(eps=eps, min_samples=min_samples).fit(X), args.repeats, args.warmup_runs)
        sk_labels = sk_model.labels_
        sk_row = {
            "method": "DBSCAN",
            "case": case,
            "framework": "sklearn",
            "backend": "cpu",
            "status": "ok",
            "fit_ms": ms,
            "fit_ms_all": all_ms,
            "n": int(X.shape[0]),
            "p": int(X.shape[1]),
            "n_clusters": int(len(set(sk_labels) - {-1})),
            "n_noise": int(np.sum(sk_labels == -1)),
            "n_core": int(len(sk_model.core_sample_indices_)),
        }
        if labels_ref is not None:
            sk_row["ari_vs_cython"] = float(adjusted_rand_score(labels_ref, sk_labels))
            sk_row["noise_mask_match_vs_cython"] = bool(np.array_equal(labels_ref == -1, sk_labels == -1))
        rows.append(sk_row)

        for device in devices:
            row = {"method": "DBSCAN", "case": case, "framework": "statgpu", "backend": device, "n": int(X.shape[0]), "p": int(X.shape[1])}
            if not available(device):
                row.update({"status": "skipped"})
                rows.append(row)
                continue
            Xd = as_device(X, device)
            model, ms, all_ms = time_rep(
                lambda device=device, Xd=Xd: DBSCAN(eps=eps, min_samples=min_samples, device=device, batch_size=512).fit(Xd),
                args.repeats,
                args.warmup_runs,
            )
            labels = to_numpy(model.labels_)
            row.update(
                {
                    "status": "ok",
                    "fit_ms": ms,
                    "fit_ms_all": all_ms,
                    "n_clusters": int(len(set(labels) - {-1})),
                    "n_noise": int(np.sum(labels == -1)),
                    "n_core": int(len(to_numpy(model.core_sample_indices_))),
                    "ari_vs_cython": float(adjusted_rand_score(labels_ref, labels)) if labels_ref is not None else None,
                    "noise_mask_match_vs_cython": bool(np.array_equal(labels_ref == -1, labels == -1)) if labels_ref is not None else None,
                }
            )
            rows.append(row)

    result = {"seed": args.seed, "extension_loaded": bool(extension_loaded), "rows": rows}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
