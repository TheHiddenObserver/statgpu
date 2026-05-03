"""Remote validation matrix for Phase 2 unsupervised estimators."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.unsupervised import AgglomerativeClustering, DBSCAN, GaussianMixture, NMF


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _sync():
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


def _time_call(fn, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
        _sync()
    vals = []
    out = None
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return out, float(np.median(vals)), vals


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


def make_blobs(seed: int, n: int, p: int, k: int, scale: float = 0.5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, p))
    labels = rng.integers(0, k, size=n)
    return (centers[labels] + rng.normal(scale=scale, size=(n, p))).astype(np.float64)


def make_nmf(seed: int, n: int, p: int, k: int):
    rng = np.random.default_rng(seed)
    W = rng.random((n, k))
    H = rng.random((k, p))
    return (W @ H + 0.01 * rng.random((n, p))).astype(np.float64)


def _ari(a, b):
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(a, b))
    except Exception:
        return None


def _scalar(x) -> float:
    return float(np.asarray(_to_numpy(x)).item())


def _make_gmm_initial_state(X: np.ndarray, n_components: int, seed: int, reg_covar: float):
    rng = np.random.default_rng(seed + 13007)
    indices = rng.choice(X.shape[0], size=int(n_components), replace=False)
    means = np.ascontiguousarray(X[indices], dtype=np.float64)
    weights = np.full((int(n_components),), 1.0 / float(n_components), dtype=np.float64)
    global_var = np.mean((X - np.mean(X, axis=0)) ** 2, axis=0) + float(reg_covar)
    covariances = np.tile(global_var, (int(n_components), 1)).astype(np.float64, copy=False)
    return weights, means, covariances


def _fit_statgpu_gmm_fixed_init(X, initial_state, device: str, max_iter: int, tol: float, reg_covar: float) -> GaussianMixture:
    weights0, means0, covariances0 = initial_state
    model = GaussianMixture(
        n_components=int(means0.shape[0]),
        covariance_type="diag",
        tol=tol,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=1,
        init_params="random",
        random_state=0,
        device=device,
    )
    backend = model._get_backend()
    X_arr = backend.asarray(X, dtype=backend.float64)
    weights = backend.asarray(weights0, dtype=backend.float64)
    means = backend.asarray(means0, dtype=backend.float64)
    covariances = backend.asarray(covariances0, dtype=backend.float64)
    lower_bound = -np.inf
    converged = False
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        prev_lower_bound = lower_bound
        lower_bound, resp = model._e_step(backend, X_arr, weights, means, covariances)
        weights, means, covariances = model._m_step(backend, X_arr, resp)
        if abs(lower_bound - prev_lower_bound) < float(tol):
            converged = True
            break
    model.weights_ = weights
    model.means_ = means
    model.covariances_ = covariances
    model.precisions_cholesky_ = 1.0 / backend.sqrt(covariances)
    model.converged_ = bool(converged)
    model.n_iter_ = int(n_iter)
    model.lower_bound_ = float(lower_bound)
    model.n_features_in_ = int(X_arr.shape[1])
    model._backend_name = backend.name
    model._fitted = True
    return model


def _make_nmf_initial_state(X: np.ndarray, n_components: int, seed: int):
    rng = np.random.RandomState(seed)
    mean = max(float(np.mean(X)), np.finfo(np.float64).eps)
    scale = np.sqrt(mean / float(n_components))
    H = np.abs(rng.standard_normal((int(n_components), X.shape[1]))) * scale + 1e-8
    W = np.abs(rng.standard_normal((X.shape[0], int(n_components)))) * scale + 1e-8
    return np.ascontiguousarray(W, dtype=np.float64), np.ascontiguousarray(H, dtype=np.float64)


def _fit_statgpu_nmf_fixed_init(X, initial_state, device: str, max_iter: int, tol: float) -> NMF:
    W0, H0 = initial_state
    model = NMF(n_components=int(H0.shape[0]), max_iter=max_iter, tol=tol, random_state=0, device=device)
    backend = model._get_backend()
    X_arr = backend.asarray(X, dtype=backend.float64)
    W = backend.asarray(W0, dtype=backend.float64)
    H = backend.asarray(H0, dtype=backend.float64)
    eps = np.finfo(np.float64).eps
    previous_error = None
    error = None
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        W = model._update_w(backend, X_arr, W, H, eps)
        H = model._update_h(backend, X_arr, W, H, eps)
        if n_iter % 10 == 0 or n_iter == int(max_iter):
            error = model._reconstruction_error(backend, X_arr, W, H)
            if previous_error is not None and abs(previous_error - error) / max(previous_error, eps) <= float(tol):
                break
            previous_error = error
    if error is None:
        error = model._reconstruction_error(backend, X_arr, W, H)
    model.components_ = H
    model._fit_W = W
    model.reconstruction_err_ = float(error)
    model.n_iter_ = int(n_iter)
    model.n_components_ = int(H0.shape[0])
    model.n_features_in_ = int(X_arr.shape[1])
    model._backend_name = backend.name
    model._fitted = True
    return model


def bench_dbscan(X, devices, repeats, warmup):
    rows = []
    refs = {}
    for device in devices:
        if not _device_available(device):
            rows.append({"method": "DBSCAN", "framework": "statgpu", "backend": device, "status": "skipped"})
            continue
        X_fit = _as_device_input(X, device)
        model, ms, all_ms = _time_call(lambda device=device: DBSCAN(eps=1.2, min_samples=5, device=device).fit(X_fit), repeats, warmup)
        labels = _to_numpy(model.labels_)
        row = {"method": "DBSCAN", "framework": "statgpu", "backend": device, "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "n_clusters": int(len(set(labels) - {-1})), "n_noise": int(np.sum(labels == -1)), "n_core": int(len(_to_numpy(model.core_sample_indices_)))}
        if device == "cpu":
            refs["cpu"] = labels
        elif "cpu" in refs:
            row["ari_vs_cpu"] = _ari(refs["cpu"], labels)
            row["noise_mask_match_vs_cpu"] = bool(np.array_equal(refs["cpu"] == -1, labels == -1))
        rows.append(row)
    try:
        from sklearn.cluster import DBSCAN as SkDBSCAN

        model, ms, all_ms = _time_call(lambda: SkDBSCAN(eps=1.2, min_samples=5).fit(X), repeats, warmup)
        row = {"method": "DBSCAN", "framework": "sklearn", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "n_clusters": int(len(set(model.labels_) - {-1})), "n_noise": int(np.sum(model.labels_ == -1)), "n_core": int(len(model.core_sample_indices_))}
        if "cpu" in refs:
            row["ari_vs_statgpu_cpu"] = _ari(refs["cpu"], model.labels_)
            row["noise_mask_match_vs_statgpu_cpu"] = bool(np.array_equal(refs["cpu"] == -1, model.labels_ == -1))
        rows.append(row)
    except Exception as exc:
        rows.append({"method": "DBSCAN", "framework": "sklearn", "status": "skipped", "notes": repr(exc)})
    return rows


def bench_gmm(X, devices, repeats, warmup, seed):
    rows = []
    cpu_score = None
    max_iter = 60
    tol = 1e-5
    reg_covar = 1e-6
    initial_state = _make_gmm_initial_state(X, n_components=4, seed=seed, reg_covar=reg_covar)
    for device in devices:
        if not _device_available(device):
            rows.append({"method": "GaussianMixture", "framework": "statgpu", "backend": device, "status": "skipped"})
            continue
        X_fit = _as_device_input(X, device)
        model, ms, all_ms = _time_call(
            lambda device=device: _fit_statgpu_gmm_fixed_init(
                X_fit,
                initial_state,
                device=device,
                max_iter=max_iter,
                tol=tol,
                reg_covar=reg_covar,
            ),
            repeats,
            warmup,
        )
        score = float(model.score(X_fit))
        row = {"method": "GaussianMixture", "framework": "statgpu", "backend": device, "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "score": score, "lower_bound": float(model.lower_bound_), "n_iter": int(model.n_iter_), "converged": bool(model.converged_), "init": "fixed_weights_means_diag_covariances"}
        if device == "cpu":
            cpu_score = score
        elif cpu_score is not None:
            row["abs_score_diff_vs_cpu"] = float(abs(score - cpu_score))
        rows.append(row)
    try:
        from sklearn.mixture import GaussianMixture as SkGMM

        weights0, means0, covariances0 = initial_state
        model, ms, all_ms = _time_call(
            lambda: SkGMM(
                n_components=4,
                covariance_type="diag",
                tol=tol,
                reg_covar=reg_covar,
                max_iter=max_iter,
                n_init=1,
                init_params="random",
                random_state=seed,
                weights_init=weights0,
                means_init=means0,
                precisions_init=1.0 / covariances0,
            ).fit(X),
            repeats,
            warmup,
        )
        score = float(model.score(X))
        row = {"method": "GaussianMixture", "framework": "sklearn", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "score": score, "n_iter": int(model.n_iter_), "converged": bool(model.converged_), "init": "fixed_weights_means_diag_covariances"}
        if cpu_score is not None:
            row["abs_score_diff_vs_statgpu_cpu"] = float(abs(score - cpu_score))
        rows.append(row)
    except Exception as exc:
        rows.append({"method": "GaussianMixture", "framework": "sklearn", "status": "skipped", "notes": repr(exc)})
    return rows


def bench_nmf(X, devices, repeats, warmup, seed):
    rows = []
    cpu_err = None
    max_iter = 120
    tol = 1e-4
    initial_state = _make_nmf_initial_state(X, n_components=8, seed=seed)
    for device in devices:
        if not _device_available(device):
            rows.append({"method": "NMF", "framework": "statgpu", "backend": device, "status": "skipped"})
            continue
        X_fit = _as_device_input(X, device)
        model, ms, all_ms = _time_call(
            lambda device=device: _fit_statgpu_nmf_fixed_init(
                X_fit,
                initial_state,
                device=device,
                max_iter=max_iter,
                tol=tol,
            ),
            repeats,
            warmup,
        )
        row = {"method": "NMF", "framework": "statgpu", "backend": device, "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "reconstruction_err": float(model.reconstruction_err_), "n_iter": int(model.n_iter_), "init": "fixed_W_H"}
        if device == "cpu":
            cpu_err = float(model.reconstruction_err_)
        elif cpu_err is not None:
            row["abs_reconstruction_err_diff_vs_cpu"] = float(abs(model.reconstruction_err_ - cpu_err))
        rows.append(row)
    try:
        from sklearn.decomposition import NMF as SkNMF

        W0, H0 = initial_state
        model, ms, all_ms = _time_call(
            lambda: SkNMF(
                n_components=8,
                init="custom",
                solver="mu",
                beta_loss="frobenius",
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
            ).fit_transform(X, W=W0.copy(), H=H0.copy()),
            repeats,
            warmup,
        )
        sk_model = SkNMF(
            n_components=8,
            init="custom",
            solver="mu",
            beta_loss="frobenius",
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
        )
        W = sk_model.fit_transform(X, W=W0.copy(), H=H0.copy())
        row = {"method": "NMF", "framework": "sklearn", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "reconstruction_err": float(sk_model.reconstruction_err_), "n_iter": int(sk_model.n_iter_), "init": "fixed_W_H"}
        if cpu_err is not None:
            row["rel_reconstruction_err_vs_statgpu_cpu"] = float(sk_model.reconstruction_err_ / cpu_err)
        rows.append(row)
    except Exception as exc:
        rows.append({"method": "NMF", "framework": "sklearn", "status": "skipped", "notes": repr(exc)})
    return rows


def bench_agglomerative(X, repeats, warmup):
    rows = []
    ref_labels = None
    model, ms, all_ms = _time_call(lambda: AgglomerativeClustering(n_clusters=4, device="cpu").fit(X), repeats, warmup)
    ref_labels = model.labels_
    rows.append({"method": "AgglomerativeClustering", "framework": "statgpu", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "n_clusters": int(len(set(model.labels_))), "n_merges": int(model.children_.shape[0])})
    try:
        from sklearn.cluster import AgglomerativeClustering as SkAgg

        model, ms, all_ms = _time_call(lambda: SkAgg(n_clusters=4, linkage="single", metric="euclidean").fit(X), repeats, warmup)
        rows.append({"method": "AgglomerativeClustering", "framework": "sklearn", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "ari_vs_statgpu_cpu": _ari(ref_labels, model.labels_)})
    except Exception as exc:
        rows.append({"method": "AgglomerativeClustering", "framework": "sklearn", "status": "skipped", "notes": repr(exc)})
    try:
        from scipy.cluster.hierarchy import fcluster, linkage

        def fit_scipy():
            Z = linkage(X, method="single", metric="euclidean")
            return fcluster(Z, t=4, criterion="maxclust") - 1

        labels, ms, all_ms = _time_call(fit_scipy, repeats, warmup)
        rows.append({"method": "AgglomerativeClustering", "framework": "scipy", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "ari_vs_statgpu_cpu": _ari(ref_labels, labels)})
    except Exception as exc:
        rows.append({"method": "AgglomerativeClustering", "framework": "scipy", "status": "skipped", "notes": repr(exc)})
    rows.extend(bench_r_agnes(X, ref_labels, repeats))
    return rows


def bench_r_agnes(X, ref_labels, repeats):
    if shutil.which("Rscript") is None:
        return [{"method": "AgglomerativeClustering", "framework": "R cluster", "status": "skipped", "notes": "Rscript not found"}]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "X.bin"
        script = Path(td) / "agnes.R"
        np.ascontiguousarray(X, dtype=np.float64).tofile(path)
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
path <- args[[1]]
n <- as.integer(args[[2]])
p <- as.integer(args[[3]])
repeats <- as.integer(args[[4]])
if (!requireNamespace('cluster', quietly=TRUE)) {
  cat('{"status":"skipped","notes":"cluster package not available"}')
  quit(status=0)
}
x <- readBin(path, what='numeric', n=n*p, size=8, endian='little')
X <- matrix(x, nrow=n, ncol=p, byrow=TRUE)
times <- numeric(repeats)
for (i in seq_len(repeats)) {
  t0 <- proc.time()[['elapsed']]
  fit <- cluster::agnes(X, method='single', metric='euclidean')
  labels <- cutree(as.hclust(fit), k=4)
  times[[i]] <- (proc.time()[['elapsed']] - t0) * 1000
}
times_txt <- paste(sprintf('%.12f', times), collapse=',')
cat(sprintf('{"status":"ok","fit_ms":%.12f,"fit_ms_all":[%s],"n_clusters":%d}', median(times), times_txt, length(unique(labels))))
""",
            encoding="utf-8",
        )
        proc = subprocess.run(["Rscript", str(script), str(path), str(X.shape[0]), str(X.shape[1]), str(repeats)], text=True, capture_output=True, timeout=600)
        if proc.returncode != 0:
            return [{"method": "AgglomerativeClustering", "framework": "R cluster", "status": "error", "notes": proc.stderr[-500:]}]
        parsed = json.loads(proc.stdout)
        return [{"method": "AgglomerativeClustering", "framework": "R cluster", "backend": "cpu", **parsed}]


def bench_umap_tsne(X, repeats, warmup, seed):
    rows = []
    try:
        import umap
        from sklearn.manifold import trustworthiness

        model, ms, all_ms = _time_call(lambda: umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed, n_epochs=30).fit_transform(X), repeats, warmup)
        rows.append({"method": "UMAP", "framework": "umap-learn", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "shape": list(np.asarray(model).shape), "trustworthiness": float(trustworthiness(X, np.asarray(model), n_neighbors=10))})
    except Exception as exc:
        rows.append({"method": "UMAP", "framework": "umap-learn", "status": "skipped", "notes": repr(exc)})
    try:
        from openTSNE import TSNE
        from sklearn.manifold import trustworthiness

        model, ms, all_ms = _time_call(lambda: np.asarray(TSNE(n_components=2, perplexity=30, n_iter=100, initialization="pca", random_state=seed, n_jobs=1).fit(X)), repeats, warmup)
        rows.append({"method": "TSNE", "framework": "openTSNE", "backend": "cpu", "status": "ok", "fit_ms": ms, "fit_ms_all": all_ms, "shape": list(np.asarray(model).shape), "trustworthiness": float(trustworthiness(X, np.asarray(model), n_neighbors=10))})
    except Exception as exc:
        rows.append({"method": "TSNE", "framework": "openTSNE", "status": "skipped", "notes": repr(exc)})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--devices", default="cpu,cuda,torch")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--json-out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    X_cluster = make_blobs(args.seed, 600, 8, 4, scale=0.35)
    X_gmm = make_blobs(args.seed + 1, 1200, 12, 4, scale=0.6)
    X_nmf = make_nmf(args.seed + 2, 800, 64, 8)
    X_embed = make_blobs(args.seed + 3, 400, 16, 6, scale=0.5)

    rows: List[Dict[str, Any]] = []
    rows.extend(bench_dbscan(X_cluster, devices, args.repeats, args.warmup_runs))
    rows.extend(bench_gmm(X_gmm, devices, args.repeats, args.warmup_runs, args.seed))
    rows.extend(bench_nmf(X_nmf, devices, args.repeats, args.warmup_runs, args.seed))
    rows.extend(bench_agglomerative(X_cluster[:200], args.repeats, args.warmup_runs))
    rows.extend(bench_umap_tsne(X_embed, args.repeats, args.warmup_runs, args.seed))

    result = {"seed": args.seed, "devices": devices, "repeats": args.repeats, "warmup_runs": args.warmup_runs, "rows": rows}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
