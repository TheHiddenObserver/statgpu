"""Phase 3B validation for GaussianMixture covariance types and agglomerative linkages."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statgpu.unsupervised import AgglomerativeClustering, GaussianMixture


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _backend_array(X, device):
    if device == "cuda":
        import cupy as cp

        return cp.asarray(X)
    if device == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is not available")
        return torch.as_tensor(X, dtype=torch.float64, device="cuda")
    return X


def _time_fit(factory, X, repeats, warmup):
    for _ in range(warmup):
        factory().fit(X)
    times = []
    model = None
    for _ in range(repeats):
        start = time.perf_counter()
        model = factory().fit(X)
        times.append((time.perf_counter() - start) * 1000.0)
    return model, times


def _adjusted_rand_score(labels_true, labels_pred):
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(labels_true, labels_pred))
    except Exception:
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)
        classes, class_idx = np.unique(labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
        contingency = np.zeros((classes.size, clusters.size), dtype=np.int64)
        np.add.at(contingency, (class_idx, cluster_idx), 1)

        def comb2(values):
            values = np.asarray(values, dtype=np.float64)
            return np.sum(values * (values - 1.0) / 2.0)

        sum_comb = comb2(contingency)
        sum_rows = comb2(contingency.sum(axis=1))
        sum_cols = comb2(contingency.sum(axis=0))
        total = comb2(labels_true.size)
        if total == 0.0:
            return 1.0
        expected = sum_rows * sum_cols / total
        maximum = 0.5 * (sum_rows + sum_cols)
        denom = maximum - expected
        if denom == 0.0:
            return 1.0
        return float((sum_comb - expected) / denom)


def make_gmm_data(n, p, k, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, p))
    labels = np.arange(n) % k
    X = centers[labels] + rng.normal(scale=0.45, size=(n, p))
    return X.astype(np.float64)


def make_cluster_data(n, p, k, seed):
    rng = np.random.default_rng(seed + 17)
    centers = rng.normal(scale=5.0, size=(k, p))
    labels = np.arange(n) % k
    X = centers[labels] + rng.normal(scale=0.25, size=(n, p))
    return X.astype(np.float64), labels


def run_gmm_statgpu(X, args):
    out = {}
    for covariance_type in ("diag", "spherical", "tied", "full"):
        out[covariance_type] = {}
        for device in args.devices.split(","):
            device = device.strip()
            if not device:
                continue
            try:
                X_backend = _backend_array(X, device)
                factory = lambda cov=covariance_type, dev=device: GaussianMixture(
                    n_components=args.k,
                    covariance_type=cov,
                    max_iter=args.gmm_iter,
                    tol=args.tol,
                    random_state=args.seed,
                    device=dev,
                )
                model, times = _time_fit(factory, X_backend, args.repeats, args.warmup)
                proba = _to_numpy(model.predict_proba(X_backend))
                out[covariance_type][f"statgpu_{device}"] = {
                    "status": "ok",
                    "fit_ms_mean": float(np.mean(times)),
                    "fit_ms_std": float(np.std(times)),
                    "score": float(model.score(X_backend)),
                    "aic": float(model.aic(X_backend)),
                    "bic": float(model.bic(X_backend)),
                    "proba_row_sum_max_abs_diff": float(np.max(np.abs(proba.sum(axis=1) - 1.0))),
                    "n_iter": int(model.n_iter_),
                    "converged": bool(model.converged_),
                    "host_to_device_included": False,
                }
            except Exception as exc:
                out[covariance_type][f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}
    return out


def run_gmm_sklearn(X, args):
    try:
        from sklearn.mixture import GaussianMixture as SkGaussianMixture
    except Exception as exc:
        return {cov: {"sklearn_cpu": {"status": "skipped", "reason": repr(exc)}} for cov in ("diag", "spherical", "tied", "full")}

    out = {}
    for covariance_type in ("diag", "spherical", "tied", "full"):
        factory = lambda cov=covariance_type: SkGaussianMixture(
            n_components=args.k,
            covariance_type=cov,
            max_iter=args.gmm_iter,
            tol=args.tol,
            random_state=args.seed,
            reg_covar=1e-6,
            n_init=1,
        )
        try:
            model, times = _time_fit(factory, X, args.repeats, args.warmup)
            out[covariance_type] = {
                "sklearn_cpu": {
                    "status": "ok",
                    "fit_ms_mean": float(np.mean(times)),
                    "fit_ms_std": float(np.std(times)),
                    "score": float(model.score(X)),
                    "aic": float(model.aic(X)),
                    "bic": float(model.bic(X)),
                    "n_iter": int(model.n_iter_),
                    "converged": bool(model.converged_),
                }
            }
        except Exception as exc:
            out[covariance_type] = {"sklearn_cpu": {"status": "skipped", "reason": repr(exc)}}
    return out


def run_agglomerative_statgpu(X, true_labels, args):
    out = {}
    for linkage in ("single", "complete", "average", "ward"):
        try:
            factory = lambda link=linkage: AgglomerativeClustering(n_clusters=args.k, linkage=link, device="cpu")
            model, times = _time_fit(factory, X, args.repeats, args.warmup)
            out[linkage] = {
                "statgpu_cpu": {
                    "status": "ok",
                    "fit_ms_mean": float(np.mean(times)),
                    "fit_ms_std": float(np.std(times)),
                    "n_clusters": int(len(np.unique(model.labels_))),
                    "last_distance": float(model.distances_[-1]) if model.distances_.size else 0.0,
                    "ari_vs_truth": _adjusted_rand_score(true_labels, model.labels_),
                }
            }
        except Exception as exc:
            out[linkage] = {"statgpu_cpu": {"status": "skipped", "reason": repr(exc)}}
    return out


def run_agglomerative_external(X, true_labels, args):
    out = {link: {} for link in ("single", "complete", "average", "ward")}
    try:
        from sklearn.cluster import AgglomerativeClustering as SkAgglomerative

        for linkage in out:
            factory = lambda link=linkage: SkAgglomerative(n_clusters=args.k, linkage=link, metric="euclidean")
            model, times = _time_fit(factory, X, args.repeats, args.warmup)
            out[linkage]["sklearn_cpu"] = {
                "status": "ok",
                "fit_ms_mean": float(np.mean(times)),
                "fit_ms_std": float(np.std(times)),
                "ari_vs_truth": _adjusted_rand_score(true_labels, model.labels_),
                "n_clusters": int(len(np.unique(model.labels_))),
            }
    except Exception as exc:
        for linkage in out:
            out[linkage]["sklearn_cpu"] = {"status": "skipped", "reason": repr(exc)}

    try:
        from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage

        for link in out:
            method = "ward" if link == "ward" else link
            factory = lambda method=method: scipy_linkage(X, method=method, metric="euclidean")
            Z, times = _time_fit(lambda: _ScipyLinkageWrapper(factory), None, args.repeats, args.warmup)
            labels = fcluster(Z.Z, t=args.k, criterion="maxclust") - 1
            out[link]["scipy_cpu"] = {
                "status": "ok",
                "fit_ms_mean": float(np.mean(times)),
                "fit_ms_std": float(np.std(times)),
                "ari_vs_truth": _adjusted_rand_score(true_labels, labels),
                "last_distance": float(Z.Z[-1, 2]) if Z.Z.size else 0.0,
            }
    except Exception as exc:
        for linkage in out:
            out[linkage]["scipy_cpu"] = {"status": "skipped", "reason": repr(exc)}

    if args.skip_r:
        for linkage in out:
            out[linkage]["R_cluster_agnes"] = {"status": "skipped", "reason": "--skip-r"}
    else:
        r_result = run_r_agnes(X, args)
        for linkage in out:
            out[linkage]["R_cluster_agnes"] = r_result.get(linkage, {"status": "skipped", "reason": "missing R result"})
    return out


class _ScipyLinkageWrapper:
    def __init__(self, factory):
        self.Z = factory()

    def fit(self, X):
        return self


def run_r_agnes(X, args):
    mapping = {"single": "single", "complete": "complete", "average": "average", "ward": "ward"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "X.csv"
        json_path = tmp_path / "out.json"
        np.savetxt(csv_path, X, delimiter=",")
        script = tmp_path / "agnes.R"
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
X <- as.matrix(read.csv(args[1], header=FALSE))
out_path <- args[2]
methods <- c(single="single", complete="complete", average="average", ward="ward")
res <- list()
ok <- requireNamespace("cluster", quietly=TRUE) && requireNamespace("jsonlite", quietly=TRUE)
if (!ok) {
  for (nm in names(methods)) res[[nm]] <- list(status="skipped", reason="cluster or jsonlite unavailable")
} else {
  for (nm in names(methods)) {
    t <- system.time(cluster::agnes(X, diss=FALSE, method=methods[[nm]], keep.diss=FALSE))
    res[[nm]] <- list(status="ok", fit_ms_mean=unname(t[["elapsed"]]) * 1000)
  }
}
jsonlite::write_json(res, out_path, auto_unbox=TRUE)
""",
            encoding="utf-8",
        )
        try:
            subprocess.run(["Rscript", str(script), str(csv_path), str(json_path)], check=True, capture_output=True, text=True)
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {link: {"status": "skipped", "reason": repr(exc)} for link in mapping}


def write_summary(results, path):
    lines = [
        "# Unsupervised Phase 3B Validation Summary",
        "",
        "## Runtime Table",
        "",
        "| Model | Variant | Backend/framework | Mean ms | Status |",
        "|---|---|---:|---:|---|",
    ]
    for model_name, variants in results["models"].items():
        for variant, entries in variants.items():
            for backend_name, payload in entries.items():
                mean = payload.get("fit_ms_mean", "")
                mean_text = f"{mean:.3f}" if isinstance(mean, (float, int)) else ""
                lines.append(f"| {model_name} | {variant} | {backend_name} | {mean_text} | {payload.get('status', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--gmm-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", default="cpu,cuda,torch")
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    X_gmm = make_gmm_data(args.n, args.p, args.k, args.seed)
    X_cluster, true_labels = make_cluster_data(args.n, min(args.p, 8), args.k, args.seed)
    gmm = run_gmm_statgpu(X_gmm, args)
    sk_gmm = run_gmm_sklearn(X_gmm, args)
    for cov, entries in sk_gmm.items():
        gmm.setdefault(cov, {}).update(entries)

    agg = run_agglomerative_statgpu(X_cluster, true_labels, args)
    agg_external = run_agglomerative_external(X_cluster, true_labels, args)
    for linkage, entries in agg_external.items():
        agg.setdefault(linkage, {}).update(entries)

    results = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "args": {
                **vars(args),
                "json_out": str(args.json_out) if args.json_out else None,
                "md_out": str(args.md_out) if args.md_out else None,
            },
        },
        "models": {
            "GaussianMixture": gmm,
            "AgglomerativeClustering": agg,
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        write_summary(results, args.md_out)
    if args.print_json:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
