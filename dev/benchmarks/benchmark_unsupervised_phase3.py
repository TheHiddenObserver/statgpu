"""Phase 3 unsupervised accuracy/runtime comparison.

This script is intended for Matpool remote validation. It never stores SSH
credentials and only writes environment, timing, and numerical metrics.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# When this file is executed as ``python dev/benchmarks/...py``, Python puts
# ``dev/benchmarks`` before the repository root on sys.path. Prefer the current
# worktree over any editable install/import hook in the local environment.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from statgpu.unsupervised import MiniBatchKMeans, TSNE, TruncatedSVD, UMAP


def _available(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        return exc


def _sync_backend(device):
    if device == "cuda":
        cp = _available("cupy")
        if not isinstance(cp, Exception):
            cp.cuda.Stream.null.synchronize()
    if device == "torch":
        torch = _available("torch")
        if not isinstance(torch, Exception) and torch.cuda.is_available():
            torch.cuda.synchronize()


def _to_backend_array(X, device):
    if device == "cpu":
        return X
    if device == "cuda":
        cp = _available("cupy")
        if isinstance(cp, Exception):
            raise cp
        return cp.asarray(X)
    if device == "torch":
        torch = _available("torch")
        if isinstance(torch, Exception):
            raise torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA is unavailable")
        return torch.as_tensor(X, dtype=torch.float64, device="cuda")
    raise ValueError(device)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _time_call(fn, device="cpu", warmup=1, repeats=3):
    cold_t0 = time.perf_counter()
    cold_out = fn()
    _sync_backend(device)
    cold = time.perf_counter() - cold_t0
    for _ in range(warmup):
        fn()
        _sync_backend(device)
    times = []
    last = cold_out
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        _sync_backend(device)
        times.append(time.perf_counter() - t0)
    return last, {
        "cold_ms": cold * 1000.0,
        "warm_mean_ms": float(np.mean(times) * 1000.0),
        "warm_std_ms": float(np.std(times) * 1000.0),
        "repeats": int(repeats),
        "warmup": int(warmup),
    }


def _projector_diff(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return float(np.linalg.norm(A.T @ A - B.T @ B))


def _center_match_distance(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return float("nan")
    D = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2) ** 0.5
    rows, cols = linear_sum_assignment(D)
    return float(D[rows, cols].mean())


def _trustworthiness(X, embedding):
    try:
        from sklearn.manifold import trustworthiness
    except Exception:
        return None
    n_neighbors = min(5, X.shape[0] - 1)
    return float(trustworthiness(X, embedding, n_neighbors=n_neighbors))


def _knn_preservation(X, embedding, k=5):
    k = min(k, X.shape[0] - 1)
    dx = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    dy = ((embedding[:, None, :] - embedding[None, :, :]) ** 2).sum(axis=2)
    nx = np.argsort(dx, axis=1)[:, 1 : k + 1]
    ny = np.argsort(dy, axis=1)[:, 1 : k + 1]
    score = [len(set(nx[i]).intersection(set(ny[i]))) / float(k) for i in range(X.shape[0])]
    return float(np.mean(score))


def make_data(args):
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.p))
    centers = rng.normal(scale=4.0, size=(args.k, args.p))
    labels = rng.integers(0, args.k, size=args.n)
    X_cluster = centers[labels] + 0.35 * rng.normal(size=(args.n, args.p))
    init_idx = rng.choice(args.n, size=args.k, replace=False)
    return X.astype(np.float64), X_cluster.astype(np.float64), X_cluster[init_idx].copy()


def _devices(args):
    return [item.strip() for item in str(args.devices).split(",") if item.strip()]


def run_truncated_svd(X, args):
    out = {}
    reference = None
    for device in _devices(args):
        print(f"  TruncatedSVD statgpu_{device}", flush=True)
        try:
            X_dev = _to_backend_array(X, device)
            model = TruncatedSVD(
                n_components=args.components,
                algorithm=args.svd_algorithm,
                random_state=args.seed,
                device=device,
            )
            fitted, timing = _time_call(lambda: model.fit(X_dev), device, args.warmup, args.repeats)
            comps = _to_numpy(fitted.components_)
            svals = _to_numpy(fitted.singular_values_)
            if reference is None:
                reference = (comps, svals)
            out[f"statgpu_{device}"] = {
                "timing": timing,
                "singular_values_max_abs_diff_vs_statgpu_cpu": float(np.max(np.abs(svals - reference[1]))),
                "projector_diff_vs_statgpu_cpu": _projector_diff(comps, reference[0]),
                "status": "ok",
            }
        except Exception as exc:
            out[f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}

    if args.skip_external:
        out["sklearn_cpu"] = {"status": "skipped", "reason": "--skip-external"}
        out["statsmodels_cpu"] = {"status": "skipped", "reason": "--skip-external"}
        out["cuml_gpu"] = {"status": "skipped", "reason": "--skip-external"}
        return out

    print("  TruncatedSVD sklearn_cpu", flush=True)
    sklearn = _available("sklearn.decomposition")
    if not isinstance(sklearn, Exception):
        model = sklearn.TruncatedSVD(n_components=args.components, algorithm="randomized", random_state=args.seed)
        fitted, timing = _time_call(lambda: model.fit(X), "cpu", args.warmup, args.repeats)
        out["sklearn_cpu"] = {
            "timing": timing,
            "singular_values_max_abs_diff_vs_statgpu_cpu": float(
                np.max(np.abs(fitted.singular_values_ - reference[1]))
            ) if reference is not None else None,
            "projector_diff_vs_statgpu_cpu": _projector_diff(fitted.components_, reference[0])
            if reference is not None else None,
            "status": "ok",
        }
    else:
        out["sklearn_cpu"] = {"status": "skipped", "reason": repr(sklearn)}

    print("  TruncatedSVD statsmodels_cpu", flush=True)
    sm_pca = _available("statsmodels.multivariate.pca")
    if not isinstance(sm_pca, Exception):
        try:
            _, timing = _time_call(lambda: sm_pca.PCA(X, ncomp=args.components, method="svd"), "cpu", args.warmup, args.repeats)
            out["statsmodels_cpu"] = {"timing": timing, "status": "ok", "note": "PCA-style centered baseline"}
        except Exception as exc:
            out["statsmodels_cpu"] = {"status": "skipped", "reason": repr(exc)}
    else:
        out["statsmodels_cpu"] = {"status": "skipped", "reason": repr(sm_pca)}

    cuml = _available("cuml")
    out["cuml_gpu"] = {"status": "skipped", "reason": repr(cuml)} if isinstance(cuml, Exception) else {
        "status": "skipped",
        "reason": "cuML TruncatedSVD/PCA timing not wired in this script",
    }
    return out


def run_minibatch_kmeans(X, init_centers, args):
    out = {}
    reference_centers = None
    reference_labels = None
    for device in _devices(args):
        print(f"  MiniBatchKMeans statgpu_{device}", flush=True)
        try:
            X_dev = _to_backend_array(X, device)
            init_dev = _to_backend_array(init_centers, device)
            model = MiniBatchKMeans(
                n_clusters=args.k,
                init=init_dev,
                batch_size=args.batch_size,
                max_iter=args.mb_iter,
                max_no_improvement=None,
                random_state=args.seed,
                device=device,
            )
            fitted, timing = _time_call(lambda: model.fit(X_dev), device, args.warmup, args.repeats)
            centers = _to_numpy(fitted.cluster_centers_)
            labels = _to_numpy(fitted.labels_)
            if reference_centers is None:
                reference_centers = centers
                reference_labels = labels
            ari = None
            try:
                from sklearn.metrics import adjusted_rand_score
                ari = float(adjusted_rand_score(reference_labels, labels))
            except Exception:
                pass
            out[f"statgpu_{device}"] = {
                "timing": timing,
                "inertia": float(fitted.inertia_),
                "center_distance_vs_statgpu_cpu": _center_match_distance(centers, reference_centers),
                "ari_vs_statgpu_cpu": ari,
                "status": "ok",
            }
        except Exception as exc:
            out[f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}

    if args.skip_external:
        out["sklearn_cpu"] = {"status": "skipped", "reason": "--skip-external"}
        out["cuml_gpu"] = {"status": "skipped", "reason": "--skip-external"}
        return out

    print("  MiniBatchKMeans sklearn_cpu", flush=True)
    sk_cluster = _available("sklearn.cluster")
    if not isinstance(sk_cluster, Exception):
        model = sk_cluster.MiniBatchKMeans(
            n_clusters=args.k,
            init=init_centers,
            n_init=1,
            batch_size=args.batch_size,
            max_iter=args.mb_iter,
            max_no_improvement=None,
            random_state=args.seed,
        )
        fitted, timing = _time_call(lambda: model.fit(X), "cpu", args.warmup, args.repeats)
        out["sklearn_cpu"] = {
            "timing": timing,
            "inertia": float(fitted.inertia_),
            "center_distance_vs_statgpu_cpu": _center_match_distance(fitted.cluster_centers_, reference_centers)
            if reference_centers is not None else None,
            "status": "ok",
        }
    else:
        out["sklearn_cpu"] = {"status": "skipped", "reason": repr(sk_cluster)}
    cuml = _available("cuml")
    out["cuml_gpu"] = {"status": "skipped", "reason": repr(cuml)} if isinstance(cuml, Exception) else {
        "status": "skipped",
        "reason": "cuML MiniBatchKMeans timing not available",
    }
    return out


def run_umap(X, args):
    out = {}
    for device in _devices(args):
        print(f"  UMAP statgpu_{device}", flush=True)
        try:
            X_dev = _to_backend_array(X, device)
            model = UMAP(
                n_neighbors=args.neighbors,
                n_components=2,
                n_epochs=args.manifold_epochs,
                init="random",
                random_state=args.seed,
                device=device,
            )
            fitted, timing = _time_call(lambda: model.fit(X_dev), device, args.warmup, args.repeats)
            embedding = _to_numpy(fitted.embedding_)
            out[f"statgpu_{device}"] = {
                "timing": timing,
                "trustworthiness": _trustworthiness(X, embedding),
                "knn_preservation": _knn_preservation(X, embedding),
                "status": "ok",
            }
        except Exception as exc:
            out[f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}

    if args.skip_external:
        out["umap_learn_cpu"] = {"status": "skipped", "reason": "--skip-external"}
        out["cuml_gpu"] = {"status": "skipped", "reason": "--skip-external"}
        return out

    print("  UMAP umap_learn_cpu", flush=True)
    umap_mod = _available("umap")
    if not isinstance(umap_mod, Exception):
        model = umap_mod.UMAP(
            n_neighbors=args.neighbors,
            n_components=2,
            n_epochs=args.manifold_epochs,
            init="random",
            random_state=args.seed,
        )
        fitted, timing = _time_call(lambda: model.fit(X), "cpu", args.warmup, args.repeats)
        embedding = np.asarray(fitted.embedding_)
        out["umap_learn_cpu"] = {
            "timing": timing,
            "trustworthiness": _trustworthiness(X, embedding),
            "knn_preservation": _knn_preservation(X, embedding),
            "status": "ok",
        }
    else:
        out["umap_learn_cpu"] = {"status": "skipped", "reason": repr(umap_mod)}
    cuml = _available("cuml")
    out["cuml_gpu"] = {"status": "skipped", "reason": repr(cuml)} if isinstance(cuml, Exception) else {
        "status": "skipped",
        "reason": "cuML UMAP timing not wired in this script",
    }
    return out


def run_tsne(X, args):
    out = {}
    for device in _devices(args):
        print(f"  TSNE statgpu_{device}", flush=True)
        try:
            X_dev = _to_backend_array(X, device)
            model = TSNE(
                n_components=2,
                perplexity=args.perplexity,
                max_iter=args.tsne_iter,
                init="random",
                random_state=args.seed,
                device=device,
            )
            fitted, timing = _time_call(lambda: model.fit(X_dev), device, args.warmup, args.repeats)
            embedding = _to_numpy(fitted.embedding_)
            out[f"statgpu_{device}"] = {
                "timing": timing,
                "kl_divergence": float(fitted.kl_divergence_),
                "trustworthiness": _trustworthiness(X, embedding),
                "status": "ok",
            }
        except Exception as exc:
            out[f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}

    if args.skip_external:
        out["sklearn_cpu"] = {"status": "skipped", "reason": "--skip-external"}
        out["openTSNE_cpu"] = {"status": "skipped", "reason": "--skip-external"}
        out["cuml_gpu"] = {"status": "skipped", "reason": "--skip-external"}
        return out

    print("  TSNE sklearn_cpu", flush=True)
    sk_manifold = _available("sklearn.manifold")
    if not isinstance(sk_manifold, Exception):
        try:
            model = sk_manifold.TSNE(
                n_components=2,
                perplexity=args.perplexity,
                max_iter=args.tsne_iter,
                init="random",
                random_state=args.seed,
                method="exact",
            )
        except TypeError:
            model = sk_manifold.TSNE(
                n_components=2,
                perplexity=args.perplexity,
                n_iter=args.tsne_iter,
                init="random",
                random_state=args.seed,
                method="exact",
            )
        fitted, timing = _time_call(lambda: model.fit(X), "cpu", args.warmup, args.repeats)
        embedding = np.asarray(fitted.embedding_)
        out["sklearn_cpu"] = {
            "timing": timing,
            "kl_divergence": float(getattr(fitted, "kl_divergence_", math.nan)),
            "trustworthiness": _trustworthiness(X, embedding),
            "status": "ok",
        }
    else:
        out["sklearn_cpu"] = {"status": "skipped", "reason": repr(sk_manifold)}

    print("  TSNE openTSNE_cpu", flush=True)
    opentsne = _available("openTSNE")
    if not isinstance(opentsne, Exception):
        try:
            model = opentsne.TSNE(
                n_components=2,
                perplexity=args.perplexity,
                n_iter=args.tsne_iter,
                initialization="random",
                random_state=args.seed,
            )
            embedding, timing = _time_call(lambda: model.fit(X), "cpu", args.warmup, args.repeats)
            embedding = np.asarray(embedding)
            out["openTSNE_cpu"] = {
                "timing": timing,
                "trustworthiness": _trustworthiness(X, embedding),
                "status": "ok",
            }
        except Exception as exc:
            out["openTSNE_cpu"] = {"status": "skipped", "reason": repr(exc)}
    else:
        out["openTSNE_cpu"] = {"status": "skipped", "reason": repr(opentsne)}
    cuml = _available("cuml")
    out["cuml_gpu"] = {"status": "skipped", "reason": repr(cuml)} if isinstance(cuml, Exception) else {
        "status": "skipped",
        "reason": "cuML TSNE timing not wired in this script",
    }
    return out


def run_r_svd_smoke(X, args):
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"status": "skipped", "reason": "Rscript not found"}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.csv"
        np.savetxt(path, X, delimiter=",")
        code = (
            "args <- commandArgs(trailingOnly=TRUE); "
            "x <- as.matrix(read.csv(args[1], header=FALSE)); "
            "t <- system.time({ s <- svd(x, nu=0, nv=as.integer(args[2])) }); "
            "cat(as.numeric(t[['elapsed']]) * 1000)"
        )
        proc = subprocess.run(
            [rscript, "-e", code, str(path), str(args.components)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return {"status": "skipped", "reason": proc.stderr.strip()}
        return {"status": "ok", "timing": {"warm_mean_ms": float(proc.stdout.strip()), "source": "R system.time"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--p", type=int, default=24)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mb-iter", type=int, default=20)
    parser.add_argument("--manifold-epochs", type=int, default=30)
    parser.add_argument("--tsne-iter", type=int, default=250)
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--svd-algorithm", choices=["randomized", "full"], default="randomized")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--devices", default="cpu,cuda,torch")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--debug-timeout", type=int, default=0)
    args = parser.parse_args()
    if args.debug_timeout:
        faulthandler.enable()
        faulthandler.dump_traceback_later(int(args.debug_timeout), repeat=True)

    X, X_cluster, init_centers = make_data(args)
    args_metadata = {
        **vars(args),
        "json_out": str(args.json_out) if args.json_out else None,
        "md_out": str(args.md_out) if args.md_out else None,
    }
    results = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "args": args_metadata,
        },
        "models": {
            "TruncatedSVD": None,
            "MiniBatchKMeans": None,
            "UMAP": None,
            "TSNE": None,
        },
    }
    print("running TruncatedSVD", flush=True)
    results["models"]["TruncatedSVD"] = run_truncated_svd(X, args)
    if args.skip_r:
        results["models"]["TruncatedSVD"]["R_cpu"] = {"status": "skipped", "reason": "--skip-r"}
    else:
        print("running R SVD smoke", flush=True)
        results["models"]["TruncatedSVD"]["R_cpu"] = run_r_svd_smoke(X, args)
    print("running MiniBatchKMeans", flush=True)
    results["models"]["MiniBatchKMeans"] = run_minibatch_kmeans(X_cluster, init_centers, args)
    print("running UMAP", flush=True)
    results["models"]["UMAP"] = run_umap(X, args)
    print("running TSNE", flush=True)
    results["models"]["TSNE"] = run_tsne(X, args)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Unsupervised Phase 3 Benchmark Summary", "", "| Model | Backend/framework | Warm mean ms | Status |", "|---|---:|---:|---|"]
        for model, entries in results["models"].items():
            for name, payload in entries.items():
                timing = payload.get("timing", {}) if isinstance(payload, dict) else {}
                warm = timing.get("warm_mean_ms", "")
                lines.append(f"| {model} | {name} | {warm} | {payload.get('status', '')} |")
        args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.print_json or (args.json_out is None and args.md_out is None):
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        if args.json_out:
            print(f"wrote {args.json_out}", flush=True)
        if args.md_out:
            print(f"wrote {args.md_out}", flush=True)


if __name__ == "__main__":
    main()
