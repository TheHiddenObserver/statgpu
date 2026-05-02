"""
Matrix benchmark for statgpu unsupervised estimators vs sklearn/statsmodels/R.

The script is intended for remote validation where GPU backends and R are
available. It keeps GPU performance claims on device-resident inputs by using
``--preload-device-data``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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


def _sync():
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


def _time_call(fn, repeats=1, warmup=0):
    for _ in range(max(0, int(warmup))):
        fn()
        _sync()
    vals = []
    out = None
    for _ in range(max(1, int(repeats))):
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


def make_data(seed: int, n: int, p: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, p))
    labels = rng.integers(0, k, size=n)
    X = centers[labels] + rng.normal(scale=0.6, size=(n, p))
    return X.astype(np.float64, copy=False)


def _projector(components: np.ndarray) -> np.ndarray:
    return components.T @ components


def _pca_metrics(model, ref_ev, ref_components) -> Dict[str, float]:
    ev = _to_numpy(model.explained_variance_)
    comp = _to_numpy(model.components_)
    return {
        "max_abs_explained_variance_diff_vs_ref": float(np.max(np.abs(ev - ref_ev))),
        "max_abs_projector_diff_vs_ref": float(np.max(np.abs(_projector(comp) - _projector(ref_components)))),
    }


def bench_statgpu_pca(X, components, devices, solver, repeats, warmup, preload) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    refs: Dict[int, PCA] = {}
    for m in components:
        for device in devices:
            if not _device_available(device):
                rows.append({"method": "PCA", "framework": "statgpu", "backend": device, "n_components": m, "status": "skipped"})
                continue
            X_fit = _as_device_input(X, device) if preload else X
            model, fit_ms, fit_ms_all = _time_call(
                lambda device=device, m=m: PCA(n_components=m, svd_solver=solver, device=device).fit(X_fit),
                repeats=repeats,
                warmup=warmup,
            )
            if device == "cpu":
                refs[m] = model
            row: Dict[str, Any] = {
                "method": "PCA",
                "framework": "statgpu",
                "backend": device,
                "solver": solver,
                "n_components": m,
                "status": "ok",
                "fit_ms": fit_ms,
                "fit_ms_all": fit_ms_all,
                "explained_variance_sum": float(np.sum(_to_numpy(model.explained_variance_))),
            }
            if m in refs and device != "cpu":
                row.update(_pca_metrics(model, _to_numpy(refs[m].explained_variance_), _to_numpy(refs[m].components_)))
            rows.append(row)
    return rows


def bench_sklearn_pca(X, components, ref_by_component, repeats, warmup, solver="full") -> List[Dict[str, Any]]:
    rows = []
    try:
        from sklearn.decomposition import PCA as SklearnPCA
    except Exception as exc:
        return [{"method": "PCA", "framework": "sklearn", "status": "skipped", "notes": repr(exc)}]
    for m in components:
        model, fit_ms, fit_ms_all = _time_call(
            lambda m=m: SklearnPCA(n_components=m, svd_solver=solver, random_state=20260430).fit(X),
            repeats=repeats,
            warmup=warmup,
        )
        row: Dict[str, Any] = {
            "method": "PCA",
            "framework": "sklearn",
            "backend": "cpu",
            "solver": solver,
            "n_components": m,
            "status": "ok",
            "fit_ms": fit_ms,
            "fit_ms_all": fit_ms_all,
            "explained_variance_sum": float(np.sum(model.explained_variance_)),
        }
        ref = ref_by_component.get(m)
        if ref is not None:
            row.update(_pca_metrics(model, _to_numpy(ref.explained_variance_), _to_numpy(ref.components_)))
        rows.append(row)
    return rows


def bench_statsmodels_pca(X, components, ref_by_component, repeats, warmup) -> List[Dict[str, Any]]:
    rows = []
    try:
        from statsmodels.multivariate.pca import PCA as SMPCA
    except Exception as exc:
        return [{"method": "PCA", "framework": "statsmodels", "status": "skipped", "notes": repr(exc)}]
    for m in components:
        try:
            model, fit_ms, fit_ms_all = _time_call(
                lambda m=m: SMPCA(X, ncomp=m, standardize=False, demean=True, normalize=False, method="svd"),
                repeats=repeats,
                warmup=warmup,
            )
            eigenvals = np.asarray(model.eigenvals[:m], dtype=np.float64) / float(X.shape[0] - 1)
            components_sm = np.asarray(model.loadings.iloc[:, :m].T if hasattr(model.loadings, "iloc") else model.loadings[:, :m].T)
            row: Dict[str, Any] = {
                "method": "PCA",
                "framework": "statsmodels",
                "backend": "cpu",
                "solver": "svd",
                "n_components": m,
                "status": "ok",
                "fit_ms": fit_ms,
                "fit_ms_all": fit_ms_all,
                "explained_variance_sum": float(np.sum(eigenvals)),
            }
            ref = ref_by_component.get(m)
            if ref is not None:
                row["max_abs_explained_variance_diff_vs_ref"] = float(np.max(np.abs(eigenvals - _to_numpy(ref.explained_variance_))))
                row["max_abs_projector_diff_vs_ref"] = float(np.max(np.abs(_projector(components_sm) - _projector(_to_numpy(ref.components_)))))
            rows.append(row)
        except Exception as exc:
            rows.append({"method": "PCA", "framework": "statsmodels", "n_components": m, "status": "error", "notes": repr(exc)})
    return rows


def _write_matrix_bin(X: np.ndarray, path: Path) -> None:
    np.ascontiguousarray(X, dtype=np.float64).tofile(path)


def bench_r_pca(X, components, repeats, timeout) -> List[Dict[str, Any]]:
    if shutil.which("Rscript") is None:
        return [{"method": "PCA", "framework": "R", "status": "skipped", "notes": "Rscript not found"}]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data_bin = td_path / "X.bin"
        script = td_path / "pca.R"
        _write_matrix_bin(X, data_bin)
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
path <- args[[1]]
n <- as.integer(args[[2]])
p <- as.integer(args[[3]])
m <- as.integer(args[[4]])
repeats <- as.integer(args[[5]])
x <- readBin(path, what='numeric', n=n*p, size=8, endian='little')
X <- matrix(x, nrow=n, ncol=p, byrow=TRUE)
times <- numeric(repeats)
for (i in seq_len(repeats)) {
  t0 <- proc.time()[['elapsed']]
  fit <- prcomp(X, center=TRUE, scale.=FALSE, rank.=m)
  times[[i]] <- (proc.time()[['elapsed']] - t0) * 1000
}
vars <- (fit$sdev[seq_len(m)] ^ 2)
times_txt <- paste(sprintf('%.12f', times), collapse=',')
cat(sprintf('{"fit_ms":%.12f,"fit_ms_all":[%s],"explained_variance_sum":%.12f}', median(times), times_txt, sum(vars)))
""",
            encoding="utf-8",
        )
        for m in components:
            try:
                proc = subprocess.run(
                    ["Rscript", str(script), str(data_bin), str(X.shape[0]), str(X.shape[1]), str(m), str(repeats)],
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    rows.append({"method": "PCA", "framework": "R", "n_components": m, "status": "error", "notes": proc.stderr[-500:]})
                    continue
                parsed = json.loads(proc.stdout)
                rows.append(
                    {
                        "method": "PCA",
                        "framework": "R",
                        "backend": "cpu",
                        "solver": "prcomp",
                        "n_components": m,
                        "status": "ok",
                        **parsed,
                    }
                )
            except subprocess.TimeoutExpired:
                rows.append({"method": "PCA", "framework": "R", "n_components": m, "status": "timeout"})
    return rows


def bench_r_kmeans(X, refs: Dict[int, KMeans], repeats: int, timeout: int, max_iter: int) -> List[Dict[str, Any]]:
    if shutil.which("Rscript") is None:
        return [{"method": "KMeans", "framework": "R", "status": "skipped", "notes": "Rscript not found"}]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data_bin = td_path / "X.bin"
        centers_bin = td_path / "centers.bin"
        script = td_path / "kmeans.R"
        _write_matrix_bin(X, data_bin)
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
x_path <- args[[1]]
c_path <- args[[2]]
n <- as.integer(args[[3]])
p <- as.integer(args[[4]])
k <- as.integer(args[[5]])
repeats <- as.integer(args[[6]])
max_iter <- as.integer(args[[7]])
x <- readBin(x_path, what='numeric', n=n*p, size=8, endian='little')
centers <- readBin(c_path, what='numeric', n=k*p, size=8, endian='little')
X <- matrix(x, nrow=n, ncol=p, byrow=TRUE)
C <- matrix(centers, nrow=k, ncol=p, byrow=TRUE)
times <- numeric(repeats)
for (i in seq_len(repeats)) {
  t0 <- proc.time()[['elapsed']]
  fit <- kmeans(X, centers=C, iter.max=max_iter, algorithm='Lloyd')
  times[[i]] <- (proc.time()[['elapsed']] - t0) * 1000
}
times_txt <- paste(sprintf('%.12f', times), collapse=',')
cat(sprintf('{"fit_ms":%.12f,"fit_ms_all":[%s],"inertia":%.12f,"n_iter":%d}', median(times), times_txt, fit$tot.withinss, fit$iter))
""",
            encoding="utf-8",
        )
        for k, ref in refs.items():
            np.ascontiguousarray(_to_numpy(ref.cluster_centers_), dtype=np.float64).tofile(centers_bin)
            try:
                proc = subprocess.run(
                    [
                        "Rscript",
                        str(script),
                        str(data_bin),
                        str(centers_bin),
                        str(X.shape[0]),
                        str(X.shape[1]),
                        str(k),
                        str(repeats),
                        str(max_iter),
                    ],
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    rows.append({"method": "KMeans", "framework": "R", "k": k, "status": "error", "notes": proc.stderr[-500:]})
                    continue
                parsed = json.loads(proc.stdout)
                rows.append(
                    {
                        "method": "KMeans",
                        "framework": "R",
                        "backend": "cpu",
                        "k": k,
                        "status": "ok",
                        **parsed,
                        "abs_inertia_diff_vs_ref": float(abs(parsed["inertia"] - ref.inertia_)),
                    }
                )
            except subprocess.TimeoutExpired:
                rows.append({"method": "KMeans", "framework": "R", "k": k, "status": "timeout"})
    return rows


def _match_centers(a, b):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return np.nan
    distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(distances)
    return float(np.max(np.sqrt(distances[rows, cols])))


def bench_kmeans_matrix(X, ks, devices, repeats, warmup, preload, seed, max_iter) -> List[Dict[str, Any]]:
    rows = []
    refs: Dict[int, KMeans] = {}
    for k in ks:
        for device in devices:
            if not _device_available(device):
                rows.append({"method": "KMeans", "framework": "statgpu", "backend": device, "k": k, "status": "skipped"})
                continue
            X_fit = _as_device_input(X, device) if preload else X
            model, fit_ms, fit_ms_all = _time_call(
                lambda device=device, k=k: KMeans(n_clusters=k, n_init=2, max_iter=max_iter, random_state=seed, device=device).fit(X_fit),
                repeats=repeats,
                warmup=warmup,
            )
            if device == "cpu":
                refs[k] = model
            row = {"method": "KMeans", "framework": "statgpu", "backend": device, "k": k, "status": "ok", "fit_ms": fit_ms, "fit_ms_all": fit_ms_all, "inertia": float(model.inertia_), "n_iter": int(model.n_iter_)}
            if k in refs and device != "cpu":
                row["abs_inertia_diff_vs_ref"] = float(abs(model.inertia_ - refs[k].inertia_))
                row["max_center_distance_vs_ref"] = _match_centers(_to_numpy(model.cluster_centers_), _to_numpy(refs[k].cluster_centers_))
            rows.append(row)
    try:
        from sklearn.cluster import KMeans as SkKMeans
        for k in ks:
            ref = refs.get(k)
            init = _to_numpy(ref.cluster_centers_) if ref is not None else "k-means++"
            n_init = 1 if ref is not None else 2
            model, fit_ms, fit_ms_all = _time_call(
                lambda k=k, init=init, n_init=n_init: SkKMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=seed, algorithm="lloyd").fit(X),
                repeats=repeats,
                warmup=warmup,
            )
            row = {"method": "KMeans", "framework": "sklearn", "backend": "cpu", "k": k, "status": "ok", "fit_ms": fit_ms, "fit_ms_all": fit_ms_all, "inertia": float(model.inertia_), "n_iter": int(model.n_iter_)}
            if ref is not None:
                row["abs_inertia_diff_vs_ref"] = float(abs(model.inertia_ - ref.inertia_))
                row["max_center_distance_vs_ref"] = _match_centers(model.cluster_centers_, _to_numpy(ref.cluster_centers_))
            rows.append(row)
    except Exception as exc:
        rows.append({"method": "KMeans", "framework": "sklearn", "status": "skipped", "notes": repr(exc)})
    rows.extend(bench_r_kmeans(X, refs, repeats, timeout=600, max_iter=max_iter))
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--p", type=int, default=200)
    p.add_argument("--k-data", type=int, default=10)
    p.add_argument("--components", type=str, default="5,10,50,100")
    p.add_argument("--ks", type=str, default="5,10,25")
    p.add_argument("--devices", type=str, default="cpu,cuda,torch")
    p.add_argument("--pca-solver", choices=["auto", "randomized"], default="auto")
    p.add_argument("--methods", type=str, default="pca,kmeans")
    p.add_argument("--preload-device-data", action="store_true")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--r-timeout", type=int, default=600)
    p.add_argument("--kmeans-max-iter", type=int, default=80)
    p.add_argument("--json-out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    components = [int(x) for x in args.components.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    methods = {x.strip().lower() for x in args.methods.split(",") if x.strip()}
    X = make_data(args.seed, args.n, args.p, args.k_data)

    rows: List[Dict[str, Any]] = []
    refs: Dict[int, PCA] = {}
    if "pca" in methods:
        sg_rows = bench_statgpu_pca(X, components, devices, args.pca_solver, args.repeats, args.warmup_runs, args.preload_device_data)
        rows.extend(sg_rows)
        for row in sg_rows:
            if row.get("framework") == "statgpu" and row.get("backend") == "cpu" and row.get("status") == "ok":
                # Refit once to keep a precise object reference for external metrics.
                m = int(row["n_components"])
                refs[m] = PCA(n_components=m, svd_solver=args.pca_solver, device="cpu").fit(X)
        rows.extend(bench_sklearn_pca(X, components, refs, args.repeats, args.warmup_runs, solver="full"))
        rows.extend(bench_statsmodels_pca(X, components, refs, args.repeats, args.warmup_runs))
        rows.extend(bench_r_pca(X, components, args.repeats, args.r_timeout))
    if "kmeans" in methods:
        rows.extend(bench_kmeans_matrix(X, ks, devices, args.repeats, args.warmup_runs, args.preload_device_data, args.seed, args.kmeans_max_iter))

    result = {
        "seed": args.seed,
        "n": args.n,
        "p": args.p,
        "k_data": args.k_data,
        "components": components,
        "ks": ks,
        "devices": devices,
        "methods": sorted(methods),
        "pca_solver": args.pca_solver,
        "preload_device_data": bool(args.preload_device_data),
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
