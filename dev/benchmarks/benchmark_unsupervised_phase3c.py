"""Phase 3C validation for IncrementalPCA and MiniBatchNMF."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statgpu.unsupervised import IncrementalPCA, MiniBatchNMF


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


def make_pca_data(n, p, rank, seed):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, rank))
    loadings = rng.normal(size=(rank, p))
    return (latent @ loadings + 0.05 * rng.normal(size=(n, p))).astype(np.float64)


def make_nmf_data(n, p, rank, seed):
    rng = np.random.default_rng(seed + 31)
    W = rng.random((n, rank))
    H = rng.random((rank, p))
    return (W @ H + 0.02 * rng.random((n, p))).astype(np.float64)


def _reconstruction_error(X, X_inv):
    return float(np.linalg.norm(np.asarray(X) - np.asarray(X_inv)))


def run_incremental_pca(X, args):
    out = {}
    cpu_reference = None
    for device in args.devices.split(","):
        device = device.strip()
        if not device:
            continue
        try:
            X_backend = _backend_array(X, device)
            factory = lambda dev=device: IncrementalPCA(
                n_components=args.components,
                batch_size=args.batch_size,
                whiten=False,
                device=dev,
            )
            model, times = _time_fit(factory, X_backend, args.repeats, args.warmup)
            Xt = model.transform(X_backend)
            X_inv = model.inverse_transform(Xt)
            payload = {
                "status": "ok",
                "fit_ms_mean": float(np.mean(times)),
                "fit_ms_std": float(np.std(times)),
                "explained_variance": _to_numpy(model.explained_variance_).tolist(),
                "explained_variance_ratio_sum": float(np.sum(_to_numpy(model.explained_variance_ratio_))),
                "reconstruction_error": _reconstruction_error(X, _to_numpy(X_inv)),
                "n_samples_seen": int(model.n_samples_seen_),
                "host_to_device_included": False,
            }
            out[f"statgpu_{device}"] = payload
            if device == "cpu":
                cpu_reference = payload
        except Exception as exc:
            out[f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}

    try:
        from sklearn.decomposition import IncrementalPCA as SkIncrementalPCA

        factory = lambda: SkIncrementalPCA(n_components=args.components, batch_size=args.batch_size)
        model, times = _time_fit(factory, X, args.repeats, args.warmup)
        Xt = model.transform(X)
        X_inv = model.inverse_transform(Xt)
        out["sklearn_cpu"] = {
            "status": "ok",
            "fit_ms_mean": float(np.mean(times)),
            "fit_ms_std": float(np.std(times)),
            "explained_variance": model.explained_variance_.tolist(),
            "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
            "reconstruction_error": _reconstruction_error(X, X_inv),
            "n_samples_seen": int(model.n_samples_seen_),
        }
    except Exception as exc:
        out["sklearn_cpu"] = {"status": "skipped", "reason": repr(exc)}
    return out


def run_minibatch_nmf(X, args):
    out = {}
    for device in args.devices.split(","):
        device = device.strip()
        if not device:
            continue
        try:
            X_backend = _backend_array(X, device)
            factory = lambda dev=device: MiniBatchNMF(
                n_components=args.components,
                batch_size=args.batch_size,
                max_iter=args.max_iter,
                random_state=args.seed,
                device=dev,
            )
            model, times = _time_fit(factory, X_backend, args.repeats, args.warmup)
            W = model.transform(X_backend)
            X_inv = model.inverse_transform(W)
            out[f"statgpu_{device}"] = {
                "status": "ok",
                "fit_ms_mean": float(np.mean(times)),
                "fit_ms_std": float(np.std(times)),
                "reconstruction_error": _reconstruction_error(X, _to_numpy(X_inv)),
                "components_min": float(np.min(_to_numpy(model.components_))),
                "W_min": float(np.min(_to_numpy(W))),
                "n_iter": int(model.n_iter_),
                "host_to_device_included": False,
            }
        except Exception as exc:
            out[f"statgpu_{device}"] = {"status": "skipped", "reason": repr(exc)}

    try:
        from sklearn.decomposition import MiniBatchNMF as SkMiniBatchNMF

        factory = lambda: SkMiniBatchNMF(
            n_components=args.components,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            random_state=args.seed,
            init="random",
            beta_loss="frobenius",
        )
        model, times = _time_fit(factory, X, args.repeats, args.warmup)
        W = model.transform(X)
        X_inv = model.inverse_transform(W)
        out["sklearn_cpu"] = {
            "status": "ok",
            "fit_ms_mean": float(np.mean(times)),
            "fit_ms_std": float(np.std(times)),
            "reconstruction_error": _reconstruction_error(X, X_inv),
            "components_min": float(np.min(model.components_)),
            "W_min": float(np.min(W)),
            "n_iter": int(model.n_iter_),
        }
    except Exception as exc:
        out["sklearn_cpu"] = {"status": "skipped", "reason": repr(exc)}
    return out


def write_summary(results, path):
    lines = ["# Unsupervised Phase 3C Validation Summary", "", "| Model | Backend/framework | Mean ms | Quality metric | Status |", "|---|---:|---:|---:|---|"]
    for model, entries in results["models"].items():
        for name, payload in entries.items():
            mean = payload.get("fit_ms_mean", "")
            mean_text = f"{mean:.3f}" if isinstance(mean, (float, int)) else ""
            metric = payload.get("reconstruction_error", "")
            metric_text = f"{metric:.6f}" if isinstance(metric, (float, int)) else ""
            lines.append(f"| {model} | {name} | {mean_text} | {metric_text} | {payload.get('status', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", default="cpu,cuda,torch")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    X_pca = make_pca_data(args.n, args.p, args.components, args.seed)
    X_nmf = make_nmf_data(args.n, args.p, args.components, args.seed)
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
            "IncrementalPCA": run_incremental_pca(X_pca, args),
            "MiniBatchNMF": run_minibatch_nmf(X_nmf, args),
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
