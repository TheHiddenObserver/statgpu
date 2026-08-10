#!/usr/bin/env python3
"""Physical performance probe for Panel Stage-C covariance paths.

This benchmark is intentionally separate from the correctness validator. It
measures synchronized end-to-end fit time for representative covariance paths
and records raw samples without making a speedup promise.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from statgpu.panel import PanelOLS, PooledOLS, RandomEffects


PERFORMANCE_SCHEMA_VERSION = 2
DEFAULT_HIGH_T_SCALE = "10000x2x200"
HIGH_T_CASES = ("pooled_dk_qs", "panel_entity_dk_qs")


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status():
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_scales(text):
    values = []
    for token in text.split(","):
        n_text, k_text = token.strip().lower().split("x", 1)
        n, k = int(n_text), int(k_text)
        if n <= 0 or k <= 0:
            raise ValueError("scales must be positive NxK pairs")
        values.append((n, k))
    return values


def _parse_high_t_scale(text):
    parts = text.strip().lower().split("x")
    if len(parts) != 3:
        raise ValueError("high-T scale must be an NxKxT triple")
    n, k, n_times = (int(v) for v in parts)
    if n <= 0 or k <= 0 or n_times < 2 or n < n_times:
        raise ValueError("high-T scale requires positive N/K, T>=2, and N>=T")
    return n, k, n_times


def _timing_row(*, backend, case, scenario, n, k, n_times, repeats, samples):
    return {
        "backend": backend,
        "case": case,
        "scenario": scenario,
        "n_samples": int(n),
        "n_features": int(k),
        "n_times": int(n_times),
        "repeats": int(repeats),
        "median_seconds": float(np.median(samples)),
        "samples_seconds": [float(v) for v in samples],
    }


def _sync(backend):
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch
        torch.cuda.synchronize()


def _dataset(n, k, seed, *, n_times=20):
    if int(n_times) < 2 or int(n) < int(n_times):
        raise ValueError("dataset requires n_times>=2 and n>=n_times")
    rng = np.random.default_rng(seed)
    n_times = int(n_times)
    n_entities = max(2, int(np.ceil(n / n_times)))
    entity = np.repeat(np.arange(n_entities), n_times)[:n]
    time_ids = np.tile(np.arange(n_times), n_entities)[:n]
    X = rng.normal(size=(n, k)).astype(np.float64)
    beta = np.linspace(0.15, 0.75, k, dtype=np.float64)
    alpha_values = rng.normal(scale=0.35, size=n_entities)
    y = X @ beta + alpha_values[entity] + rng.normal(scale=0.25, size=n)
    clusters = np.column_stack([entity, time_ids])
    return X, y.astype(np.float64), entity, time_ids, clusters


def _to_backend(X, y, entity, time_ids, backend):
    if backend == "cupy":
        import cupy as cp
        return (
            cp.asarray(X), cp.asarray(y),
            cp.asarray(entity, dtype=cp.int64),
            cp.asarray(time_ids, dtype=cp.int64),
        )
    if backend == "torch":
        import torch
        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
            torch.as_tensor(entity, dtype=torch.int64, device="cuda"),
            torch.as_tensor(time_ids, dtype=torch.int64, device="cuda"),
        )
    raise ValueError(backend)


def _device(backend):
    return {"cupy": "cuda", "torch": "torch"}[backend]


def _gpu_name(backend):
    if backend == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch backend requested but CUDA is unavailable")
        return torch.cuda.get_device_name(0)
    raise ValueError(backend)


def _fit_case(case, X, y, entity, time_ids, clusters, backend):
    device = _device(backend)
    if case == "pooled_nonrobust":
        return PooledOLS(cov_type="nonrobust", device=device).fit(X, y)
    if case == "pooled_hc3":
        return PooledOLS(cov_type="hc3", device=device).fit(X, y)
    if case == "pooled_cluster_two_way":
        return PooledOLS(cov_type="clustered", group_debias=True, device=device).fit(
            X, y, cluster=clusters
        )
    if case == "pooled_dk_qs":
        return PooledOLS(cov_type="dk", bandwidth=2, kernel="qs", device=device).fit(
            X, y, time_index=time_ids
        )
    if case == "panel_entity_nonrobust":
        return PanelOLS(entity_effects=True, cov_type="nonrobust", device=device).fit(
            X, y, entity_ids=entity
        )
    if case == "panel_entity_hc3":
        return PanelOLS(entity_effects=True, cov_type="hc3", device=device).fit(
            X, y, entity_ids=entity
        )
    if case == "panel_entity_dk":
        return PanelOLS(
            entity_effects=True, cov_type="dk", bandwidth=2, device=device
        ).fit(X, y, entity_ids=entity, time_ids=time_ids)
    if case == "panel_entity_dk_qs":
        return PanelOLS(
            entity_effects=True, cov_type="dk", bandwidth=2, kernel="qs", device=device
        ).fit(X, y, entity_ids=entity, time_ids=time_ids)
    if case == "random_effects_nonrobust":
        return RandomEffects(device=device).fit(X, y, entity_ids=entity)
    if case == "random_effects_hc3":
        return RandomEffects(cov_type="hc3", device=device).fit(X, y, entity_ids=entity)
    raise ValueError(case)


def _timed(case, X, y, entity, time_ids, clusters, backend):
    _sync(backend)
    start = time.perf_counter()
    model = _fit_case(case, X, y, entity, time_ids, clusters, backend)
    _sync(backend)
    elapsed = time.perf_counter() - start
    executed = getattr(model, "_backend_name", None)
    if executed is None:
        raise AssertionError(f"{case}: fit did not persist executed backend provenance")
    if executed != backend:
        raise AssertionError(f"{case}: requested {backend}, executed {executed}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--scales", default="10000x2,100000x2,100000x10")
    parser.add_argument(
        "--high-t-scale",
        default=DEFAULT_HIGH_T_SCALE,
        help="additional NxKxT scenario used only for QS all-lag cases",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if _git_sha() != args.expected_sha:
        raise RuntimeError(f"wrong source head: {_git_sha()} != {args.expected_sha}")
    dirty = _git_status()
    if dirty.strip():
        raise RuntimeError("performance benchmark requires a clean tree:\n" + dirty)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    backends = [v.strip() for v in args.backends.split(",") if v.strip()]
    if not backends or any(v not in {"cupy", "torch"} for v in backends):
        raise ValueError("backends must contain cupy and/or torch")

    cases = [
        "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",
        "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",
        "random_effects_nonrobust", "random_effects_hc3",
    ]
    rows = []
    for scale_idx, (n, k) in enumerate(_parse_scales(args.scales)):
        X_np, y_np, entity_np, time_np, clusters = _dataset(
            n, k, 20260812 + scale_idx, n_times=20
        )
        for backend in backends:
            X, y, entity, time_ids = _to_backend(
                X_np, y_np, entity_np, time_np, backend
            )
            for case in cases:
                _timed(case, X, y, entity, time_ids, clusters, backend)
                samples = [
                    _timed(case, X, y, entity, time_ids, clusters, backend)
                    for _ in range(args.repeats)
                ]
                rows.append(
                    _timing_row(
                        backend=backend,
                        case=case,
                        scenario="base",
                        n=n,
                        k=k,
                        n_times=len(np.unique(time_np)),
                        repeats=args.repeats,
                        samples=samples,
                    )
                )

    high_n, high_k, high_t = _parse_high_t_scale(args.high_t_scale)
    X_np, y_np, entity_np, time_np, clusters = _dataset(
        high_n, high_k, 20260899, n_times=high_t
    )
    for backend in backends:
        X, y, entity, time_ids = _to_backend(
            X_np, y_np, entity_np, time_np, backend
        )
        for case in HIGH_T_CASES:
            _timed(case, X, y, entity, time_ids, clusters, backend)
            samples = [
                _timed(case, X, y, entity, time_ids, clusters, backend)
                for _ in range(args.repeats)
            ]
            rows.append(
                _timing_row(
                    backend=backend,
                    case=case,
                    scenario="high_t_qs",
                    n=high_n,
                    k=high_k,
                    n_times=len(np.unique(time_np)),
                    repeats=args.repeats,
                    samples=samples,
                )
            )

    payload = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "git_sha": args.expected_sha,
        "working_tree_clean": True,
        "benchmark": "panel_stage_c_covariance_fit_overhead",
        "timing_scope": "synchronized end-to-end estimator fit",
        "input_residency": (
            "X/y/entity/time preloaded on selected GPU backend; "
            "cluster labels remain CPU metadata"
        ),
        "high_t_scale": args.high_t_scale,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_by_backend": {backend: _gpu_name(backend) for backend in backends},
            "packages": {
                name: _version(name)
                for name in ("statgpu", "numpy", "cupy", "torch")
            },
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Panel Stage C performance evidence recorded: {args.out}")


if __name__ == "__main__":
    main()
