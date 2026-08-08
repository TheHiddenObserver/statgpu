#!/usr/bin/env python3
"""Measure Stage-B full-content identity overhead on physical GPU backends.

This benchmark isolates the cost of the collision-resistant X/y SHA-256 used by
Hausman sample identity. It compares ordinary PanelOLS/RandomEffects fit time
against the same fit with only ``_full_content_digest`` replaced by a constant
stub. Numerical work, low-order audit reductions, estimator setup, and all other
Stage-B code remain unchanged in the baseline.

The script is intentionally separate from ``validate_panel_stage_b_gpu.py``:
that runner remains correctness/provenance-only and its frontend source must not
acquire inferred timing or speedup fields.
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

import statgpu.panel._diagnostics as diagnostics
from statgpu.panel import PanelOLS, RandomEffects


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_scales(value: str):
    out = []
    for item in value.split(","):
        n_text, k_text = item.strip().lower().split("x", 1)
        n, k = int(n_text), int(k_text)
        if n <= 0 or k <= 0:
            raise ValueError("benchmark scales must be positive NxK pairs")
        out.append((n, k))
    return out


def _sync(backend: str):
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch

        torch.cuda.synchronize()


def _to_backend(X, y, entity, backend: str):
    if backend == "cupy":
        import cupy as cp

        return (
            cp.asarray(X),
            cp.asarray(y),
            cp.asarray(entity, dtype=cp.int64),
        )
    if backend == "torch":
        import torch

        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
            torch.as_tensor(entity, dtype=torch.int64, device="cuda"),
        )
    raise ValueError(backend)


def _device_arg(backend: str):
    return {"cupy": "cuda", "torch": "torch"}[backend]


def _dataset(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k)).astype(np.float64)
    beta = np.linspace(0.2, 0.8, k, dtype=np.float64)
    entity = np.arange(n, dtype=np.int64) // 20
    n_entities = int(entity.max()) + 1
    alpha = np.linspace(-0.5, 0.5, n_entities, dtype=np.float64)[entity]
    y = X @ beta + alpha + rng.normal(scale=0.2, size=n)
    return X, y.astype(np.float64), entity


def _fit(model_name: str, X, y, entity, backend: str):
    device = _device_arg(backend)
    if model_name == "PanelOLS":
        model = PanelOLS(entity_effects=True, cov_type="nonrobust", device=device)
    elif model_name == "RandomEffects":
        model = RandomEffects(device=device)
    else:
        raise ValueError(model_name)
    model.fit(X, y, entity_ids=entity)
    return model


def _timed_fit(model_name, X, y, entity, backend, *, disable_digest: bool):
    original = diagnostics._full_content_digest
    if disable_digest:
        diagnostics._full_content_digest = lambda _X, _y: "0" * 64
    try:
        _sync(backend)
        start = time.perf_counter()
        _fit(model_name, X, y, entity, backend)
        _sync(backend)
        return time.perf_counter() - start
    finally:
        diagnostics._full_content_digest = original


def _median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument(
        "--scales",
        default="10000x2,100000x2,100000x10,500000x2",
        help="comma-separated NxK pairs",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    dirty = _git_status_porcelain()
    if dirty.strip():
        raise RuntimeError(
            "identity benchmark requires a clean working tree; uncommitted changes:\n"
            + dirty
        )
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    if not backends or any(x not in {"cupy", "torch"} for x in backends):
        raise ValueError("--backends must contain cupy and/or torch")
    scales = _parse_scales(args.scales)

    rows = []
    for scale_index, (n, k) in enumerate(scales):
        X_np, y_np, entity_np = _dataset(n, k, seed=20260808 + scale_index)
        for backend in backends:
            X, y, entity = _to_backend(X_np, y_np, entity_np, backend)
            for model_name in ("PanelOLS", "RandomEffects"):
                # Warm both paths before measurement to avoid one-time import/
                # allocator effects being attributed to the digest.
                _timed_fit(
                    model_name, X, y, entity, backend, disable_digest=False
                )
                _timed_fit(
                    model_name, X, y, entity, backend, disable_digest=True
                )

                with_digest = []
                without_digest = []
                for _ in range(args.repeats):
                    with_digest.append(
                        _timed_fit(
                            model_name,
                            X,
                            y,
                            entity,
                            backend,
                            disable_digest=False,
                        )
                    )
                    without_digest.append(
                        _timed_fit(
                            model_name,
                            X,
                            y,
                            entity,
                            backend,
                            disable_digest=True,
                        )
                    )

                normal = _median(with_digest)
                baseline = _median(without_digest)
                overhead = normal - baseline
                ratio = normal / baseline if baseline > 0.0 else None
                rows.append(
                    {
                        "backend": backend,
                        "model": model_name,
                        "n_samples": n,
                        "n_features": k,
                        "repeats": args.repeats,
                        "with_digest_seconds": normal,
                        "without_digest_seconds": baseline,
                        "digest_overhead_seconds": overhead,
                        "with_over_without_ratio": ratio,
                        "with_digest_samples": with_digest,
                        "without_digest_samples": without_digest,
                    }
                )

    payload = {
        "schema_version": 1,
        "git_sha": sha,
        "working_tree_clean": True,
        "benchmark": "panel_stage_b_full_content_identity_overhead",
        "timing_scope": "end-to-end estimator fit with vs without only the SHA-256 full-content digest",
        "target_scale_source": "PR122 fresh-review performance finding",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
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
    print(f"PASS — identity-overhead benchmark recorded: {args.out}")


if __name__ == "__main__":
    main()
