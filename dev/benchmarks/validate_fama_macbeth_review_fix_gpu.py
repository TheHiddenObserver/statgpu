#!/usr/bin/env python3
"""Exact-head physical GPU validation for the PR126 Fama-MacBeth review fixes.

This runner verifies correctness/backend provenance for the chronology, formula,
rank, and no-intercept fixes and records a synchronized timing sample for the
rank-revealing retained-period solve. Timing is evidence only; no speedup claim
is derived from it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.panel import FamaMacBeth

SCHEMA_VERSION = 2


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device(backend: str) -> str:
    return {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]


def _arrays(X, y, backend: str):
    if backend == "numpy":
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(X, dtype=cp.float64), cp.asarray(y, dtype=cp.float64)
    if backend == "torch":
        import torch

        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        )
    raise ValueError(backend)


def _sync(backend: str):
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch

        torch.cuda.synchronize()


def _chronology_fixture():
    x_period = np.asarray([-2.0, -0.75, 0.25, 1.25, 2.5])
    period_params = ((0.2, 0.4), (1.1, -0.8), (-0.7, 0.3))
    x = np.tile(x_period, len(period_params))
    y = np.concatenate(
        [intercept + slope * x_period for intercept, slope in period_params]
    )
    labels = np.repeat(np.asarray(["t1", "t2", "t10"], dtype=object), x_period.size)
    numeric = np.repeat(np.arange(len(period_params)), x_period.size)
    return x[:, None], y, labels, numeric


def _ordered(labels):
    import pandas as pd

    return pd.Categorical(labels, categories=["t1", "t2", "t10"], ordered=True)


def _snapshot(model):
    return {
        "coef": np.asarray(_to_numpy(model.coef_), dtype=np.float64),
        "bse": np.asarray(_to_numpy(model.bse_), dtype=np.float64),
        "cov_params": np.asarray(_to_numpy(model.cov_params_), dtype=np.float64),
    }


def _max_abs(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _assert_snapshot(reference, actual, *, rtol=5e-6, atol=5e-7):
    diffs = {}
    for key in ("coef", "bse", "cov_params"):
        np.testing.assert_allclose(actual[key], reference[key], rtol=rtol, atol=atol)
        diffs[key] = _max_abs(actual[key], reference[key])
    return diffs


def _chronology_case(backend: str):
    X, y, labels, numeric = _chronology_fixture()
    ordered = _ordered(labels)
    ref = FamaMacBeth(bandwidth=1, device="cpu").fit(X, y, time_ids=numeric)
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(bandwidth=1, device=_device(backend)).fit(
        Xb, yb, time_ids=ordered
    )
    lexical = FamaMacBeth(bandwidth=1, device="cpu").fit(
        X, y, time_ids=np.asarray(ordered, dtype=object)
    )
    if np.allclose(
        np.asarray(_to_numpy(actual.cov_params_)),
        np.asarray(_to_numpy(lexical.cov_params_)),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError("chronology negative control lost power")
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "max_abs_differences": _assert_snapshot(_snapshot(ref), _snapshot(actual)),
    }


def _formula_case(backend: str):
    import pandas as pd

    X, y, labels, numeric = _chronology_fixture()
    ordered = _ordered(labels)
    x = X[:, 0].copy()
    x[1] = np.nan
    data = pd.DataFrame({"y": y, "x": x})
    ref = FamaMacBeth(bandwidth=1, device="cpu").fit(
        formula="y ~ x", data=data, time_ids=numeric
    )
    actual = FamaMacBeth(bandwidth=1, device=_device(backend)).fit(
        formula="y ~ x", data=data, time_ids=ordered
    )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "max_abs_differences": _assert_snapshot(_snapshot(ref), _snapshot(actual)),
    }


def _rank_fixture():
    x = np.concatenate(
        [
            np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.ones(5),
            np.asarray([-1.5, -0.5, 0.5, 1.5, 2.5]),
        ]
    )
    time_ids = np.repeat(np.arange(3), 5)
    y = 0.5 + 0.8 * x + np.repeat(np.asarray([0.0, 0.4, -0.3]), 5)
    return x[:, None], y, time_ids


def _rank_rejection(backend: str):
    X, y, time_ids = _rank_fixture()
    Xb, yb = _arrays(X, y, backend)
    try:
        FamaMacBeth(device=_device(backend)).fit(Xb, yb, time_ids=time_ids)
    except ValueError as exc:
        if "rank deficient" not in str(exc):
            raise
        return True
    raise AssertionError("rank-deficient retained period was not rejected")


def _no_intercept_rejections(backend: str):
    import pandas as pd

    X, y, _labels, numeric = _chronology_fixture()
    data = pd.DataFrame({"y": y, "x": X[:, 0]})
    result = {}
    for formula in ("y ~ 0 + x", "y ~ x - 1"):
        try:
            FamaMacBeth(device=_device(backend)).fit(
                formula=formula, data=data, time_ids=numeric
            )
        except ValueError as exc:
            if "no-intercept formulas are not supported" not in str(exc):
                raise
            result[formula] = True
        else:
            raise AssertionError(f"no-intercept formula was accepted: {formula}")
    return result


def _timing_fixture():
    rng = np.random.default_rng(20260816)
    n_times, per_period, p = 64, 128, 4
    time_ids = np.repeat(np.arange(n_times), per_period)
    X = rng.normal(size=(n_times * per_period, p))
    beta = np.asarray([0.7, -0.4, 0.25, 0.9])
    period_shift = np.repeat(rng.normal(scale=0.3, size=n_times), per_period)
    y = 0.5 + X @ beta + period_shift + rng.normal(scale=0.4, size=X.shape[0])
    return X.astype(np.float64), y.astype(np.float64), time_ids


def _timing_case(backend: str, warmup: int, repeats: int):
    X, y, time_ids = _timing_fixture()
    reference = FamaMacBeth(bandwidth=2, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    Xb, yb = _arrays(X, y, backend)
    device = _device(backend)
    for _ in range(warmup):
        FamaMacBeth(bandwidth=2, device=device).fit(Xb, yb, time_ids=time_ids)
        _sync(backend)
    samples = []
    last = None
    for _ in range(repeats):
        _sync(backend)
        start = time.perf_counter()
        last = FamaMacBeth(bandwidth=2, device=device).fit(
            Xb, yb, time_ids=time_ids
        )
        _sync(backend)
        samples.append(time.perf_counter() - start)
    if last is None or last._backend_name != backend:
        raise AssertionError(f"requested {backend}, executed {getattr(last, '_backend_name', None)}")
    numerical_differences = _assert_snapshot(
        _snapshot(reference), _snapshot(last)
    )
    return {
        "status": "success",
        "executed_backend": last._backend_name,
        "n_times": 64,
        "observations_per_period": 128,
        "n_features": 4,
        "warmup": warmup,
        "repeats": repeats,
        "samples_seconds": samples,
        "median_seconds": float(statistics.median(samples)),
        "max_abs_differences_vs_numpy": numerical_differences,
    }


def _environment(backends):
    gpu_by_backend = {}
    if "cupy" in backends:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        gpu_by_backend["cupy"] = name.decode() if isinstance(name, bytes) else name
    if "torch" in backends:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable")
        gpu_by_backend["torch"] = torch.cuda.get_device_name(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "patsy": _version("patsy"),
        "cupy": _version("cupy"),
        "torch": _version("torch"),
        "gpu_by_backend": gpu_by_backend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    if not _git_clean():
        raise RuntimeError("physical acceptance requires a clean working tree")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if not backends or any(value not in {"cupy", "torch"} for value in backends):
        raise ValueError("--backends must contain cupy and/or torch")

    results = {}
    for backend in backends:
        chronology = _chronology_case(backend)
        formula = _formula_case(backend)
        if chronology["executed_backend"] != backend or formula["executed_backend"] != backend:
            raise AssertionError(f"{backend}: backend provenance mismatch")
        results[backend] = {
            "status": "success",
            "executed_backend": backend,
            "array_ordered_categorical": chronology,
            "formula_ordered_categorical_alignment": formula,
            "rank_deficient_retained_period_rejected": _rank_rejection(backend),
            "no_intercept_formula_rejections": _no_intercept_rejections(backend),
            "performance": _timing_case(backend, args.warmup, args.repeats),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": sha,
        "working_tree_clean": True,
        "status": "success",
        "environment": _environment(backends),
        "timing_claim": "raw synchronized timing only; no speedup claim",
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — focused Fama-MacBeth GPU validation: {args.out}")


if __name__ == "__main__":
    main()
