#!/usr/bin/env python3
"""Physical GPU gate for PooledOLS legacy-HAC chronology semantics.

The legacy HAC covariance treats rows as one ordered sequence.  Ordered pandas
categoricals must therefore use their declared category chronology rather than
lexical label order.  This runner validates that contract on both CuPy and Torch
CUDA, including formula missing-row alignment and a lexical negative control.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from statgpu.panel import PooledOLS

SCHEMA_VERSION = 1
_REQUIRED_BACKENDS = {"cupy", "torch"}


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _device(backend: str) -> str:
    return {"cupy": "cuda", "torch": "torch"}[backend]


def _arrays(X, y, backend: str):
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


def _fixture(seed=12698):
    rng = np.random.default_rng(seed)
    n_entities = 18
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), n_entities)
    numeric = np.tile(np.arange(3), n_entities)
    ordered = pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    x = rng.normal(size=labels.size)
    time_shock = np.tile(np.array([0.0, 1.5, -1.0]), n_entities)
    y = 0.4 + 0.8 * x + time_shock + rng.normal(scale=0.1, size=labels.size)
    return x[:, None], y, ordered, numeric


def _snapshot(model):
    return {
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "bse": np.asarray(model.bse_, dtype=np.float64),
        "tvalues": np.asarray(model.tvalues_, dtype=np.float64),
        "pvalues": np.asarray(model.pvalues_, dtype=np.float64),
        "conf_int": np.asarray(model.conf_int_, dtype=np.float64),
    }


def _max_abs(left, right):
    result = {}
    for key in left:
        diff = np.max(np.abs(left[key] - right[key]))
        result[key] = float(diff)
    return result


def _assert_close(left, right):
    for key in left:
        np.testing.assert_allclose(left[key], right[key], rtol=5e-6, atol=5e-7)


def _run_backend(backend: str):
    X_np, y_np, ordered, numeric = _fixture()
    X, y = _arrays(X_np, y_np, backend)
    device = _device(backend)

    ordered_fit = PooledOLS(
        cov_type="hac", bandwidth=4, device=device
    ).fit(X, y, time_index=ordered)
    numeric_fit = PooledOLS(
        cov_type="hac", bandwidth=4, device=device
    ).fit(X, y, time_index=numeric)
    lexical_fit = PooledOLS(
        cov_type="hac", bandwidth=4, device=device
    ).fit(X, y, time_index=np.asarray(ordered, dtype=object))

    for model in (ordered_fit, numeric_fit, lexical_fit):
        if model._backend_name != backend:
            raise AssertionError(
                f"requested {backend} but estimator executed {model._backend_name}"
            )

    ordered_snapshot = _snapshot(ordered_fit)
    numeric_snapshot = _snapshot(numeric_fit)
    lexical_snapshot = _snapshot(lexical_fit)
    _assert_close(ordered_snapshot, numeric_snapshot)
    lexical_bse_gap = float(
        np.max(np.abs(ordered_snapshot["bse"] - lexical_snapshot["bse"]))
    )
    if lexical_bse_gap <= 1e-10:
        raise AssertionError("lexical chronology negative control did not separate")

    x_gap = X_np[:, 0].copy()
    x_gap[[5, 31]] = np.nan
    data = pd.DataFrame({"y": y_np, "x": x_gap})
    ordered_formula = PooledOLS(
        cov_type="hac", bandwidth=3, device=device
    ).fit(formula="y ~ x", data=data, time_index=ordered)
    numeric_formula = PooledOLS(
        cov_type="hac", bandwidth=3, device=device
    ).fit(formula="y ~ x", data=data, time_index=numeric)
    if ordered_formula._backend_name != backend or numeric_formula._backend_name != backend:
        raise AssertionError(f"{backend}: formula backend provenance mismatch")
    ordered_formula_snapshot = _snapshot(ordered_formula)
    numeric_formula_snapshot = _snapshot(numeric_formula)
    _assert_close(ordered_formula_snapshot, numeric_formula_snapshot)

    return {
        "status": "success",
        "requested_backend": backend,
        "executed_backend": ordered_fit._backend_name,
        "array_ordered_vs_numeric_max_abs": _max_abs(
            ordered_snapshot, numeric_snapshot
        ),
        "formula_ordered_vs_numeric_max_abs": _max_abs(
            ordered_formula_snapshot, numeric_formula_snapshot
        ),
        "lexical_negative_control_bse_max_abs": lexical_bse_gap,
    }


def _environment(backends):
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpu_by_backend": {},
    }
    if "cupy" in backends:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy CUDA is unavailable")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        payload["cupy"] = cp.__version__
        payload["gpu_by_backend"]["cupy"] = (
            name.decode() if isinstance(name, bytes) else name
        )
    if "torch" in backends:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable")
        payload["torch"] = torch.__version__
        payload["gpu_by_backend"]["torch"] = torch.cuda.get_device_name(0)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    if not _git_clean():
        raise RuntimeError("physical acceptance requires a clean working tree")

    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if len(backends) != 2 or set(backends) != _REQUIRED_BACKENDS:
        raise ValueError("physical acceptance requires exactly cupy,torch")

    results = {backend: _run_backend(backend) for backend in backends}
    clean_after = _git_clean()
    if not clean_after:
        raise RuntimeError("working tree changed during physical validation")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": sha,
        "working_tree_clean_before": True,
        "working_tree_clean_after_checks": clean_after,
        "required_backends": sorted(_REQUIRED_BACKENDS),
        "validated_backends": backends,
        "status": "success",
        "validation_tier": "remote-full",
        "environment": _environment(backends),
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Pooled HAC chronology GPU validation: {args.out}")


if __name__ == "__main__":
    main()
