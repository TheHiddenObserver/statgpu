#!/usr/bin/env python3
"""Measure Fama-MacBeth NumPy/GPU crossover across panel scales.

This is performance evidence, not a correctness acceptance replacement. The
historical 64x128x4 fixture is retained as the micro/launch-overhead regime and
larger balanced panels show whether backend-specific period scheduling reaches
a useful GPU crossover on resident device arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.benchmarks.validate_fama_macbeth_review_fix_gpu import (
    _assert_inference_descriptors,
    _assert_snapshot,
    _environment,
    _inference_descriptor,
    _snapshot,
    _timed_fit,
)


FIXTURES = {
    "micro": {"n_times": 64, "observations_per_period": 128, "n_features": 4},
    "medium": {"n_times": 128, "observations_per_period": 1024, "n_features": 8},
    "large": {"n_times": 128, "observations_per_period": 4096, "n_features": 16},
}
_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean():
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _fixture(name, spec):
    seed = 20260817 + list(FIXTURES).index(name)
    rng = np.random.default_rng(seed)
    n_times = int(spec["n_times"])
    per_period = int(spec["observations_per_period"])
    p = int(spec["n_features"])
    time_ids = np.repeat(np.arange(n_times), per_period)
    X = rng.normal(size=(n_times * per_period, p))
    beta = rng.normal(scale=0.45, size=p)
    period_shift = np.repeat(rng.normal(scale=0.3, size=n_times), per_period)
    y = 0.5 + X @ beta + period_shift + rng.normal(scale=0.4, size=X.shape[0])
    return X.astype(np.float64), y.astype(np.float64), time_ids


def _timing_summary(samples, n_rows):
    median = float(statistics.median(samples))
    return {
        "samples_seconds": [float(value) for value in samples],
        "median_seconds": median,
        "rows_per_second": float(n_rows / median),
    }


def _solver_provenance(model):
    return {
        "solver_mode": model._period_solver_mode,
        "solver_batches": int(model._period_solver_batches),
        "control_syncs": int(model._period_rank_syncs),
        "svd_fallbacks": int(getattr(model, "_period_svd_fallbacks", 0)),
    }


def _case(name, spec, backends, warmup, repeats):
    X, y, time_ids = _fixture(name, spec)
    n_rows = int(X.shape[0])
    reference, numpy_samples = _timed_fit(
        X, y, time_ids, "numpy", warmup=warmup, repeats=repeats
    )
    numpy_summary = _timing_summary(numpy_samples, n_rows)
    results = {
        "shape": {
            **spec,
            "n_rows": n_rows,
            "design_columns_with_intercept": int(X.shape[1] + 1),
        },
        "numpy": {
            **numpy_summary,
            **_solver_provenance(reference),
        },
        "backends": {},
    }

    prediction_X = X[:16]
    reference_snapshot = _snapshot(reference, prediction_X)
    reference_inference = _inference_descriptor(reference)
    for backend in backends:
        candidate, samples = _timed_fit(
            X, y, time_ids, backend, warmup=warmup, repeats=repeats
        )
        _assert_inference_descriptors(
            reference_inference,
            _inference_descriptor(candidate),
        )
        diffs = _assert_snapshot(
            reference_snapshot,
            _snapshot(candidate, prediction_X),
        )
        summary = _timing_summary(samples, n_rows)
        results["backends"][backend] = {
            **summary,
            "backend_over_numpy_median_ratio": float(
                summary["median_seconds"] / numpy_summary["median_seconds"]
            ),
            "speedup_over_numpy": float(
                numpy_summary["median_seconds"] / summary["median_seconds"]
            ),
            **_solver_provenance(candidate),
            "max_abs_differences_vs_numpy": diffs,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--validation-tier",
        required=True,
        choices=("local-minimal", "local-full", "remote-full"),
        help=(
            "evidence tier supplied by the runner orchestrator; the script never "
            "infers remote execution so local runs cannot silently claim remote-full"
        ),
    )
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    if not _git_clean():
        raise RuntimeError("scaling benchmark requires a clean working tree")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if not backends or any(value not in {"cupy", "torch"} for value in backends):
        raise ValueError("backends must be a non-empty subset of cupy,torch")

    cases = {
        name: _case(name, spec, backends, args.warmup, args.repeats)
        for name, spec in FIXTURES.items()
    }
    payload = {
        "schema_version": 2,
        "git_sha": sha,
        "validation_tier": args.validation_tier,
        "status": "success",
        "environment": _environment(backends),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENV_NAMES
        },
        "timing_protocol": {
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "input_residency": (
                "NumPy arrays resident on host; CuPy/Torch arrays transferred to GPU "
                "before warmup/timed samples"
            ),
            "synchronization": "GPU synchronized immediately before and after every timed fit",
        },
        "timing_scope": (
            "resident-array end-to-end FamaMacBeth.fit timing with explicit GPU "
            "synchronization before and after each sample; host-to-device input transfer "
            "is intentionally excluded"
        ),
        "interpretation": (
            "backend_over_numpy_median_ratio < 1 (speedup_over_numpy > 1) means "
            "the GPU backend is faster than the serial NumPy reference on that scale"
        ),
        "solver_interpretation": (
            "GPU solver_mode=gram-certified means every retained period passed the "
            "conservative Gram-spectrum certificate and svd_fallbacks=0. control_syncs "
            "counts the host control transfers needed to choose the fail-closed path; "
            "uncertified periods must fall back to the maintained SVD rank policy."
        ),
        "fixtures": cases,
    }
    if not _git_clean():
        raise RuntimeError("working tree changed during scaling benchmark")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Fama-MacBeth scaling benchmark: {args.out}")


if __name__ == "__main__":
    main()
