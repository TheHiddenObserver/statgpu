#!/usr/bin/env python3
"""Optimized-source physical GPU gate for Fama-MacBeth.

The long-standing focused PR126 runner remains the correctness oracle for its
chronology/formula/rank/inference matrix. This wrapper executes that runner on
the same exact head, then adds machine-auditable solver provenance for the
condition-certified Gram fast path and its SVD fallback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.benchmarks import validate_fama_macbeth_review_fix_gpu as focused

SCHEMA_VERSION = 5


def _run_focused(temp_out: Path, *, expected_sha: str, backends: str, warmup: int, repeats: int):
    runner = Path(focused.__file__).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--out",
            str(temp_out),
            "--expected-sha",
            expected_sha,
            "--backends",
            backends,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "focused Fama-MacBeth physical runner failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _solver_provenance(backend: str):
    X, y, time_ids = focused._timing_fixture()
    model, _samples = focused._timed_fit(
        X,
        y,
        time_ids,
        backend,
        warmup=0,
        repeats=1,
    )
    return {
        "solver_mode": getattr(model, "_period_solver_mode", None),
        "solver_batches": int(getattr(model, "_period_solver_batches", -1)),
        "control_syncs": int(getattr(model, "_period_rank_syncs", -1)),
        "svd_fallbacks": int(getattr(model, "_period_svd_fallbacks", -1)),
        "n_periods": int(model.n_periods),
    }


def _expected_provenance(backend: str):
    if backend == "numpy":
        return {
            "solver_mode": "serial",
            "solver_batches": 64,
            "control_syncs": 64,
            "svd_fallbacks": 0,
            "n_periods": 64,
        }
    if backend in {"cupy", "torch"}:
        return {
            "solver_mode": "gram-certified",
            "solver_batches": 1,
            "control_syncs": 1,
            "svd_fallbacks": 0,
            "n_periods": 64,
        }
    raise ValueError(f"unsupported backend provenance request: {backend}")


def _rewrite_performance(payload, backends):
    numpy_solver = _solver_provenance("numpy")
    if numpy_solver != _expected_provenance("numpy"):
        raise AssertionError(f"unexpected NumPy timing solver provenance: {numpy_solver}")

    for backend in backends:
        provenance = _solver_provenance(backend)
        expected = _expected_provenance(backend)
        if provenance != expected:
            raise AssertionError(
                f"{backend}: unexpected optimized timing solver provenance: "
                f"{provenance} != {expected}"
            )
        performance = payload["backends"][backend]["performance"]
        performance["solver_provenance"] = {
            "numpy": numpy_solver,
            backend: provenance,
        }
        performance["optimization_notes"] = {
            "period_solver": (
                "GPU periods are grouped by exact row count and first use a batched "
                "Gram-spectrum certificate. Clearly well-conditioned designs with "
                "lambda_min(X'X)/lambda_max(X'X) > 1e-4 use a batched Gram solve; "
                "uncertified periods are not allowed to consume that candidate."
            ),
            "rank_cutoff": (
                "The condition certificate does not replace the maintained SVD rank "
                "definition. Every uncertified period falls back to the original "
                "max(n_t,k)*eps*s_max SVD policy; Torch may stack the unsafe subset, "
                "while CuPy retains supported 2-D SVD fallback solves."
            ),
            "control_synchronization": (
                "control_syncs counts host synchronizations needed to select a fail-closed "
                "path. On an all-certified bucket this is the certificate-mask transfer; "
                "an SVD fallback adds a rank-vector transfer."
            ),
            "distribution_inference": (
                "p-values and critical values use the selected NumPy/CuPy/Torch "
                "inference backend directly; GPU fits do not round-trip the statistic "
                "vector through NumPy/SciPy for distribution evaluation"
            ),
            "reporting_snapshot": (
                "Fama-MacBeth packs coefficient/BSE/statistic/p-value/CI reporting fields "
                "on the active backend and performs one small NumPy reporting snapshot "
                "after numerical inference"
            ),
            "input_validation": (
                "direct X/y fits reuse the BaseEstimator public finite-input guard and "
                "do not repeat a second full device scan inside FamaMacBeth; formula "
                "arrays retain the post-Patsy internal finite check"
            ),
            "remaining_structure": (
                "the certificate mask still crosses the device boundary once per exact-size "
                "bucket for fail-closed fallback selection, and the full fit still includes "
                "covariance, inference, reporting and fit statistics; use the scaling runner "
                "for workload crossover evidence"
            ),
            "interpretation": (
                "ratio > 1 means the requested backend is slower than NumPy on the micro "
                "fixture; no universal GPU speedup claim is made"
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if focused._git_sha() != args.expected_sha:
        raise RuntimeError(
            f"wrong source head: {focused._git_sha()} != {args.expected_sha}"
        )
    if not focused._git_clean():
        raise RuntimeError("physical acceptance requires a clean working tree")
    backends = focused._validate_acceptance_backends(args.backends.split(","))

    with tempfile.TemporaryDirectory(prefix="statgpu_fmb_optimized_") as temp_dir:
        temp_out = Path(temp_dir) / "focused.json"
        _run_focused(
            temp_out,
            expected_sha=args.expected_sha,
            backends=",".join(backends),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        payload = json.loads(temp_out.read_text(encoding="utf-8"))

    _rewrite_performance(payload, backends)
    payload["schema_version"] = SCHEMA_VERSION
    payload["optimized_wrapper"] = True
    payload["focused_runner_schema_version"] = focused.SCHEMA_VERSION
    payload["performance_interpretation"] = (
        "micro timing remains comparable with historical PR126 focused evidence; "
        "scaling/crossover conclusions must use benchmark_fama_macbeth_scaling_gpu.py"
    )

    if not focused._git_clean():
        raise RuntimeError("working tree changed during optimized physical validation")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — optimized Fama-MacBeth GPU validation: {args.out}")


if __name__ == "__main__":
    main()
