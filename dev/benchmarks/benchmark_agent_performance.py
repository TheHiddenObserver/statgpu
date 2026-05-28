"""Agent performance benchmark: verify pruning speedup and CV overhead.

Usage:
    python dev/benchmarks/benchmark_agent_performance.py
    python dev/benchmarks/benchmark_agent_performance.py --json-out results/agent_performance.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


@dataclass
class PerformanceResult:
    scenario: str
    data_config: str
    n_fits: int
    total_time_ms: float
    cv_overhead_ratio: float
    details: str


def _time_agent(X, y, cv_folds=0, device="cpu") -> tuple:
    """Time an agent run and return (time_ms, n_models, result)."""
    from statgpu.agent import StatGPUAnalysisAgent
    agent = StatGPUAnalysisAgent(device=device, cv_folds=cv_folds)
    t0 = time.perf_counter()
    result = agent.analyze(X=X, y=y)
    elapsed = (time.perf_counter() - t0) * 1000
    n_models = len([m for m in result.models if m.error is None])
    return elapsed, n_models, result


def bench_pruning_speedup(seed: int) -> PerformanceResult:
    """Compare high-dim case: pruning should skip failed OLS."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(50, 100))
    y = X[:, 0] + rng.normal(size=50) * 0.1

    # With pruning (default)
    elapsed, n_models, result = _time_agent(X, y, cv_folds=0)
    model_names = [m.name for m in result.models if m.error is None]
    ols_absent = "LinearRegression" not in model_names

    return PerformanceResult(
        scenario="pruning_speedup",
        data_config="n=50, p=100",
        n_fits=n_models,
        total_time_ms=elapsed,
        cv_overhead_ratio=0.0,
        details=f"models={model_names}, ols_pruned={ols_absent}",
    )


def bench_cv_overhead(seed: int) -> PerformanceResult:
    """Measure CV overhead ratio."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(500, 10))
    y = X[:, 0] + rng.normal(size=500) * 0.1

    # Without CV
    elapsed_no_cv, n_no_cv, _ = _time_agent(X, y, cv_folds=0)

    # With CV
    elapsed_cv, n_cv, _ = _time_agent(X, y, cv_folds=5)

    ratio = elapsed_cv / elapsed_no_cv if elapsed_no_cv > 0 else float("inf")

    return PerformanceResult(
        scenario="cv_overhead",
        data_config="n=500, p=10",
        n_fits=n_cv,
        total_time_ms=elapsed_cv,
        cv_overhead_ratio=ratio,
        details=f"no_cv={elapsed_no_cv:.1f}ms, cv={elapsed_cv:.1f}ms, ratio={ratio:.2f}x",
    )


def bench_large_scale(seed: int) -> PerformanceResult:
    """Large-scale single run."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(10000, 50))
    y = X[:, 0] + rng.normal(size=10000) * 0.1

    elapsed, n_models, _ = _time_agent(X, y, cv_folds=0)

    return PerformanceResult(
        scenario="large_scale",
        data_config="n=10000, p=50",
        n_fits=n_models,
        total_time_ms=elapsed,
        cv_overhead_ratio=0.0,
        details=f"total={elapsed:.1f}ms",
    )


def bench_small_fast(seed: int) -> PerformanceResult:
    """Small dataset, should be very fast."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] + rng.normal(size=100) * 0.1

    elapsed, n_models, _ = _time_agent(X, y, cv_folds=0)

    return PerformanceResult(
        scenario="small_fast",
        data_config="n=100, p=3",
        n_fits=n_models,
        total_time_ms=elapsed,
        cv_overhead_ratio=0.0,
        details=f"total={elapsed:.1f}ms",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent performance benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", help="JSON output path")
    args = parser.parse_args()

    benchmarks = [
        ("Pruning speedup (high-dim)", bench_pruning_speedup),
        ("CV overhead", bench_cv_overhead),
        ("Large scale", bench_large_scale),
        ("Small fast", bench_small_fast),
    ]

    all_results: List[PerformanceResult] = []

    print("=" * 80)
    print("Agent Performance Benchmark")
    print("=" * 80)

    for name, bench_fn in benchmarks:
        print(f"\n--- {name} ---")
        try:
            r = bench_fn(args.seed)
            all_results.append(r)
            print(f"  {r.data_config}: {r.total_time_ms:.1f}ms, {r.n_fits} models")
            if r.cv_overhead_ratio > 0:
                print(f"  CV overhead: {r.cv_overhead_ratio:.2f}x")
            print(f"  {r.details}")
        except Exception as e:
            all_results.append(PerformanceResult(name, "", 0, 0, 0, f"EXCEPTION: {e}"))
            print(f"  [FAIL] Exception: {e}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Summary: {len(all_results)} benchmarks completed")
    for r in all_results:
        print(f"  {r.scenario}: {r.total_time_ms:.1f}ms ({r.n_fits} fits)")
    print(f"{'=' * 80}")

    if args.json_out:
        output = {
            "config": vars(args),
            "results": [asdict(r) for r in all_results],
        }
        with open(args.json_out, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON saved to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
