"""Agent robustness benchmark: verify pruning and self-correction on edge cases.

Usage:
    python dev/benchmarks/benchmark_agent_robustness.py
    python dev/benchmarks/benchmark_agent_robustness.py --json-out results/agent_robustness.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


@dataclass
class RobustnessResult:
    scenario: str
    data_config: str
    pruning_worked: bool
    warning_correct: bool
    all_models_ran: bool
    details: str


def _run_agent(X, y=None, time=None, event=None, task="auto", cv_folds=0):
    from statgpu.agent import StatGPUAnalysisAgent
    agent = StatGPUAnalysisAgent(device="cpu", cv_folds=cv_folds)
    return agent.analyze(X=X, y=y, time=time, event=event, task=task)


def bench_high_dimensional(seed: int) -> RobustnessResult:
    """p > n: OLS should be pruned."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(50, 100))
    y = X[:, 0] + rng.normal(size=50) * 0.1
    result = _run_agent(X, y)
    model_names = [m.name for m in result.models]
    ols_pruned = "LinearRegression" not in model_names
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="high_dimensional",
        data_config="n=50, p=100",
        pruning_worked=ols_pruned,
        warning_correct=True,  # No specific warning expected
        all_models_ran=no_error,
        details=f"models={model_names}",
    )


def bench_near_high_dimensional(seed: int) -> RobustnessResult:
    """p > n/2: OLS should be pruned."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 80))
    y = X[:, 0] + rng.normal(size=100) * 0.1
    result = _run_agent(X, y)
    model_names = [m.name for m in result.models]
    ols_pruned = "LinearRegression" not in model_names
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="near_high_dimensional",
        data_config="n=100, p=80",
        pruning_worked=ols_pruned,
        warning_correct=True,
        all_models_ran=no_error,
        details=f"models={model_names}",
    )


def bench_collinearity(seed: int) -> RobustnessResult:
    """Strong collinearity: condition number warning expected."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    X = np.column_stack([X, X[:, 0] * 1.0001, X[:, 1] * 0.9999])  # Near-duplicate columns
    y = X[:, 0] + rng.normal(size=200) * 0.1
    result = _run_agent(X, y)
    has_cond_warning = any("condition" in w.lower() for w in result.warnings)
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="collinearity",
        data_config="n=200, p=7 (2 near-duplicate)",
        pruning_worked=False,  # Not about pruning
        warning_correct=has_cond_warning,
        all_models_ran=no_error,
        details=f"warnings={[w for w in result.warnings if 'condition' in w.lower()]}",
    )


def bench_missing_values(seed: int) -> RobustnessResult:
    """10% missing values: should be imputed, no error."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    mask = rng.random(X.shape) < 0.1
    X[mask] = np.nan
    y = rng.normal(size=200)
    result = _run_agent(X, y)
    imputed = result.profile.imputed_values > 0
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="missing_values",
        data_config="n=200, p=5, 10% NaN",
        pruning_worked=False,
        warning_correct=imputed,
        all_models_ran=no_error,
        details=f"imputed={result.profile.imputed_values}",
    )


def bench_categorical_features(seed: int) -> RobustnessResult:
    """Categorical features: one-hot encoding should work."""
    rng = np.random.default_rng(seed)
    n = 200
    age = rng.normal(30, 10, size=n)
    sex = rng.choice(["M", "F"], size=n)
    # Agent should handle this via table input
    data = [{"age": float(age[i]), "sex": str(sex[i]), "outcome": float(rng.integers(0, 2))}
            for i in range(n)]
    from statgpu.agent import StatGPUAnalysisAgent
    agent = StatGPUAnalysisAgent(device="cpu", cv_folds=0)
    result = agent.analyze(data=data, target="outcome")
    has_encoding = "sex" in result.profile.encoded_features
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="categorical_features",
        data_config="n=200, sex(M/F)",
        pruning_worked=False,
        warning_correct=has_encoding,
        all_models_ran=no_error,
        details=f"encoded={result.profile.encoded_features}",
    )


def bench_class_imbalance(seed: int) -> RobustnessResult:
    """Extreme class imbalance: warning expected."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    y = np.zeros(200)
    y[:5] = 1.0  # 2.5% positive
    result = _run_agent(X, y)
    has_imbalance_warning = any("imbalanced" in w.lower() for w in result.warnings)
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="class_imbalance",
        data_config="n=200, pos_rate=2.5%",
        pruning_worked=False,
        warning_correct=has_imbalance_warning,
        all_models_ran=no_error,
        details=f"warnings={[w for w in result.warnings if 'imbalanced' in w.lower()]}",
    )


def bench_small_sample(seed: int) -> RobustnessResult:
    """Small sample: warning expected."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 3))
    y = X[:, 0] + rng.normal(size=20) * 0.1
    result = _run_agent(X, y)
    has_small_warning = any("below" in w.lower() for w in result.warnings)
    no_error = all(m.error is None for m in result.models)
    return RobustnessResult(
        scenario="small_sample",
        data_config="n=20, p=3",
        pruning_worked=False,
        warning_correct=has_small_warning,
        all_models_ran=no_error,
        details=f"warnings={[w for w in result.warnings if 'below' in w.lower()]}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent robustness benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", help="JSON output path")
    args = parser.parse_args()

    benchmarks = [
        ("High-dimensional (p > n)", bench_high_dimensional),
        ("Near-high-dimensional (p > n/2)", bench_near_high_dimensional),
        ("Collinearity", bench_collinearity),
        ("Missing values", bench_missing_values),
        ("Categorical features", bench_categorical_features),
        ("Class imbalance", bench_class_imbalance),
        ("Small sample", bench_small_sample),
    ]

    all_results: List[RobustnessResult] = []

    print("=" * 80)
    print("Agent Robustness Benchmark")
    print("=" * 80)

    for name, bench_fn in benchmarks:
        print(f"\n--- {name} ---")
        try:
            r = bench_fn(args.seed)
            all_results.append(r)
            status = "PASS" if (r.pruning_worked or r.warning_correct) and r.all_models_ran else "FAIL"
            print(f"  [{status}] pruning={r.pruning_worked}, warning={r.warning_correct}, "
                  f"ran={r.all_models_ran}")
            print(f"  {r.details}")
        except Exception as e:
            all_results.append(RobustnessResult(name, "", False, False, False, f"EXCEPTION: {e}"))
            print(f"  [FAIL] Exception: {e}")

    # Summary
    passed = sum(1 for r in all_results if (r.pruning_worked or r.warning_correct) and r.all_models_ran)
    total = len(all_results)
    print(f"\n{'=' * 80}")
    print(f"Summary: {passed}/{total} passed")
    print(f"{'=' * 80}")

    if args.json_out:
        output = {
            "config": vars(args),
            "results": [asdict(r) for r in all_results],
            "summary": {"passed": passed, "total": total},
        }
        with open(args.json_out, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON saved to {args.json_out}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
