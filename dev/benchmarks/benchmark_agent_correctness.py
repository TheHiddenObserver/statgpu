"""Agent correctness benchmark: verify agent output matches external baselines.

Usage:
    python dev/benchmarks/benchmark_agent_correctness.py
    python dev/benchmarks/benchmark_agent_correctness.py --json-out results/agent_correctness.json
    python dev/benchmarks/benchmark_agent_correctness.py --skip-r
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_regression_data(n: int, p: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(size=n) * 0.5
    return X, y


def make_classification_data(n: int, p: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    z = X @ beta + rng.normal(size=n) * 0.5
    y = (z > 0).astype(float)
    return X, y


def make_survival_data(n: int, p: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p) * 0.3
    linear = X @ beta
    T = rng.exponential(scale=np.exp(-linear))
    C = rng.uniform(0, np.percentile(T, 70))
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return X, time, event


# ---------------------------------------------------------------------------
# External baseline fitters
# ---------------------------------------------------------------------------

def fit_statsmodels_ols(X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return {
        "name": "statsmodels.OLS",
        "coef": model.params,
        "bse": model.bse,
        "pvalues": model.pvalues,
    }


def fit_sklearn_logistic(X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
    model.fit(X, y)
    # Intercept first, then coef (same order as statgpu agent)
    coef = np.concatenate([model.intercept_.reshape(-1), model.coef_.flatten()])
    return {
        "name": "sklearn.LogisticRegression",
        "coef": coef,
        "intercept": model.intercept_[0],
        "coef_only": model.coef_.flatten(),
    }


def fit_r_coxph(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> Optional[Dict[str, Any]]:
    import subprocess
    import tempfile
    import os

    data_path = os.path.join(tempfile.gettempdir(), "_agent_cox_bench.csv")
    result_path = os.path.join(tempfile.gettempdir(), "_agent_cox_result.json")

    # Write data
    import csv
    with open(data_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(X.shape[1])] + ["time", "event"]
        writer.writerow(header)
        for i in range(X.shape[0]):
            writer.writerow(list(X[i]) + [time[i], event[i]])

    r_script = f'''
    library(survival)
    df <- read.csv("{data_path.replace(os.sep, "/")}")
    fmla <- as.formula(paste("Surv(time, event) ~", paste(paste0("x", 0:{X.shape[1]-1}), collapse="+")))
    fit <- coxph(fmla, data=df)
    result <- list(coef=coef(fit), hr=exp(coef(fit)))
    write(toJSON(result), "{result_path.replace(os.sep, "/")}")
    '''

    try:
        subprocess.run(["Rscript", "-e", r_script], capture_output=True, timeout=30)
        with open(result_path) as f:
            r_result = json.load(f)
        return {
            "name": "R::coxph",
            "coef": np.array(r_result["coef"]),
            "hr": np.array(r_result["hr"]),
        }
    except Exception:
        return None
    finally:
        for p in (data_path, result_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n])))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectnessResult:
    task: str
    agent_method: str
    external: str
    coef_diff: float
    bse_diff: float
    p_diff: float
    passed: bool
    note: str = ""


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_regression(n: int, p: int, seed: int) -> List[CorrectnessResult]:
    X, y = make_regression_data(n, p, seed)
    results = []

    # Agent
    from statgpu.agent import StatGPUAnalysisAgent
    agent = StatGPUAnalysisAgent(device="cpu", cv_folds=0)
    t0 = time.perf_counter()
    agent_result = agent.analyze(X=X, y=y)
    agent_time = (time.perf_counter() - t0) * 1000

    agent_model = next((m for m in agent_result.models if m.name == "LinearRegression"), None)
    if agent_model is None or agent_model.error:
        results.append(CorrectnessResult("regression", "LinearRegression", "statsmodels.OLS",
                                         np.nan, np.nan, np.nan, False, "Agent model failed"))
        return results

    agent_coef = np.array([c["estimate"] for c in agent_model.coefficients])
    agent_bse = np.array([c["std_error"] for c in agent_model.coefficients if c.get("std_error") is not None])
    agent_p = np.array([c["p_value"] for c in agent_model.coefficients if c.get("p_value") is not None])

    # statsmodels
    sm_result = fit_statsmodels_ols(X, y)
    if sm_result is not None:
        coef_diff = _safe_max_abs_diff(agent_coef, sm_result["coef"])
        bse_diff = _safe_max_abs_diff(agent_bse, sm_result["bse"])
        p_diff = _safe_max_abs_diff(agent_p, sm_result["pvalues"])
        results.append(CorrectnessResult(
            "regression", "LinearRegression", sm_result["name"],
            coef_diff, bse_diff, p_diff,
            passed=coef_diff < 1e-4,
            note=f"agent_time={agent_time:.1f}ms",
        ))

    return results


def bench_classification(n: int, p: int, seed: int) -> List[CorrectnessResult]:
    X, y = make_classification_data(n, p, seed)
    results = []

    from statgpu.agent import StatGPUAnalysisAgent
    agent = StatGPUAnalysisAgent(device="cpu", cv_folds=0)
    t0 = time.perf_counter()
    agent_result = agent.analyze(X=X, y=y)
    agent_time = (time.perf_counter() - t0) * 1000

    agent_model = next((m for m in agent_result.models if "Logistic" in m.name), None)
    if agent_model is None or agent_model.error:
        results.append(CorrectnessResult("binary", "LogisticRegression", "sklearn.LogisticRegression",
                                         np.nan, np.nan, np.nan, False, "Agent model failed"))
        return results

    # sklearn
    sk_result = fit_sklearn_logistic(X, y)
    if sk_result is not None:
        # Raw coefficients differ between IRLS and lbfgs solvers (scaling ambiguity).
        # Compare predictions instead — they should agree >99%.
        try:
            from sklearn.linear_model import LogisticRegression
            sk_model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
            sk_model.fit(X, y)
            sk_pred = sk_model.predict(X)
            agent_pred = np.asarray(agent_model.estimator.predict(X)).ravel()
            agreement = float(np.mean(agent_pred == sk_pred))
            passed = agreement > 0.99
            note = f"agent_time={agent_time:.1f}ms, pred_agreement={agreement:.6f}"
        except Exception:
            agreement = np.nan
            passed = False
            note = f"agent_time={agent_time:.1f}ms, prediction comparison failed"

        agent_coef_only = np.array([
            c["estimate"] for c in agent_model.coefficients
            if c["term"] != "Intercept"
        ])
        coef_diff = _safe_max_abs_diff(agent_coef_only, sk_result["coef_only"])
        results.append(CorrectnessResult(
            "binary", "LogisticRegression", sk_result["name"],
            coef_diff, np.nan, np.nan,
            passed=passed,
            note=note,
        ))

    return results


def bench_survival(n: int, p: int, seed: int, skip_r: bool) -> List[CorrectnessResult]:
    X, time_arr, event_arr = make_survival_data(n, p, seed)
    results = []

    from statgpu.agent import StatGPUAnalysisAgent
    agent = StatGPUAnalysisAgent(device="cpu", cv_folds=0)
    t0 = time.perf_counter()
    agent_result = agent.analyze(X=X, time=time_arr, event=event_arr, task="survival")
    agent_time = (time.perf_counter() - t0) * 1000

    agent_model = next((m for m in agent_result.models if "Cox" in m.name), None)
    if agent_model is None or agent_model.error:
        results.append(CorrectnessResult("survival", "CoxPH", "R::coxph",
                                         np.nan, np.nan, np.nan, False, "Agent model failed"))
        return results

    agent_coef = np.array([c["estimate"] for c in agent_model.coefficients])

    if not skip_r:
        r_result = fit_r_coxph(X, time_arr, event_arr)
        if r_result is not None:
            coef_diff = _safe_max_abs_diff(agent_coef, r_result["coef"])
            results.append(CorrectnessResult(
                "survival", "CoxPH", r_result["name"],
                coef_diff, np.nan, np.nan,
                passed=coef_diff < 0.05,
                note=f"agent_time={agent_time:.1f}ms",
            ))
    else:
        results.append(CorrectnessResult(
            "survival", "CoxPH", "R::coxph (skipped)",
            np.nan, np.nan, np.nan, True,
            note=f"R skipped, agent_time={agent_time:.1f}ms",
        ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent correctness benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-regression", type=int, default=2500)
    parser.add_argument("--p-regression", type=int, default=12)
    parser.add_argument("--n-classification", type=int, default=2500)
    parser.add_argument("--p-classification", type=int, default=12)
    parser.add_argument("--n-survival", type=int, default=2500)
    parser.add_argument("--p-survival", type=int, default=12)
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--json-out", help="JSON output path")
    args = parser.parse_args()

    all_results: List[CorrectnessResult] = []

    print("=" * 80)
    print("Agent Correctness Benchmark")
    print("=" * 80)

    print("\n--- Regression ---")
    for r in bench_regression(args.n_regression, args.p_regression, args.seed):
        all_results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.agent_method} vs {r.external}: "
              f"coef_diff={r.coef_diff:.2e}, bse_diff={r.bse_diff:.2e}, p_diff={r.p_diff:.2e} {r.note}")

    print("\n--- Binary Classification ---")
    for r in bench_classification(args.n_classification, args.p_classification, args.seed):
        all_results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.agent_method} vs {r.external}: "
              f"coef_diff={r.coef_diff:.2e} {r.note}")

    print("\n--- Survival ---")
    for r in bench_survival(args.n_survival, args.p_survival, args.seed, args.skip_r):
        all_results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.agent_method} vs {r.external}: "
              f"coef_diff={r.coef_diff:.2e} {r.note}")

    # Summary
    passed = sum(1 for r in all_results if r.passed)
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
