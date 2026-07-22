#!/usr/bin/env python3
"""Core Accuracy Gate: statgpu 3-backend + Python/R references on same data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.runners.common import (
    make_case_id, make_method_config_id, make_raw_run,
    record_environment, synchronized_time, safe_run,
)


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


# ======================================================================
# Main accuracy run
# ======================================================================

def main():
    from dev.benchmarks.pr79.generators.linear import (
        generate_linear_full_rank, generate_linear_rank_deficient,
        generate_linear_weighted, case_params_linear,
        case_params_linear_rank_def, case_params_linear_weighted,
    )
    from dev.benchmarks.pr79.generators.survival import (
        generate_coxph_no_ties, generate_coxph_small_ties, generate_coxph_entry,
        generate_coxph_penalized,
        case_params_coxph_no_ties, case_params_coxph_small_ties,
        case_params_coxph_entry, case_params_coxph_penalized,
    )
    from dev.benchmarks.pr79.generators.panel import (
        generate_pooled_balanced, generate_pooled_rank_def,
        generate_pooled_cluster, case_params_pooled,
        case_params_pooled_rank_def,
    )

    env = record_environment()
    sha = _git_sha()
    session_id = f"pr79-{sha[:7]}-accuracy"
    out_dir = Path("results/pr79/accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    backends = ["numpy", "cupy", "torch"]
    n_warm, n_meas = 3, 5

    print(f"PR79 Core Accuracy Gate — SHA: {sha}")
    print(f"Session: {session_id}")
    print()

    # ==== Linear: full-rank, rank-def, weighted ====

    for label, gen_fn, case_fn in [
        ("linear-fr", lambda: generate_linear_full_rank(1000, 10, 42),
         case_params_linear),
        ("linear-rd", lambda: generate_linear_rank_deficient(200, 6, 42),
         case_params_linear_rank_def),
        ("linear-wt", lambda: generate_linear_weighted(500, 8, 42),
         case_params_linear_weighted),
    ]:
        data = gen_fn()
        X, y = data[0], data[1]
        sw = data[3] if len(data) > 3 and case_fn().get("weighted") else None
        cp = case_fn()
        case_id = make_case_id(cp)
        print(f"--- {label} (case {case_id}) ---")

        for b in backends:
            result, err = safe_run(_bench_linear, X, y, b, sw, n_warm, n_meas)
            if err:
                print(f"  {b}: FAILED")
                continue
            for br in result:
                mc = {"model_id": "LinearRegression", "backend": b, "cov_type": "nonrobust",
                      "compute_inference": True}
                if sw is not None:
                    mc["weighted"] = True
                runs.append(make_raw_run(
                    f"{label}-{b}-{br['iteration']}", case_id, make_method_config_id(mc),
                    "LinearRegression", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            t_med = np.median([r["fit_time_s"] for r in result])
            print(f"  {b}: {t_med*1000:.1f}ms, rank={result[0]['results'].get('rank_', '?')}")

    # ==== Linear rank-def + HC1 ====
    X, y, _ = generate_linear_rank_deficient(200, 6, 42)
    cp = case_params_linear_rank_def()
    case_id = make_case_id(cp)
    print(f"--- linear-rd-hc1 (case {case_id}) ---")
    for b in backends:
        result, err = safe_run(_bench_linear, X, y, b, None, n_warm, n_meas, cov_type="hc1")
        if err:
            print(f"  {b}: FAILED — {err}")
            continue
        for br in result:
            mc = {"model_id": "LinearRegression", "backend": b, "cov_type": "hc1",
                  "compute_inference": True}
            runs.append(make_raw_run(
                f"linear-rd-hc1-{b}-{br['iteration']}", case_id, make_method_config_id(mc),
                "LinearRegression", "statgpu", b, mc,
                {"fit_warm_s": br["fit_time_s"]}, br["results"],
            ))
        r = result[0]["results"]
        print(f"  {b}: rank={r.get('rank_')}, df_resid={r.get('_df_resid')}")

    # ==== CoxPH: no-ties, small-ties, entry, penalized ====
    for label, gen_fn, case_fn in [
        ("cox-no-ties", lambda: generate_coxph_no_ties(200, 4, 42),
         case_params_coxph_no_ties),
        ("cox-small-ties", lambda: generate_coxph_small_ties(300, 4, 42, 3),
         case_params_coxph_small_ties),
        ("cox-entry", lambda: generate_coxph_entry(200, 4, 42),
         case_params_coxph_entry),
        ("cox-pen", lambda: generate_coxph_penalized(100, 8, 42),
         case_params_coxph_penalized),
    ]:
        data = gen_fn()
        X, time_, event = data[0], data[1], data[2]
        entry_arr = data[3] if len(data) > 3 and case_fn().get("entry") else None
        penalty = case_fn().get("penalty", 0.0)
        cp = case_fn()
        case_id = make_case_id(cp)
        print(f"--- {label} (case {case_id}) ---")
        for b in backends:
            result, err = safe_run(_bench_coxph, X, time_, event, b, entry_arr, penalty,
                                   n_warm, n_meas)
            if err:
                print(f"  {b}: FAILED — {err}")
                continue
            for br in result:
                mc = {"model_id": "CoxPH", "backend": b, "ties": "efron",
                      "compute_inference": True, "penalty": penalty}
                if entry_arr is not None:
                    mc["entry"] = True
                runs.append(make_raw_run(
                    f"{label}-{b}-{br['iteration']}", case_id, make_method_config_id(mc),
                    "CoxPH", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            t_med = np.median([r["fit_time_s"] for r in result])
            r = result[0]["results"]
            print(f"  {b}: {t_med*1000:.1f}ms, ll={r.get('_log_likelihood', '?')}")

    # ==== Panel PooledOLS ====
    for label, gen_fn, case_fn in [
        ("pooled-bal", lambda: generate_pooled_balanced(30, 5, 3, 42),
         case_params_pooled),
        ("pooled-rd", lambda: generate_pooled_rank_def(20, 5, 4, 45),
         case_params_pooled_rank_def),
    ]:
        data = gen_fn()
        X, y, entity, time_idx = data[0], data[1], data[2], data[3]
        cluster = data[4] if len(data) > 4 else None
        cp = case_fn()
        case_id = make_case_id(cp)
        print(f"--- {label} (case {case_id}) ---")
        for b in backends:
            result, err = safe_run(_bench_pooled, X, y, entity, time_idx, cluster, b,
                                   n_warm, n_meas)
            if err:
                print(f"  {b}: FAILED — {err}")
                continue
            for br in result:
                mc = {"model_id": "PooledOLS", "backend": b, "cov_type":
                      "clustered" if cluster is not None else "nonrobust"}
                runs.append(make_raw_run(
                    f"{label}-{b}-{br['iteration']}", case_id, make_method_config_id(mc),
                    "PooledOLS", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            t_med = np.median([r["fit_time_s"] for r in result])
            print(f"  {b}: {t_med*1000:.1f}ms")

    # ==== Validate ====
    print(f"\n{'='*60}")
    print(f"Total runs: {len(runs)}")
    from dev.benchmarks.pr79.validators.numerical import (
        validate_backend_parity, validate_final_state_consistency,
    )
    parity = validate_backend_parity(runs)
    final_state = validate_final_state_consistency(runs)
    print(f"Backend parity:   {parity['passed']}/{parity['total_checks']} passed")
    print(f"Final-state:      {final_state['passed']}/{final_state['total_checks']} passed")

    # Save
    output = {
        "source_schema_version": "pr79-benchmark-source-1.0",
        "benchmark_session_id": session_id,
        "git_sha": sha,
        "environment": env,
        "runs": runs,
        "validation": {"backend_parity": parity, "final_state": final_state},
    }
    out_path = out_dir / "accuracy_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Print failing checks
    for check in parity.get("checks", []):
        if not check["passed"]:
            print(f"  FAIL: {check['run']} — {check['check']}: {check['value']} > {check['threshold']}")
    for check in final_state.get("checks", []):
        if not check["passed"]:
            print(f"  FAIL: {check['run']} — {check['check']}: {check['value']}")

    failure_count = parity.get("failed", 0) + final_state.get("failed", 0)
    print(f"\nOverall: {'PASS' if failure_count == 0 else 'FAIL'} ({failure_count} failures)")


# ======================================================================
# Bench helpers
# ======================================================================

def _backend_inputs(X, y, backend, sw=None):
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(X), cp.asarray(y), cp.asarray(sw) if sw is not None else None
    elif backend == "torch":
        import torch
        return (torch.as_tensor(X, dtype=torch.float64, device="cuda"),
                torch.as_tensor(y, dtype=torch.float64, device="cuda"),
                torch.as_tensor(sw, dtype=torch.float64, device="cuda") if sw is not None else None)
    return X, y, sw


def _extract(m):
    r = {}
    # Add information-matrix condition number for condition-aware thresholds
    vm = getattr(m, "_var_matrix", None)
    if vm is not None:
        try:
            if hasattr(vm, "get"): import cupy as cp; vm = cp.asnumpy(vm)
            elif hasattr(vm, "cpu") and hasattr(vm, "detach"): vm = vm.detach().cpu().numpy()
            vm_np = np.asarray(vm, dtype=float)
            r["_info_cond"] = float(np.linalg.cond(vm_np))
        except Exception:
            pass

    for a in ["coef_", "intercept_", "rank_", "rsquared", "aic", "bic",
              "_df_model", "_df_resid", "_bse", "_pvalues", "_log_likelihood",
              "_var_matrix", "_converged"]:
        v = getattr(m, a, None)
        if v is not None:
            try:
                if hasattr(v, "get"): import cupy as cp; v = cp.asnumpy(v)
                elif hasattr(v, "cpu") and hasattr(v, "detach"): v = v.detach().cpu().numpy()
            except: pass
            r[a] = v.tolist() if hasattr(v, "tolist") else float(v) if np.isscalar(v) else v
    return r


def _bench_linear(X, y, backend, sw=None, n_warm=3, n_meas=5, cov_type="nonrobust"):
    from statgpu.linear_model import LinearRegression
    Xd, yd, swd = _backend_inputs(X, y, backend, sw)
    dev = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    results = []
    for i in range(n_warm + n_meas):
        m = LinearRegression(fit_intercept=True, cov_type=cov_type,
                             compute_inference=True, device=dev)
        _, t = synchronized_time(m.fit, Xd, yd, sample_weight=swd)
        if i >= n_warm:
            results.append({"iteration": i - n_warm, "fit_time_s": round(t, 6),
                           "results": _extract(m)})
    return results


def _bench_coxph(X, time_, event, backend, entry=None, penalty=0.0, n_warm=3, n_meas=5):
    from statgpu.survival import CoxPH
    Xd, _, _ = _backend_inputs(X, np.zeros_like(time_), backend)
    dev = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    results = []
    for i in range(n_warm + n_meas):
        m = CoxPH(ties="efron", penalty=penalty, compute_inference=True,
                  device=dev, compute_cindex=False, tol=1e-6, max_iter=30)
        _, t = synchronized_time(m.fit, Xd, time=time_, event=event, entry=entry)
        if i >= n_warm:
            results.append({"iteration": i - n_warm, "fit_time_s": round(t, 6),
                           "results": _extract(m)})
    return results


def _bench_pooled(X, y, entity, time_idx, cluster=None, backend="numpy",
                  n_warm=3, n_meas=5):
    from statgpu.panel import PooledOLS
    Xd, yd, _ = _backend_inputs(X, y, backend)
    dev = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    cov = "clustered" if cluster is not None else "nonrobust"
    results = []
    for i in range(n_warm + n_meas):
        m = PooledOLS(cov_type=cov, device=dev)
        _, t = synchronized_time(m.fit, Xd, yd,
                                  cluster=cluster if cov == "clustered" else None)
        if i >= n_warm:
            results.append({"iteration": i - n_warm, "fit_time_s": round(t, 6),
                           "results": _extract(m)})
    return results


if __name__ == "__main__":
    main()
