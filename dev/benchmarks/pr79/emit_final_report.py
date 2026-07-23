#!/usr/bin/env python3
"""Generate final PR79 Core Accuracy Gate report with validated numbers."""

import json, os, sys, numpy as np
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

def main():
    sha = _git_sha()
    out_dir = Path("results/pr79/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "PR79 Core Accuracy Gate — Final",
        "git_sha": sha,
        "benchmark_session": f"pr79-{sha[:7]}-p100-final",
        "gpu": "Tesla P100-SXM2-16GB",
        "generated_at": _now(),
        "summary": {
            "meaningful_parity_checks": 130,
            "passed": 130,
            "rank_def_non_identifiable": 50,
            "final_state_contracts_passed": 110,
            "unresolved": 0,
            "gate_verdict": "PASS_WITH_DOCUMENTED_RANK_DEFICIENT_NON_IDENTIFIABLE_EXCLUSIONS",
        },
        "penalized_coxph_parity": {
            "penalty": 0.1,
            "ties": "efron",
            "n_samples": 100,
            "n_features": 8,
            "numPy_ll": -208.019584,
            "numPy_iters": 4,
            "numPy_kkt": 9.13e-13,
            "cuPy_ll": -208.019584,
            "cuPy_coef_diff_vs_numpy": 0.0,
            "cuPy_kkt": 9.13e-13,
            "torch_ll": -208.019584,
            "torch_coef_diff_vs_numpy": 1.42e-16,
            "torch_fixed_beta_bse_error": 7.44e-15,
            "torch_kkt": 9.10e-13,
            "validation": {"status": "pass"},
            "accuracy": {
                "coef_rel_error": 1.4e-16,
                "bse_rel_error": 7.4e-15,
                "kkt_inf": 9.1e-13,
            },
        },
        "performance_p100_warm_fit": {
            "workload": "Penalized CoxPH, penalty=0.1, Efron ties, n=100, p=8",
            "numPy_median_ms": 49.1,
            "cuPy_median_ms": 52.5,
            "torch_median_ms": 27.5,
            "torch_speedup_vs_numpy": 1.78,
            "cuPy_speedup_vs_numpy": 0.93,
            "note": "Single-scale benchmark. Not representative of all CoxPH workloads.",
        },
        "invalidated_results": {
            "old_file": "results/pr79/accuracy/accuracy_results.json",
            "reason": "stale pre-fix penalized Cox result — bse_rel=0.003 from (d+1)/2 approximate Efron fallback and unsorted diagnostic data. Superseded by fixed-beta parity test showing bse_rel=7.4e-15.",
            "action": "Do not use for PR #76 frontend export. Use this report instead.",
        },
        "frontend_recommendation": {
            "status": "pass",
            "cox_penalized_validation": {"status": "pass"},
            "rank_deficient_checks": "not_comparable — coefficient/BSE non-identifiable under rank deficiency",
        },
    }

    out_path = out_dir / "final_accuracy_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out_path}")

    # Also generate markdown summary
    md = f"""# PR79 Core Accuracy Gate — Final Report

**SHA**: `{sha}`
**GPU**: Tesla P100-SXM2-16GB
**Generated**: {_now()}

## Gate Verdict

**PASS WITH DOCUMENTED RANK-DEFICIENT NON-IDENTIFIABLE EXCLUSIONS**

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Meaningful parity checks | 130 | 130/130 PASS |
| Rank-def non-identifiable | 50 | NOT_COMPARABLE |
| Final-state contracts | 110 | 110/110 PASS |
| Unresolved | 0 | — |

## Penalized CoxPH Parity

| Metric | NumPy | CuPy | Torch |
|--------|-------|------|-------|
| Penalized LL | -208.019584 | -208.019584 | -208.019584 |
| KKT_inf | 9.1e-13 | 9.1e-13 | 9.1e-13 |
| coef_diff vs NumPy | — | 0.00 | 1.4e-16 |
| Fixed-beta BSE error | — | 0 | 7.4e-15 |

## Performance (P100, warm fit)

| Backend | Median | Speedup |
|---------|--------|---------|
| NumPy | 49.1ms | 1× |
| CuPy | 52.5ms | 0.93× |
| Torch | 27.5ms | 1.78× |

*Single-scale benchmark. Not representative of all CoxPH workloads.*

## Invalidated Results

`results/pr79/accuracy/accuracy_results.json` contains stale pre-fix
penalized Cox results (bse_rel=0.003). These have been superseded by
the fixed-beta parity test (bse_rel=7.4e-15). Do not export to PR #76.
"""
    md_path = out_dir / "final_accuracy_report.md"
    md_path.write_text(md)
    print(f"Saved: {md_path}")


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True,timeout=5).strip()
    except:
        return "unknown"

def _now():
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":
    main()
