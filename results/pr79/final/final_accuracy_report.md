# PR79 Core Accuracy Gate - Final Report

**Validated code SHA**: `bef91ad2cd19fa2ab575e701f645799eaff6aff9`  
**Report generator SHA**: `074fd5c556970332400d2866fdd9e0c70d2e37db`  
**GPU**: Tesla P100-SXM2-16GB  
**Generated**: 2026-07-23T07:12:00Z

## Gate Verdict

**PASS WITH DOCUMENTED RANK-DEFICIENT NON-IDENTIFIABLE EXCLUSIONS**

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Meaningful parity checks | 130 | 130/130 PASS |
| Rank-def non-identifiable | 50 | NOT_COMPARABLE |
| Final-state contracts | 110 | 110/110 PASS |
| Physical P100 acceptance | 33 | 33/33 PASS |
| Unresolved | 0 | PASS |

## Penalized CoxPH Parity

| Metric | NumPy | CuPy | Torch |
|--------|-------|------|-------|
| Penalized LL | -208.019584 | -208.019584 | -208.019584 |
| KKT_inf | 9.1e-13 | 9.1e-13 | 9.1e-13 |
| coef_diff vs NumPy | N/A | 0.00 | 1.4e-16 |
| Fixed-beta BSE error | N/A | 0 | 7.4e-15 |
| Iterations | 4 | 4 | 4 |
| Convergence | PASS | PASS | PASS |
| Termination | kkt_converged | kkt_converged | kkt_converged |

## Performance (P100, warm fit)

Protocol: 10 warmup fits followed by 10 measured fits.

| Backend | Median | Speedup vs NumPy |
|---------|--------|------------------|
| NumPy | 49.1 ms | 1.00x |
| CuPy | 52.5 ms | 0.93x |
| Torch | 27.5 ms | 1.78x |

*Single-scale benchmark. Not representative of all CoxPH workloads.*

## Invalidated Results

`results/pr79/accuracy/accuracy_results.json` contains stale pre-fix
penalized Cox results (`bse_rel=0.003`). They are superseded by the
fixed-beta parity result (`bse_rel=7.4e-15`) and must not be exported
to PR #76.

## Frontend Export Status

- Overall validation status: `pass`
- Penalized CoxPH validation status: `pass`
- Rank-deficient coefficient and coefficient-level BSE checks: `not_comparable`
