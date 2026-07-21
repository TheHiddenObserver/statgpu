# PR #79 Physical GPU Validation and Post-Validation Review — Final Report

Date: 2026-07-21  
Base SHA: `a4879fb4d9fb183efc01f147cd2cc501691f28c4`  
PR branch: `agent/code-review-fixes`  
Physical-GPU validated code head: `2f18e5dec9195da1a12e5eea89ee2d832557b3ad`  
Latest cleaned code head after the review-fix loop: `ff72424071ec7ca52399146dbd8a556534c9e6c3`

## Decision

**CONDITIONALLY MERGE-READY.** The complete Tesla P100 validation campaign passed on
`2f18e5d`. A subsequent review-fix loop found and repaired additional correctness and
validation-infrastructure defects. The cleaned post-review code head `ff72424` passes the
full standard GitHub Actions suite, including Python 3.9–3.12 regression matrices, static
contracts, complete test collection, and the full CPU suite.

Because the post-validation changes include `LinearRegression` CuPy/Torch weighted-fit
paths, one focused physical-GPU recheck on the exact latest code head remains required
before changing this PR from Draft to Ready for review. The older P100 results must not be
represented as exact-head validation for these later changes.

No unresolved CRITICAL/HIGH defect is known from the completed review-fix cycle.

## Evidence boundary

### Complete physical-GPU campaign — validated head `2f18e5d`

Environment:

- GPU: Tesla P100-SXM2-16GB
- Python: 3.9
- CuPy: 13.6.0
- PyTorch: 2.0.0+cu117
- Backends: NumPy, CuPy CUDA, Torch CUDA

| Gate | Scope | Result |
|---|---|---|
| A | GPU smoke | **PASS** — 160 passed, 0 failed, 2 expected skips |
| B | Three-backend correctness | **PASS** — 1100 passed, 0 failed, 124 skipped, 1 strict XFAIL |
| C | Metamorphic properties | **PASS** — 10/10; one known NaN/Inf finding recorded |
| D | Device purity | **PASS** — zero full-design transfers; three model families audited |
| E | Memory leak | **PASS** — zero leaks over 15 repeated cycles on CuPy and Torch |
| F | Performance | **PASS** — synchronized timings at three scales on both GPU backends |
| G | External validation | **PASS** — Ridge versus scikit-learn; linear regression versus statsmodels |
| Final | Complete CPU and GPU suites | **PASS** — CPU 1100 passed; GPU 1100 passed |

Gate B improved from **1036 passed / 40 failed / 159 skipped** to
**1100 passed / 0 failed / 124 skipped / 1 strict XFAIL**. The clone XFAIL under
scikit-learn <=1.2 reproduces for the same 26 estimators on base SHA `a4879fb` and is
tracked in issue #82.

### Post-validation review-fix evidence — cleaned code head `ff72424`

GitHub Actions Tests run #477 completed successfully with:

- regression matrices on Python 3.9, 3.10, 3.11, and 3.12;
- static-contract, compilation, and complete-collection gates;
- the complete CPU test suite.

Each repair commit was also gated by the focused suite
`dev/tests/test_pr79_final_review_fixes.py` together with the maintained linear and panel
regression suites before being pushed. All temporary patch/workflow infrastructure was
then deleted atomically; only production changes and permanent regression tests remain.

## Additional defects fixed by the post-validation review-fix loop

| Area | Root cause and repair | Severity |
|---|---|---|
| `LinearRegression` backend routing | Eager NumPy conversion occurred before backend resolution. Raw CuPy/Torch arrays are now preserved until backend-native conversion. | HIGH |
| `LinearRegression.predict` | Non-formula inputs were eagerly converted with `np.asarray`. Prediction now preserves backend-native inputs until dispatch. | HIGH |
| PooledOLS HAC ordering | HAC covariance implicitly depended on input row order. Optional `time_index` now validates and stably orders observations. | HIGH |
| PooledOLS rank deficiency | Residual degrees of freedom used the column count instead of effective rank. Least-squares rank now drives `df_resid`. | HIGH |
| Validation orchestrator | Pipelines could mask pytest failure; worktrees could be dirty or point at stale SHAs; the base tree could be overwritten. Commands now use `pipefail`, exact SHAs, immutable base, reset/clean checks, and required explicit head SHA. | HIGH |
| Formula intercept semantics | Formula syntax set an intercept decision and then immediately restored the public constructor value. A private effective-intercept state now controls fitting without mutating clone-visible parameters. | HIGH |
| Weighted `LinearRegression` | The intercept column was not multiplied by `sqrt(weight)`, multi-output weighting broadcast incorrectly, and raw versus weighted residual state was conflated. CPU/CuPy/Torch paths now implement the same WLS transformation, validation, fallback solve, diagnostics, and weighted R² semantics. | CRITICAL |
| Formula sample weights | Patsy could drop rows while `sample_weight` retained original length. Formula evaluation now returns retained row positions and aligns weights deterministically. | HIGH |

Permanent regression coverage includes scikit-learn/statsmodels parity, rank-deficient
PooledOLS inference, HAC row-order invariance, formula intercept behavior, invalid weight
contracts, multi-output WLS broadcasting, Patsy missing-row alignment, pipeline failure
propagation, exact-SHA worktree checks, and optional physical CuPy/Torch parity tests.

## Required exact-head physical-GPU recheck

Reset the GPU validation worktree to `ff72424071ec7ca52399146dbd8a556534c9e6c3`,
confirm a clean worktree, and run:

```bash
python -m pytest dev/tests/test_pr79_final_review_fixes.py -q -rs --tb=short
```

Acceptance criteria:

1. CuPy and Torch CUDA are both available and the two GPU-parametrized tests do not skip.
2. Weighted fit/predict parity passes for both GPU backends.
3. Formula + missing rows + original-length sample weights matches the CPU reference.
4. The exact checked-out SHA and clean-worktree status are recorded with the result.

After this focused recheck passes, update this report with the run identifier and change the
PR from Draft to Ready for review. Re-running performance and memory gates is optional
because the post-validation repairs do not introduce new persistent GPU allocations or a
new algorithmic complexity class; the focused correctness/device test is mandatory.

## Previously fixed production defects from the full GPU campaign

- panel critical-value device mismatches and categorical cluster handling;
- rank-deficient panel solving;
- Torch-only `device=` leakage into NumPy/CuPy constructors;
- CuPy 13.x/Nystroem construction failures;
- debiased-Lasso fitted-state loss;
- weighted GLM fused-dispatch recursion;
- StepwiseSelector legacy sklearn clone behavior.

## Known non-blocking follow-ups

- Issue #81: shared backend-native NaN/Inf validation consistency.
- Issue #82: coordinated constructor refactor for scikit-learn <=1.2 clone identity.
- Torch Cox Hessian `O(n*p*p)` intermediate allocation remains a separate performance item.

## Auditable repository artifacts

- Validation plan: `dev/plans/pr79_gpu_review_fix_test_plan.md`
- Physical GPU tests: `dev/tests/test_pr79_physical_gpu.py`
- Post-review regression tests: `dev/tests/test_pr79_final_review_fixes.py`
- Orchestrator: `dev/validation/pr79_gpu_orchestrator.py`
- Environment/result helpers: `dev/validation/pr79_remote_utils.py`
- Result aggregation: `dev/validation/pr79_results.py`
- Result bundle convention: `results/pr79/<UTC-run-id>/`
