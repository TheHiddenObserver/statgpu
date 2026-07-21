# PR #79 Physical GPU Validation and Post-Validation Review — Final Report

Date: 2026-07-21  
Base SHA: `a4879fb4d9fb183efc01f147cd2cc501691f28c4`  
PR branch: `agent/code-review-fixes`  
Physical-GPU validated campaign head: `2f18e5dec9195da1a12e5eea89ee2d832557b3ad`  
Exact-head physical-GPU acceptance SHA: `786af9e2eb4742a56e5203b4380b03aec63a3ac8`

## Decision

**MERGE-READY.** The complete Tesla P100 Gate A–G campaign passed, the subsequent
review-fix cycle was completed, and the exact cleaned head
`786af9e2eb4742a56e5203b4380b03aec63a3ac8` passed the mandatory focused physical-GPU
acceptance suite with **17 passed, 0 failed, and 0 skipped in 7.28 seconds**.

CuPy CUDA and Torch CUDA both executed under `STATGPU_REQUIRE_PHYSICAL_GPU=1`. The exact
SHA and clean-worktree state were recorded. Standard GitHub Actions Tests run #495 passed
on the final documentation/cleanup head `1d877d65db0926f38170ec851f0f0479937bcd61`.
No unresolved CRITICAL/HIGH defect or PR-introduced regression is known.

Issues #81 and #82 and the Torch Cox Hessian memory optimization remain explicitly tracked,
non-blocking follow-ups.

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

### Post-validation review-fix and exact-head evidence

The post-validation review repaired additional backend-routing, PooledOLS, WLS, formula,
validator, and GPU inference edge cases. Standard GitHub Actions Tests run #495 completed
successfully on the final cleanup head with:

- regression matrices on Python 3.9, 3.10, 3.11, and 3.12;
- static-contract, compilation, and complete-collection gates;
- the complete CPU test suite.

The mandatory Tesla P100 exact-head acceptance ran on clean SHA
`786af9e2eb4742a56e5203b4380b03aec63a3ac8`:

```text
17 passed in 7.28s
```

Both CuPy and Torch CUDA parameterizations executed with no skips. The suite confirmed
weighted fit/predict parity, formula missing-row weight alignment, device-purity guards,
and backend-consistent degenerate F-statistic semantics.

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
| GPU overall F-test edge cases | The early return mixed perfect-fit and intercept-only cases and returned an incorrect p-value. CuPy/Torch now return `(inf, 0.0)` for perfect non-constant fits and `(nan, nan)` when the overall test is undefined. | HIGH |

Permanent regression coverage includes scikit-learn/statsmodels parity, rank-deficient
PooledOLS inference, HAC row-order invariance, formula intercept behavior, invalid weight
contracts, multi-output WLS broadcasting, Patsy missing-row alignment, pipeline failure
propagation, exact-SHA worktree checks, and physical CuPy/Torch parity tests.

## Exact-head physical-GPU acceptance — PASS

Command:

```bash
STATGPU_REQUIRE_PHYSICAL_GPU=1 \
python -m pytest dev/tests/test_pr79_final_review_fixes.py -q -rs --tb=short
```

Recorded result on clean SHA `786af9e2eb4742a56e5203b4380b03aec63a3ac8`:

```text
17 passed in 7.28s
```

Acceptance results:

1. CuPy CUDA available and executed: PASS.
2. Torch CUDA available and executed: PASS.
3. No GPU parameterization skipped: PASS.
4. Weighted fit/predict parity: PASS.
5. Formula missing-row and sample-weight alignment: PASS.
6. Perfect-fit overall F test `(inf, 0.0)` on CuPy and Torch: PASS.
7. Intercept-only overall F test `(nan, nan)` on CuPy and Torch: PASS.
8. Exact SHA and clean-worktree state recorded: PASS.

The physical-GPU validation loop is closed. PR #79 may be marked Ready for review.

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
