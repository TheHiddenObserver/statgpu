# PR #79 Physical GPU Validation — Final Report

Date: 2026-07-21  
Base SHA: `a4879fb4d9fb183efc01f147cd2cc501691f28c4`  
PR branch: `agent/code-review-fixes`  
PR head at the start of this documentation update: `2f18e5dec9195da1a12e5eea89ee2d832557b3ad`

## Decision

**MERGE-READY.** All mandatory validation gates passed. No unresolved CRITICAL or
HIGH correctness finding and no PR #79 regression remains.

Two pre-existing, non-blocking MEDIUM findings are tracked separately:

- finite-input validation consistency: GitHub issue #81;
- scikit-learn <=1.2 legacy clone compatibility: GitHub issue #82.

## Environment

- GPU: Tesla P100-SXM2-16GB
- Python: 3.9
- CuPy: 13.6.0
- PyTorch: 2.0.0+cu117
- Backends exercised: NumPy, CuPy CUDA, Torch CUDA

The performance numbers below are hardware- and environment-specific regression
baselines, not general performance guarantees.

## Gate results

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

## Gate B progression

| Stage | Passed | Failed | Skipped/XFAIL | Notes |
|---|---:|---:|---:|---|
| Initial | 1036 | 40 | 159 skipped | Baseline |
| Final | 1100 | 0 | 124 skipped + 1 XFAIL | All failures dispositioned |

Net result: **+64 passed, -40 failed, 100% of observed failures eliminated or
formally dispositioned**.

The strict XFAIL is `test_all_default_public_estimators_clone` under
scikit-learn <=1.2. The same 26-estimator failure was reproduced on the base SHA,
confirming that it is not introduced by PR #79.

## Gate F performance baseline

Tesla P100 synchronized median timings:

| Scale | Shape | CuPy median | Torch median |
|---|---:|---:|---:|
| Small | 200 x 5 | 2.9 ms | 3.7 ms |
| Medium | 2000 x 20 | 3.2 ms | 3.8 ms |
| Large | 10000 x 50 | 4.3 ms | 5.1 ms |

## Production defects fixed during physical-GPU validation

| File | Root cause and impact | Severity |
|---|---|---|
| `statgpu/panel/_utils.py` | CPU critical-value scalar multiplied with GPU arrays | CRITICAL |
| `statgpu/panel/_pooled.py` | Same device mismatch, categorical cluster transfer, and unstable rank-deficient solve | CRITICAL |
| `statgpu/backends/_utils.py` | Torch-only `device=` keyword passed to non-Torch `asarray` | HIGH |
| `statgpu/nonparametric/kernel_methods/_nystroem.py` | `device=` passed to CuPy array construction | HIGH |
| `statgpu/linear_model/wrappers/_linear.py` | Implicit `np.asarray(cupy_array)` on GPU inputs | HIGH |
| `statgpu/linear_model/penalized/_inference_mixin.py` | Post-fit state cleared although diagnostics require it | HIGH |
| `statgpu/glm_core/_fused.py` | Weighted fused loss called itself recursively | CRITICAL |
| `statgpu/feature_selection/_stepwise.py` | Constructor parameter identity violated the legacy sklearn clone contract | MEDIUM |

## Device-purity and memory conclusions

- No audited explicit GPU path transferred a complete numerical design matrix to CPU.
- Allowed host boundaries were limited to scalar statistics and metadata such as formula
  parsing, categorical labels, sort indices, and unsupported scalar distribution calls.
- Fifteen repeated fit/use/delete cycles on both CuPy and Torch showed no unbounded live
  allocation growth.

## Known non-blocking findings

### Finite-input validation

Ridge does not yet reject every NaN/Inf input before entering CUDA kernels. The normal
finite-input paths validated by this report are correct. A shared backend-native input
validation contract is tracked in issue #81.

### scikit-learn <=1.2 clone compatibility

Twenty-six public estimators normalize or defensively copy constructor parameters in a
way that violates the legacy clone identity check. The regression is version-limited and
marked `strict=True` XFAIL; the coordinated constructor refactor is tracked in issue #82.

## Auditable repository artifacts

- Validation plan: `dev/plans/pr79_gpu_review_fix_test_plan.md`
- Physical GPU tests: `dev/tests/test_pr79_physical_gpu.py`
- Orchestrator: `dev/validation/pr79_gpu_orchestrator.py`
- Environment/result helpers: `dev/validation/pr79_remote_utils.py`
- Result aggregation: `dev/validation/pr79_results.py`
- Result bundle convention: `results/pr79/<UTC-run-id>/`

A result bundle may be stored outside Git when it contains large machine-generated logs;
the paths above define the scripts and schema needed to reproduce and interpret it.
