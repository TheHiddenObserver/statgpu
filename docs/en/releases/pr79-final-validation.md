# PR #79 Final Physical-GPU Validation

> Date: 2026-07-21  
> Hardware: Tesla P100-SXM2-16GB  
> Backends: NumPy, CuPy CUDA, Torch CUDA

PR #79 completed the physical-GPU validation required by the repository review plan.
All mandatory gates passed, with no PR-introduced regression and no unresolved
CRITICAL or HIGH correctness finding.

## Validation summary

| Gate | Result |
|---|---|
| GPU smoke | 160 passed, 0 failed, 2 expected skips |
| Three-backend correctness | 1100 passed, 0 failed, 124 skipped, 1 strict XFAIL |
| Metamorphic | 10/10 passed; one known finite-input finding |
| Device purity | Zero full-design transfers in three audited model families |
| Memory | Zero leaks over 15 CuPy and Torch repetitions |
| Performance | Synchronized measurements at three scales |
| External validation | Ridge aligned with scikit-learn; linear regression aligned with statsmodels |
| Full suites | CPU 1100 passed; GPU 1100 passed |

Gate B improved from 1036 passed and 40 failed to 1100 passed and zero failed.
The sole strict XFAIL applies to scikit-learn <=1.2 and reproduces on the base SHA,
so it is not a PR #79 regression.

## Correctness fixes

Physical-GPU execution exposed and resolved defects in panel inference, rank-deficient
PooledOLS, backend array construction, Nystroem, linear wrappers, debiased-inference
state retention, weighted GLM fused dispatch, and StepwiseSelector cloning.

The most severe defects were:

- CPU distribution scalars combined directly with GPU arrays;
- Torch-only `device=` arguments passed to NumPy/CuPy array constructors;
- implicit conversion of CuPy arrays through `np.asarray`;
- infinite recursion in weighted GLM fused loss/gradient calculation;
- post-fit inference state being cleared before diagnostics could use it.

## Performance baseline

| Shape | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

These are regression baselines for the recorded Tesla P100 environment, not portable
performance guarantees.

## Known follow-ups

- Issue #81 tracks consistent backend-native NaN/Inf validation.
- Issue #82 tracks the coordinated public-constructor refactor required for
  scikit-learn <=1.2 clone compatibility.

Neither finding blocks the finite-input paths validated in PR #79.

## Reproduction and evidence

- `dev/reviews/pr79_physical_gpu_validation.md`
- `dev/plans/pr79_gpu_review_fix_test_plan.md`
- `dev/tests/test_pr79_physical_gpu.py`
- `dev/validation/pr79_gpu_orchestrator.py`
- `dev/validation/pr79_results.py`
- result bundle convention: `results/pr79/<UTC-run-id>/`
