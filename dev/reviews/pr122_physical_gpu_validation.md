# PR #122 Panel Stage B physical GPU validation

## Current acceptance status

**PARTIAL_REMOTE_PENDING**.

The previously accepted Panel Tier-1 Stage B numerical implementation `a57efcea29b0e87ecb89865c5a6902d5773812c6` remains useful historical physical evidence, but it is no longer sufficient for final PR #122 acceptance because `dev/benchmarks/validate_panel_stage_b_gpu.py` changed after that measurement.

A Ready-for-review Codex pass identified a physical-coverage gap: all four Hausman diagnostics per CUDA backend in the `a57efcea...` artifact are structured `applicable = false` results because the fitted covariance difference is not positive semidefinite. Those results correctly validate applicability/reason parity against NumPy, but they do not execute an applicable Hausman statistic, p-value, and degrees-of-freedom path on either physical GPU backend.

The review/fix loop therefore reopened the physical gate and returned PR #122 to Draft.

## Current runner fix

The current physical runner now contains a dedicated deterministic fitted FE/RE Hausman fixture with:

- seed `20260810`;
- 12 entities and 4 observations per entity;
- one slope;
- nonzero entity-effect scale `0.005`;
- noise scale `0.1`;
- 48 observations total.

On the hosted NumPy reference under the physical-like Python 3.9 / NumPy 1.24.2 / SciPy 1.10.1 stack, this fixture is stably applicable with:

- Hausman df `1`;
- finite statistic and p-value;
- a positive FE-minus-RE diagnostic covariance margin greater than `1e-6`.

The runner now requires each requested physical backend to provide at least one diagnostic that simultaneously has:

- `status = "success"`;
- `applicable = true`;
- finite `statistic`;
- finite `pvalue`;
- finite positive `df`.

For the dedicated applicable fixture the runner also checks requested/executed backend identity and compares FE/RE coefficients, FE/RE diagnostic covariance, Hausman statistic, p-value, and df against the NumPy reference. The original four structured-inapplicable Hausman parameterizations remain in the matrix, so the eventual physical run must cover both applicability branches rather than replacing one with the other.

Hosted regression coverage also locks the nonzero-effect fixture and verifies that the aggregate physical gate fails closed when every Hausman result is inapplicable or when an `applicable=true` payload omits statistic/p-value/df.

The targeted review gate passed under Python 3.9 with NumPy 1.24.2 and SciPy 1.10.1 before the focused runner/test fix was committed.

## Historical raw evidence

The last accepted full Stage-B physical artifact is retained unchanged:

- path: `results/pr122_p100/panel_stage_b_gpu_validation_a57efcea.json`
- Git blob: `254b64776bff4e3b2b642bb4a2ae1eea25f4751c`
- schema version: 2
- measured implementation SHA: `a57efcea29b0e87ecb89865c5a6902d5773812c6`
- clean working tree: true
- overall status: success
- CuPy: 17/17 estimator cases, requested backend `cupy`, no fallback
- Torch: 17/17 estimator cases, requested backend `torch`, no fallback
- four Hausman diagnostics per backend, all matching NumPy as structured inapplicable results
- no timing or speedup measurement collected.

The focused disconnected two-way FE artifact is also retained unchanged:

- path: `results/pr122_p100/panel_stage_b_disconnected_fe_gpu_validation_a57efcea.json`
- Git blob: `3bda0b2040479ba8201e2722eb990ba086c3f3b9`
- measured implementation SHA: `a57efcea29b0e87ecb89865c5a6902d5773812c6`
- clean working tree: true
- `nobs = 9`
- legacy residual df `0`
- component-aware/public residual df `1`
- effect rank `7`
- `rank_x = 1`
- incidence components `3`
- CuPy and Torch requested/executed backend provenance verified
- corrected df=1 confidence intervals agree with NumPy to machine precision.

These artifacts continue to establish the previously reviewed estimator, inference, disconnected-FE, and structured-inapplicability behavior. They do **not** establish the newly required applicable-Hausman physical branch.

## Historical environment

The `a57efcea...` artifacts report:

- Tesla P100-SXM2-16GB;
- Python 3.9.16;
- statgpu 0.2.4;
- NumPy 1.24.2;
- SciPy 1.10.1;
- PyTorch 2.0.0.

The raw artifact records the CuPy package metadata field as `null`, but physical CuPy execution is independently established by the per-case `executed_backend = "cupy"` checks.

## Existing canonical frontend evidence

The currently committed canonical source remains the historical `a57efcea...` measurement:

- `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260809.json`
- SHA-256 `a6e47b9dec9c35040d2715b573aa2a93b084d83c574078bdb430861f0a9aa9f9`
- source id `panel-stage-b-pr122-20260809-a6e47b9dec9c`
- 42 validation-only rows.

It remains an immutable description of that historical physical run, but it must not be treated as final acceptance for the current runner. After the new exact-clean-head P100 run succeeds, a new canonical source must replace it and the frontend generated assets/contracts must be regenerated. With the new applicable Hausman diagnostic, the expected Stage-B validation row count is 44: 17 estimator cases x 2 backends plus 5 Hausman diagnostics x 2 backends.

## Required remote closure

Final physical acceptance now requires a fresh exact-clean-head P100 execution of `dev/benchmarks/validate_panel_stage_b_gpu.py` on the final runner head, with both CuPy and Torch requested.

The new artifact must demonstrate, for each backend:

1. requested backend equals executed backend with no silent fallback;
2. all 17 estimator cases remain successful against NumPy;
3. the four existing Hausman parameterizations retain their expected structured applicability behavior;
4. `hausman_applicable_nonzero_effect` is `applicable = true`;
5. its Hausman statistic, p-value, and df agree with NumPy within the runner tolerances;
6. its FE/RE coefficients and diagnostic covariance agree with NumPy within the runner tolerances;
7. the top-level run reports a clean working tree and exact expected SHA.

After that run is accepted, the lifecycle must:

- commit the immutable raw result;
- promote a new canonical Stage-B source;
- update source/coverage contracts for five Hausman diagnostics per backend and 44 validation rows;
- regenerate the six tracked frontend/docs JSON assets;
- rerun exact-final-head hosted CI;
- resolve the applicable-Hausman review thread with the new evidence;
- perform another fresh review under `.claude/skills/code-review.md` before returning the PR to Ready.

## Hard-exit conclusion

Local fix status: the current review finding is fixed in runner/test code and targeted local/hosted checks pass.

Remote status: a fresh P100 CuPy/Torch artifact is required because the physical validation runner changed.

**Hard exit: PARTIAL_REMOTE_PENDING.**
