# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PARTIAL_REMOTE_PENDING**.

The Tesla P100 evidence measured from exact clean implementation SHA `aad53587c9611da0e71a676e86ef32d9f6403f5c` remains valid historical evidence for the Stage-C numerical covariance/validator contract at that SHA, but it is no longer sufficient for final PR acceptance.

A subsequent strict review found and fixed inference/formula presentation defects in production files under `statgpu/panel/**`:

- `PooledOLS` and `BetweenOLS` formula fits now preserve the implicit `Intercept` in summary and unified inference feature-name metadata;
- `PanelSummary` now labels sandwich/normal-reference statistics as `z` / `P>|z|` while retaining the historical `tvalues_` field/API;
- related formula/inference regressions were added, including Patsy categorical-reference, interaction, transform, prediction and transformed-fit-space covariance coverage.

These fixes do not change covariance formulas, coefficient estimates, standard errors, p-values, confidence intervals, backend selection, or the physical validation runners. However, `RELEASING.md` requires fresh exact-source physical acceptance for inference-facing production changes. Therefore the previous claim that no `statgpu/panel/**` file changed after `aad53587...` is no longer true, and the physical gate is reopened until a fresh clean-head CuPy/Torch run is recorded on the final review-fix source head.

## Historical correctness and backend-provenance evidence

- path: `results/pr126_p100/panel_stage_c_gpu_validation_aad53587.json`
- measurement SHA: `aad53587c9611da0e71a676e86ef32d9f6403f5c`
- artifact repository commit: `fdf88e8990b30502958cc357099342d271792ec2`
- Git blob: `4fc52f29021321a9d7b1f87fc76950c843ef4059`
- SHA-256: `aab3ac61315b78ecaf04351f2922a444e3f44c587b31f9832ccad32839601b6c`
- GPU: Tesla P100-SXM2-16GB
- result at measurement SHA: CuPy 32/32 and Torch 32/32 = 26 estimator covariance cases + 6 direct public primitives per backend
- direct primitives: `cluster_group_debias`, `driscoll_kraay_qs`, `ill_conditioned_hc0`, `ill_conditioned_hc2`, `ill_conditioned_hc3`, `ill_conditioned_dk`
- requested backend equaled the backend persisted at the numerical fit boundary; no numerical CPU fallback was observed

## Historical synchronized performance evidence

- path: `results/pr126_p100/panel_stage_c_performance_aad53587.json`
- measurement SHA: `aad53587c9611da0e71a676e86ef32d9f6403f5c`
- artifact repository commit: `fdf88e8990b30502958cc357099342d271792ec2`
- Git blob: `fd7b06e0bb3a51e6a74e524e5bc059063b5f75eb`
- SHA-256: `99208f9276b92abf212d76655616718980095ba5c8ff9d3356a795a15ffa50c6`
- GPU: Tesla P100-SXM2-16GB for both CuPy and Torch
- rows: 58 = 54 base rows + 4 high-T QS rows
- timing scope: synchronized end-to-end estimator fit
- high-T scenario: `N=10,000`, `k=2`, `T=200`
- no speedup claim or CPU-baseline claim is made

## Canonical promotion audit

The `aad53587...` evidence remains registered as the canonical historical Stage-C source contract until replacement evidence is accepted:

- correctness source: `panel-stage-c-validation-pr126-20260810-aab3ac61315b`;
- performance source: `panel-stage-c-performance-pr126-20260810-99208f9276b9`;
- the physical parser expects 26 estimator cases plus all 6 direct primitives and emits 64 validation rows across CuPy/Torch;
- the canonical benchmark bundle contains 1984 runs after adding the eight newly promoted primitive rows;
- recorded maximum absolute differences are finite diagnostics, while the physical runner's `status=success` remains authoritative for its `rtol+atol` NumPy parity check because the artifact does not store the reference magnitudes needed to reconstruct the relative-tolerance term;
- the prior `c151550a...` and `9c0b3050...` artifacts remain immutable older historical evidence;
- deterministic benchmark frontend assets were regenerated after promotion, and temporary promotion helpers are absent from the final tree.

No canonical source identifier or raw evidence file is rewritten merely because acceptance is reopened. A fresh exact-head physical run must produce new immutable evidence and then be promoted through the same parser/manifest/frontend audit path.

## Strict review findings closed since `aad53587...`

- **[HIGH][MATRIX] fixed** — RandomEffects formula matrix now covers categorical reference levels, interactions, transforms, column order, prediction, and Patsy design equivalence.
- **[HIGH][MATRIX] fixed** — BetweenOLS and FirstDifferenceOLS HC0/HC2/HC3 now have independent statsmodels baselines on their entity-mean / first-difference fit spaces.
- **[HIGH][FORMULA/INFER] fixed** — PooledOLS/BetweenOLS formula-added intercept metadata is preserved in summaries and `ParameterInferenceResult`.
- **[MEDIUM][DOC] fixed** — EN/CN Panel docs no longer mix the old 26+2 physical status with the fresh 26+6 matrix.
- **[MEDIUM][INFER/API] fixed** — robust/HC/cluster/DK summaries display normal-reference `z` labels instead of misleading `t` labels.

Focused review regressions passed after each fix, including the Stage-A golden suite for the formula metadata change. No new numerical covariance or backend implementation defect was found in the re-reviewed production paths.

## Remaining acceptance gates

Before PR #126 can return to **PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**:

1. permanent hosted workflows must be green on the final review-fix head;
2. final re-review must have no unresolved CRITICAL, HIGH, or relevant MEDIUM finding;
3. run `dev/benchmarks/validate_panel_stage_c_gpu.py` on that exact clean head for both CuPy and Torch;
4. run `dev/benchmarks/benchmark_panel_stage_c_covariance.py` on the same exact clean head for both CuPy and Torch;
5. audit and promote the new immutable artifacts without changing the measured production source or validator contract afterward.

Until these gates close, PR #126 must remain Draft and must not be described as merge-ready.
