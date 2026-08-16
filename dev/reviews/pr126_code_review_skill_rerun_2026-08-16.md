# PR126 `.claude/skills/code-review.md` reruns — 2026-08-16

## Scope and active gates

First rerun baseline: `36a14ae0f38aec5bdcdd9eec0b45b7da2975bc05`.

Second independent rerun baseline: `3713ed47588ce3976b0addf5f84ea06227a7351a`.

Active axes in both reruns:

- `BACKEND`: NumPy / CuPy / Torch, explicit-device no-fallback contract.
- `INFER`: Fama-MacBeth coefficient-series covariance, standardized inference surface, retained-period identification, covariance-mode matrix, and external definition alignment.
- `FORMULA`: default intercept, explicit no-intercept rejection, missing-row side-array alignment, feature names.
- `MATRIX`: retained-period rank policy, covariance-mode matrix, and rank-revealing solve.
- `PERF`: per-period decomposition, materialization, transfer/synchronization cost, and target-workload timing evidence.
- `TEST`: maintained regression, Torch CPU matrix, pinned external baseline, and physical-runner contracts.
- `DOC`: root/EN/CN capability and validation-state consistency.
- `ARTIFACT`: physical evidence reproducibility and benchmark frontend staleness triggers.

Inactive axes with rationale:

- `CV`: Fama-MacBeth is non-tunable in this API.
- `LOSS`, `PENALTY`, regularization `SOLVER`: no regularized optimization capability is touched.

## First rerun findings and fixes

### [MEDIUM][DOC][fixed] stale exact-head physical-validation status

The detailed EN/CN changelogs described the post-`a99726e1` physical gate as awaiting a rerun even though the later Fama-MacBeth correctness source `464b587e83b234d78b5449666488d7f2f8ad367c` had already been validated on Tesla P100.

Fix:

- synchronized `CHANGELOG.md`, `docs/en/changelog.md`, and `docs/cn/changelog.md`;
- preserved older P100 runs as immutable historical evidence;
- recorded the accepted `464b587e...` Stage-C 35+12 and focused Fama-MacBeth evidence;
- explicitly reopened exact-head physical acceptance after subsequent numerical changes rather than incorrectly carrying historical evidence forward.

### [MEDIUM][TEST/ARTIFACT][fixed] benchmark staleness trigger did not match catalog scan root

The benchmark source catalog recursively scans `results/**/*.json`, while Benchmark Frontend CI previously enumerated only selected result directories.

Fix:

- both `push.paths` and `pull_request.paths` now include `results/**/*.json`;
- removed dependence on per-PR result-directory allowlists.

### [MEDIUM][PERF][fixed] retained-period rank guard duplicated matrix decomposition

The first correctness fix called a rank SVD and then solved normal equations, so every retained period paid for two decompositions.

Fix:

- `FamaMacBeth.fit()` routes each retained-period solve through the shared `panel_lstsq()` rank-revealing path;
- the full-rank requirement remains fail-closed before averaging/inference;
- maintained regression locks one rank-revealing SVD per retained period.

### [LOW][ARTIFACT][fixed] focused physical evidence lacked a committed runner

Fix:

- added `dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py` with exact-SHA and clean-worktree guards;
- the runner checks chronology, formula alignment, rank rejection, no-intercept behavior, backend provenance, numerical parity, and synchronized timing.

## Second independent rerun findings and fixes

### [MEDIUM][PERF/ARTIFACT][fixed] focused performance evidence could not close the performance gate

At the second rerun baseline, the focused runner recorded GPU raw timing but did not record a same-workload NumPy timing baseline, GPU/NumPy ratio, or `optimization_notes`; its numerical snapshot also omitted several inference outputs and predictions.

Fix:

- the focused runner now records synchronized NumPy and backend timing samples on the same 64-period workload, both medians, and `backend_over_numpy_median_ratio`;
- it records explicit `optimization_notes`, including the remaining serial-per-period structure and the interpretation that ratio > 1 means GPU is slower on that fixture;
- full parity covers `betas_`, `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `cov_params_`, and prediction;
- no universal GPU speedup claim is made.

### [MEDIUM][PERF][fixed] `panel_lstsq()` still materialized unused objects and synchronized an avoidable scalar

The first single-factorization change still implemented least squares through `panel_svd_pseudoinverse()`, which materialized the full pseudoinverse and an unused covariance bread. The rank cutoff also extracted `s_max` to host before extracting rank.

Fix:

- `_svd_inverse_factors()` exposes the common SVD factors/inverse singular-value mask;
- `panel_lstsq()` computes the coefficient directly from those factors and no longer materializes the residual-OLS covariance bread;
- the singular-value cutoff remains on the active backend and only the final integer rank crosses to Python control flow;
- `panel_svd_pseudoinverse()` retains its existing contract for consumers that actually require the bread.

### [HIGH][INFER/TEST][fixed] inference-capable Fama-MacBeth lacked the standard inference result surface

`FamaMacBeth` published `coef_`, `bse_`, `tvalues_`, `pvalues_`, and `conf_int_`, but did not publish `ParameterInferenceResult` or the common `_params`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, `_conf_int` aliases required by the inference matrix.

Fix:

- successful fit creates `ParameterInferenceResult(method="fama_macbeth")` from NumPy snapshots while leaving public CuPy/Torch arrays backend-native;
- Newey-West is labeled `z` / normal / `df=None`;
- nonrobust is labeled `t` / Student-t / `df=T-1`;
- metadata records coefficient-series covariance source, retained period count, effective Newey-West bandwidth, and formula feature names when available;
- NumPy and Torch-CPU maintained tests cover the standard inference aliases and `summary()` consistency;
- the focused physical runner validates the same inference result/alias contract on CuPy and Torch.

### [MEDIUM][API/INFER][fixed] failed refit could expose stale previous results

A successful fit followed by a failed new fit could leave the previous fitted/inference state visible.

Fix:

- `FamaMacBeth._reset_fit_state()` invalidates public results, standard inference aliases/result, fit statistics, backend references, formula/index metadata, and `_fitted`;
- every new `fit()` begins by resetting state before validation;
- maintained regression covers success → rank-deficient failed refit and verifies that stale coefficient/inference state is removed.

### [HIGH][INFER/BACKEND][fixed] nonrobust covariance branch lacked maintained GPU inference-matrix evidence

The focused GPU matrix exercised Newey-West but did not physically exercise the public nonrobust `t / df=T-1` inference branch.

Fix:

- focused physical validation runs both covariance modes on each of CuPy and Torch;
- the nonrobust case checks full numerical/output parity, `t` statistic labeling, Student-t distribution, `df=T-1`, and absence of a Newey-West bandwidth;
- `dev/tests/test_fama_macbeth_inference_matrix.py` covers both modes on maintained Torch CPU and is explicitly run by the Stage-C Torch CPU workflow.

### [MEDIUM][PERF][fixed] p-value evaluation synchronized one GPU scalar per parameter

Fama-MacBeth used a Python loop over `xp.abs(tvalues)` and called `_to_float_scalar()` for each coefficient. On CuPy/Torch this produced O(k) device-to-host synchronizations even though the distribution implementation is CPU-backed.

Fix:

- the complete statistic vector is transferred to NumPy once;
- the normal/Student-t survival function is evaluated vectorially once;
- the resulting p-value vector is transferred back to the active backend once;
- maintained Torch-CPU regression wraps the distribution object and verifies one vector `sf` call for both covariance modes.

### [HIGH][INFER/TEST][fixed] Fama-MacBeth lacked an available external inferential baseline

The model page previously stated that no maintained cross-package comparison existed. This was not a defensible inactive gate because pinned `linearmodels==7.0` provides Fama-MacBeth period estimates plus standard and kernel covariance estimators with parameterizations that can be aligned to statgpu.

Fix:

- added `dev/tests/test_fama_macbeth_linearmodels_external.py` to the pinned `Panel Stage C external covariance` workflow;
- the deterministic full-rank balanced fixture aligns explicit period intercepts, period ordering, and coefficient sets;
- period coefficients (`betas_` vs `all_params`) and averaged coefficients are compared in both modes;
- statgpu `nonrobust` is compared with linearmodels `cov_type="unadjusted", debiased=True` for covariance, BSE, and coefficient t-statistics; p-values/CI are intentionally not forced to agree because the APIs use different reference df after covariance alignment;
- statgpu Newey-West is compared with linearmodels `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False` for covariance, BSE, test statistics, p-values, and confidence intervals;
- the pinned linearmodels external-definition CI job passed this new comparison before documentation was updated.

### [MEDIUM][DOC][fixed] documentation lagged the strengthened inference/performance/external contract

Fix:

- root/EN/CN changelogs describe standardized inference, failure-safe refit, streamlined SVD solve, both covariance modes, same-workload NumPy/GPU timing ratio, and `optimization_notes`;
- EN/CN Fama-MacBeth pages document the remaining serial-per-period performance boundary, vectorized p-value transfer, standardized inference result, failure-safe refit, and the exact linearmodels 7.0 alignment boundary;
- docs explicitly distinguish the nonrobust covariance/BSE alignment from its intentionally different p-value reference df, so external matching cannot silently redefine statgpu inference.

## Validation boundary

Historical P100 evidence on `464b587e83b234d78b5449666488d7f2f8ad367c` remains immutable evidence for that source only. The second rerun changes `_linalg.py`, `_fama_macbeth.py`, maintained inference/external tests, the focused physical runner, and Torch/external workflows, so fresh physical acceptance is required for the final numerical source.

The last executable/test/workflow commit before model-document/review-note-only synchronization is `44f11533263562f558df0ab7b7ee86b93e515108`. Hosted acceptance for the final candidate must be grounded on this unchanged executable/test/workflow tree (or a later tree if any executable/test/workflow file changes again). The pinned Python external-definition job on this tree already passed the new linearmodels Fama-MacBeth tests; all remaining permanent hosted jobs must also complete successfully before `HOSTED_GREEN` is restored.

Both physical runners require a clean worktree at startup. Run both against one clean final SHA with outputs outside the repository, then copy successful artifacts into `results/`:

```bash
FULL_SHA="$(git rev-parse HEAD)"
SHORT_SHA="$(git rev-parse --short=8 HEAD)"
test -z "$(git status --porcelain)" || exit 1

python dev/benchmarks/validate_panel_stage_c_gpu.py \
  --out "/tmp/panel_stage_c_correctness_${SHORT_SHA}.json" \
  --expected-sha "${FULL_SHA}"

test -z "$(git status --porcelain)" || exit 1

python dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py \
  --out "/tmp/fama_macbeth_review_fix_${SHORT_SHA}.json" \
  --expected-sha "${FULL_SHA}"

test -z "$(git status --porcelain)" || exit 1

mkdir -p results/pr126_p100_fama_fix
cp "/tmp/panel_stage_c_correctness_${SHORT_SHA}.json" \
  "results/pr126_p100_fama_fix/panel_stage_c_correctness_${SHORT_SHA}.json"
cp "/tmp/fama_macbeth_review_fix_${SHORT_SHA}.json" \
  "results/pr126_p100_fama_fix/fama_macbeth_review_fix_${SHORT_SHA}.json"
```

The focused runner requires both CuPy and Torch, exercises Newey-West and nonrobust inference, checks the standard inference result/private aliases and full public output/prediction parity, and records same-workload synchronized NumPy/GPU timing evidence. Adding the evidence JSONs intentionally changes the benchmark source inventory; regenerate deterministic frontend/docs benchmark assets before finalizing the evidence commit.

## Current exit state

Until the final executable/test/workflow hosted matrix and both fresh physical commands pass:

`PARTIAL_REMOTE_PENDING`

No merge action is part of these review/fix loops.
