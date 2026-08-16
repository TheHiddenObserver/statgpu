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

## Hosted validation

The final numerical/test/workflow source `0222ff8a338cc319ddf43b3295db63b69f37f40f` passed all seven permanent hosted workflows before physical promotion:

- Tests — SUCCESS (`31941035360`)
- Panel Stage C Torch CPU — SUCCESS (`31941035399`)
- Panel Stage C external covariance — SUCCESS (`31941035332`), including pinned Python definitions, the linearmodels Fama-MacBeth alignment, and R `plm`/`sandwich`
- Benchmark Frontend CI — SUCCESS (`31941035344`)
- Maintenance compatibility — SUCCESS (`31941035356`)
- Release notes validation — SUCCESS (`31941035368`)
- Release package validation — SUCCESS (`31941035511`)

## Fresh physical GPU acceptance

Physical promotion commit `114adfa0b3ccd1f8f2c97c96d8581ab3ea5ff1c4` has parent `0222ff8a338cc319ddf43b3295db63b69f37f40f` and adds only the two physical evidence JSON files below. It does not change production code, maintained tests, workflows, benchmark runners, package metadata, or model documentation.

Accepted evidence:

- `results/pr126_p100_fama_fix/panel_stage_c_correctness_0222ff8a.json`
  - schema 2, exact `git_sha=0222ff8a338cc319ddf43b3295db63b69f37f40f`;
  - clean source tree and `status=success`;
  - Tesla P100-SXM2-16GB, NumPy 1.24.2, SciPy 1.10.1, CuPy 13.6.0, Torch 2.0.0;
  - CuPy 35 estimator cases + 12 public covariance primitives and Torch 35 + 12 all passed with requested/executed backend provenance.
- `results/pr126_p100_fama_fix/fama_macbeth_review_fix_0222ff8a.json`
  - schema 3, exact `git_sha=0222ff8a338cc319ddf43b3295db63b69f37f40f`;
  - both required GPU backends validated, clean before/after checks, `validation_tier=remote-full`, `status=success`;
  - Newey-West and nonrobust inference, standard inference aliases, chronology, formula/missing-row behavior, rank rejection, both no-intercept spellings, prediction/output parity, and backend provenance passed;
  - same-workload synchronized NumPy/GPU timing is audit evidence only and explicitly makes no universal speedup claim.

The focused artifact reports `environment.cupy=null` because its generic package-distribution lookup asks for distribution name `cupy`, while this environment installs the CUDA-specific CuPy distribution. This is a non-blocking metadata representation issue: the companion Stage-C artifact on the same exact numerical source records CuPy 13.6.0, and the focused artifact independently proves executed backend `cupy` on Tesla P100. It does not invalidate physical correctness or backend acceptance.

Therefore the physical hard gate for numerical source `0222ff8a...` is **accepted**.

## Generated benchmark-asset boundary

Adding the two accepted JSON evidence files changes the recursive benchmark source inventory. Clean-checkout Benchmark Frontend CI run `31947948950` correctly generated an updated deterministic bundle and failed only the staleness check because the committed generated JSONs still reflected the previous inventory.

The clean-CI bundle reports:

- generation id `e66e68ce70ef027404e8fd027d0b0b45736267234d619b054f719e81324c7a02`;
- catalog digest `d2e766a67e0356606bb79aa02e0a28746f3ce9a267571f1e6326dd2e8a749e2f`;
- 117 discovered/classified JSON artifacts (115 prior + the two new physical evidence files);
- 21 eligible / registered / parsed canonical sources;
- 51 historical-or-excluded sources;
- 2580 generated benchmark runs.

Both new evidence JSONs are intentionally classified `historical_or_excluded` / `undated-json`, so they do not change the canonical benchmark run set. The six generated JSON assets must be synchronized from CI artifact `benchmark-generated-assets` (artifact ID `9263828575`, SHA-256 `fb23bede519d7bc25ba007fe1d4537abdd4a44e1b302963052e7feb92819707e`) rather than regenerated from a local checkout containing git-ignored historical `results/` files.

## Current exit state

The statistical/code-review and physical GPU gates are closed. The only remaining gate is deterministic generated-asset synchronization from the clean CI artifact followed by the resulting exact-head hosted recheck:

`PHYSICAL_GPU_ACCEPTED / REVIEW_CLEAN / GENERATED_ASSETS_PENDING`

No merge action is part of these review/fix loops.
