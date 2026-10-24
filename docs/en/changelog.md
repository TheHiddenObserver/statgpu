# Changelog

> Language: English  
> Last updated: 2026-10-22<br>
> This page: Release history  
> Switch: [Chinese](../cn/changelog.md)

This page records user-visible changes for current and recent statgpu releases.

## Unreleased — working-tree repairs

### Benchmark qualification improvements (PR pending)

- Separate clean synchronized CV public-fit samples from profiled selector/refit diagnostics; preserve objective, folds, grids and numerical thresholds.
- Reject active native Torch profiling and explicit nonconverged CV/final-refit results before retaining clean samples, including Cox RuntimeWarnings.
- Keep Gaussian ingress observation outside measured fits and add explicit workflow preflight and host/mixed timing labels.
- Expand learning algorithm-stage evidence and separate helper tracing from timing; unresolved or hybrid computation is not labelled wholly native.
- Add a seven-suite qualification/campaign interface with exact selected inventories, independent thread profiles, bounded scales and immutable raw evidence. Unknown GPU exclusivity remains diagnostic-only, never a speedup or dashboard claim.
- This harness change does not modify production algorithms or Logistic defaults. New-source physical GPU validation remains pending.

### PR #257 — Full-scan correctness and evidence repairs

- Export native fitted arrays explicitly only at the benchmark oracle/report boundary, including Fama–MacBeth coefficients; native fitting and numerical tolerances are unchanged. The P100 follow-up failure remains historical pending a new-source rerun.

- Isolate injected Torch CPU test controls from ambient CUDA availability; public GAM CPU routing, numerical tolerances, and physical GPU tests are unchanged. The first P100 failure remains historical and requires a new-source rerun.
- Restore smooth-quadratic FISTA stopping on GPU without increasing iteration budgets or relaxing convergence tolerances.
- Keep squared-error L2 PenalizedGLM_CV final refits on the requested numerical backend.
- Evaluate Torch F survival probabilities for numerator degrees of freedom 2 with a stable native exact expression; retain other degrees-of-freedom behavior.
- Move Nyström landmark decomposition and GAM knot selection to their selected backends, and propagate KernelPCA decomposition failures without a hidden CPU retry. Public host reporting boundaries remain explicit.
- Strengthen algorithm-specific benchmark operation evidence and failure diagnostics; local CPU tests do not replace the pending P100 rerun. Existing full-scan evidence and diagnostic-only performance qualification remain unchanged.


These source changes are not included in the published 0.3.0 wheel or sdist.
The historical 0.3.0 release-preparation entries below retain their original
wording. Package versions are unchanged, and earlier P100 acceptance does not
validate these numerical changes.

### Native KDE bootstrap (2026-10-22)

- CuPy bootstrap now uses native RNG, refits and percentiles; Torch bootstrap
  and weighted Torch KDE fitting are supported. Shared Torch weight and query
  shape fixes also apply to kernel regression, not to `bandwidth_per_feature`.
- Public interval arrays remain NumPy. Native string selectors refit per draw;
  the NumPy one-dimensional Gaussian fast path still fixes the initial factor.
  Native seeding changes random streams and Generator state consumption.
- Optional `workspace_bytes` controls an estimated managed-buffer budget, not
  a hard peak-memory cap. See the [guide](guides/native-kde-bootstrap.md).
- Gaussian log-density excludes zero-mass rows from its reduction. Native
  selector sorting uses stable ties and PCA fixes the loading sign; NumPy
  selector algorithms remain unchanged. Weighted multivariate factors can
  differ materially even with a nondegenerate PCA axis, because weighted
  quantile resampling is not reflection-invariant. Repeated/near-repeated top
  eigenvalues introduce additional direction ambiguity and a `RuntimeWarning`,
  aggregated across fits in a public interval call. Native projection
  diagnostic arrays remain backend-resident; selector `to_dict()` stays shallow.
- Historical experimental-candidate GPU results do not validate this public
  implementation or establish its performance.

### Fixed (2026-10-20)

- **Native sampling (#173):** F sampling now translates public `dfn`/`dfd`
  to NumPy's `dfnum`/`dfden`, removing the keyword `TypeError`. All 15 native
  samplers apply an explicit output `dtype`; `None` still gives float64,
  including discrete families. NumPy/SciPy CPU random generation followed by
  selected-backend conversion is unchanged; this is not native GPU RNG.
  Regression coverage: `dev/tests/test_distribution_sampling_regressions.py`.
- **Inverse-Gaussian and negative-binomial robust covariance:** HC0/HC1 now
  use the observed loss Hessian for sandwich bread instead of expected Fisher
  scoring curvature. The penalty curvature, analytic-weight score meat,
  HC1 row-count multiplier and fixed/selected-tuning interpretation are
  unchanged. Solver Hessians and nonrobust covariance remain Fisher-based;
  the separate oracle path already used observed bread. This affects ordinary
  and supported smooth M-estimation fits and CV final refits, so robust standard
  errors can change without changing fitted coefficients. The derivative
  interpretation is interior to the unclipped mean range, not a new clipping
  derivative rule. Regression coverage:
  `dev/tests/test_glm_observed_bread.py`.
- **Cyclic cubic splines (#216):** analytic one-sided endpoint values and first
  and second derivatives replace outside-range finite differences. A resolved
  rank-three projection produces `n_knots + 1` float64 columns; unresolved rank
  fails explicitly. Constants remain representable, without a dedicated
  intercept column. Refit models built on the former basis; no reusable
  training projection or extrapolation API was added. Natural spline numerical
  limitations are unchanged. Regression coverage:
  `dev/tests/test_cyclic_spline_periodic_constraints.py`.
- **UMAP NumPy 2 dispatch (#191):** neighbor search and negative-sampling RNG
  use resolved backend identity rather than the presence of an array `device`
  attribute. NumPy CPU NNDescent no longer routes through Torch. Exact/auto
  neighbor selection, host SciPy graph assembly, spectral-initialization and
  CPU one-dimensional layout limitations are unchanged. Regression coverage:
  `dev/tests/test_unsupervised_umap.py`.

### Validation (2026-10-20)

The listed targeted regression suites and benchmark smoke cases are local
correctness checks, not physical CUDA or performance acceptance. Final counts
and source identity belong to the repair validation artifacts; unavailable GPU
routes remain pending and are not inferred from previous release evidence.

## 0.3.0 — Upcoming

This release brings together the merged changes since 0.2.5. Entries previously
marked Unreleased or planned for 0.2.6 belong to 0.3.0; this history does not
record a separate 0.2.6 release. The versioned [GitHub Release notes](../../.github/releases/v0.3.0.md)
provide the installation and upgrade overview.

### Highlights and upgrade guidance

- **Official website launched:** [statgpu.github.io](https://statgpu.github.io/) provides an online entry point to the English and Chinese documentation.
- Gaussian linear-model inference and supported sparse Gaussian residual bootstrap now perform numerical work on the fitted NumPy, CuPy, or Torch backend. Completed reporting arrays can still be NumPy arrays; GPU fitting does not imply every inference method supports GPU execution.
- Use `solver` for direct penalized fits and `LassoCV.cv_solver` for CV selection. For behavior-preserving migration, remove a direct estimator's deprecated `cpu_solver` without copying its value into `solver`; doing so can select a different algorithm. Use `inference_method="post_selection_ols"` instead of deprecated `cpu_ols`/`gpu_ols`, and select hardware through `device`. See the [solver migration guide](guides/penalized-solver-api-migration.md) and [inference methods](guides/inference-modes.md).
- **Changed inference results:** automatic node-wise Lasso tuning now depends on the standardized design rather than response scale. Existing debiased intervals and p-values can therefore change. Set keyword-only `nodewise_alpha` to a positive value when an explicit nuisance penalty is required; this does not select the prediction penalty.
- **Stricter unsupported requests:** non-Gaussian L1/ElasticNet coefficient inference and incompatible Quantile solver/penalty combinations raise instead of returning incomplete or substituted results. SCAD/MCP oracle inference remains CPU-only and conditional on the selected active set; failed oracle inference clears fitted state. Supported smooth non-Gaussian L2/no-penalty M-estimation and selected-final-refit CV inference do not correct tuning-selection uncertainty.
- Oracle convergence checks remain family-specific: only Tweedie and negative-binomial refits add relative first-order score checks. Poisson and other families retain their child solver's existing stopping criterion, including absolute tolerances where used. Small-scale data may require a stricter `tol` or rescaling; range checks are not a convergence certificate, and a relative score check does not bound coefficient or covariance error. See the [oracle numerical limits](guides/penalized-glm-inference.md#unpenalized-refits-and-family-settings).
- RidgeCV honors exact custom training subsets, including weighted and reversed-index folds. It still uses finite-precision raw Gram moments; promoting them cannot recover information already lost. Disable mixed precision or rescale inputs when appropriate, and do not interpret finite coefficients as an accuracy certificate.
- Whole-cluster bootstrap samples complete groups and can return variable-size samples. Missing cluster labels are rejected; literal string IDs such as `"nan"` remain valid. Ordinary formula prediction can still drop missing rows, although LinearRegression scoring now rejects the resulting shape mismatch.
- The expanded tutorials describe existing restrictions as well as fixes. UMAP neighborhood/initialization limitations, KernelPCA failed-refit state and KernelPCA/Nystroem Torch placement limitations, and the documented knockoff tied-statistic limitation are not repaired by the documentation refresh. Consult the [model guides](models/README.md) before relying on those workflows.

### Validation scope for release preparation (2026-10-19)

Historical PR #249 testing on source `0969d9da21b293ac569905bc95bd1fbdcb51f13c`
recorded **4,229 CPU passes**, **233 Torch CPU passes**, **824 CUDA passes**
(**412 CuPy + 412 Torch CUDA**), and **six native CUDA probes**. Collection
contained **5,288 items**, including **two optional dual-GPU cases**. The runs
used a single Tesla P100; collection is not execution, and the optional dual-GPU
cases do not establish multi-GPU acceptance. The merged baseline
`5b9d42476f7b921918b7202142f7d39ffd10cdd0` has the identical Git tree
`6065449b`. Its three hosted workflows and fourteen checks passed. These are
historical baseline results, not whole-release certification or speed claims.

Before publication, the final release source still needs its required CPU,
Python 3.9–3.12, documentation/static, packaging, clean wheel/sdist installation,
Ubuntu/Windows/macOS wheel smoke, and applicable physical-GPU checks under
[the release procedure](../../RELEASING.md). Feature-specific records below
apply only to their recorded sources, validators and hardware. Preparing these
notes does not publish a tag or a PyPI release.

### Additional documentation and tooling changes (PR #165, #170, #136, #141)

- PR #165 added [float32 L-BFGS guidance](guides/lbfgs-float32-precision-contract.md) and diagnostic tests. Native float32 optimization can follow different rounding paths across backends; compare objective and stationarity, and use same-backend float64 references when tighter coefficient reproducibility matters. This was a diagnostic/documentation change, not a replacement L-BFGS implementation.
- PR #170 improved Chinese model and unsupervised-learning explanations; PR #168 subsequently reorganized the mirrored English/Chinese tutorial and reference navigation.
- PR #136 and #141 updated contributor procedures and project tooling without adding a public numerical method.

The entries below retain detailed API, statistical, and numerical changes.

### Numerical contract fixes (PR #249, 2026-10-17 ~ 2026-10-19)

#### Fixed (2026-10-17 ~ 2026-10-19)

- **RidgeCV custom training rows (#243)**:
  - Added a fail-closed check for an exactly zero stored Gram diagonal with a nonzero matching cross-product, before regularization. CuPy float32 comparisons now use same-width integer views with the sign bit masked on the original device, fixing both missed inconsistencies and false rejections observed with floating conversion. Signed zeros remain zero; nonzero subnormal storage and genuine zero-feature/zero-response cases are preserved. Other backend/dtype paths retain their existing comparison behavior. This detects inconsistent moments but does not reconstruct factor information lost to rounding, cancellation or flush-to-zero. Disable mixed precision or rescale inputs when this error occurs.
  - Retained the original subnormal-factor mixed-precision example as an explicit precision-failure regression where its moments are lost, with a float64 analytic success control. The promoted-coefficient scoring success regression now uses normal Gram entries while still requiring coefficients beyond float32 range, and checks score separation as well as alpha selection. This aligns the test with the previously narrowed raw-moment contract; it does not claim the original lost-factor case is now accurately solved.
  - Corrected difficult native Gram solves that could return a finite, non-minimum-norm rank-one solution without raising a linear-algebra error. Numerical-rank checks now precede singular fallback; exact independent blocks preserve small identifiable directions, tiny systems use bounded upward scaling, and ambiguous coupled scale/rank cases raise explicitly. The ordinary batched path and raw weighted objective remain unchanged; this does not restore the withdrawn factor-reconstruction guarantees.
  - Supported effective penalties in the float32 subnormal storage interval are checked before narrowing and use same-device float64 arithmetic when material, avoiding CuPy flush-to-zero rejection. Values outside the configured precision's positive finite rounding interval still raise. Empty-feature final refits avoid an inapplicable zero-dimensional CuPy triangular solve while retaining the existing fit/inference lifecycle.
  - Preserved the original physical-failure inputs, with the explicit subnormal-factor test contract correction described above. Final-refit near-zero coefficient checks now use an independent high-precision objective and a precision-derived formation/solve error bound rather than requiring a cancellation result to be bitwise zero; other strict CV score, selected-alpha and prediction assertions retain their tolerances. CPU/Torch CPU execution does not establish physical CUDA success.
  - Weighted zero-feature designs with an explicit alpha grid again retain valid intercept-only or zero-prediction CV behavior. Empty slope spectra bypass inapplicable reductions while retaining fold scoring, validation and final refitting. Automatic alpha generation still requires at least one slope column.
  - Withdrew the additional factor-SVD, exact-duplicate compression, compensated-projection, and extreme/subnormal-factor recovery promises considered during development. CPU weighted folds retain the original raw weighted objective; difficult GPU folds use same-device native per-alpha Gram solves with numerical-rank fallback. This promotes already-formed moments to float64 but does not reconstruct the original factors. Computable existing raw-Gram cases remain covered, including rank-one/duplicate designs and raw-weighted subnormal-scale examples. Native normal-equation conditioning and finite-range limits remain; finite coefficients alone do not certify accuracy. See the [RidgeCV numerical boundary](reference/linear-model-api.md#custom-ridgecv-training-subsets).
  - Weighted validation subtracts finite residuals before applying square-root weights, preserving cancellation where the direct residual is representable. Nonfinite weighted validation MSE raises `FloatingPointError` with scaling guidance. RidgeCV now also rejects nonfinite final-refit coefficients or intercept before publishing any CV/refit state; direct Ridge behavior is unchanged.
  - The unweighted fold-statistics shortcut previously inferred training rows from the validation complement even when callers supplied smaller training subsets. RidgeCV now uses exactly the requested training and validation rows with or without sample weights, including gaps, embargoes, and validation sets that do not cover all observations.
  - Weighted folds retain anchored centering and positive subset-weight validation. Raw weighted Gram matrices and cross-products preserve the original weighted objective; common-weight scale invariance is limited by their range and precision at the solve dtype, including subnormal precision loss. Weighted GPU moments are formed in native float64 before conversion; weighted centering and weighted validation stay in float64, while ordinary mixed-precision solves use float32 Gram matrices. The effective fold-scaled penalty is formed before this conversion. GPU eigensolves do not impose an absolute positive eigenvalue floor. Unrepresentable positive effective regularization or nonfinite moments, regularized systems or coefficients raise `FloatingPointError`; this does not detect every finite inaccurate solve.
  - Reversed NumPy training or validation indices previously retained negative strides that Torch rejected, particularly when checking native Torch sample weights. Validated indices now receive a layout-only copy when necessary, including reversed single-element arrays, preserving row order and caller-owned inputs. Regression tests compare complete, subset, partial, and singleton splits against independent closed-form CV scores and full-data refits with both intercept settings and unweighted, unit-weight, and varying-weight inputs. A separate Torch CPU conversion test covers this indexing boundary; physical CUDA tests remain hardware-dependent.
  - Indices must be nonempty, one-dimensional, integral, in bounds, unique within each subset, and disjoint between training and validation. Explicit folds are scored even with fewer than four rows or one alpha candidate; `cv>=2` remains required, but supplied splits determine the actual folds. Each training and validation subset must have positive total sample weight, while individual zero weights remain allowed. Full-data-minus-validation statistics are reused only when every training set is the actual validation complement and the validation sets partition all rows. The selected model is still refitted on every supplied fit row.
  - Implementation and regression coverage: `statgpu/linear_model/cv/_ridge_cv.py`, `dev/tests/test_issue243_ridge_custom_folds.py`, and the independent analytic fold-loss checks in `dev/tests/test_pr168_linear_cycle7.py`.
- **Unpenalized SCAD/MCP oracle refits (#227)**:
  - Kept the original-order negative-binomial strict-convergence regression and added a target-equivalent row-reordered success control at unchanged tolerances. One Python 3.11 / SciPy 1.10 / NumPy 1.24 BLAS environment reaches a loss-resolution floor before the child score tolerance; the existing explicit rejection and complete state invalidation are tested, including a deterministic initial-child warning control. This test-portability repair does not change the shared solver or claim row-order-invariant convergence.
  - Non-Gaussian oracle refits now check each family's unclipped predictor/mean/covariance range before publishing uncertainty; Tweedie and negative-binomial refits also check relative first-order scores. This rejects clipped Tweedie/Gamma/inverse-Gaussian refits, Poisson mean-floor distortion, negative-binomial early stopping and model-based curvature-floor distortion, and logistic covariance-floor distortion. The checks use only positive-weight rows and preserve eligible exact endpoints. A failed check raises `RuntimeError` and clears fitted/reporting state. Smaller `tol` can help an unclipped early-stopped refit but cannot remove clipping; use `compute_inference=False` when only point estimation is required. NB alone may use one score-scaled absolute-tolerance retry, retaining the original final relative threshold and at most two child fits of `max_iter` each; failures remain transactional. Gaussian refits and shared solvers are unchanged. See the [family-specific ranges and failure guidance](guides/penalized-glm-inference.md#unpenalized-refits-and-family-settings).
  - Updated retained physical-GPU RidgeCV mixed-precision regressions to require explicit failure when raw float32 Gram products or penalties are unrepresentable, consistent with the previously documented recovery-scope reduction. Kept float64 and ordinary-scale success controls, and added native Torch CPU selector checks; no physical-GPU case was removed or skipped by this correction.
  - Gamma/Tweedie nonrobust oracle dispersion now uses the actual family variance in Pearson residuals instead of an absolute denominator floor. Small, representable response scales therefore no longer produce spuriously narrow intervals. Existing analytic weights and residual-degree conventions are preserved; nonrepresentable scales still raise explicitly.
  - Corrected oracle z statistics and normal-reference p-values for tiny positive standard errors. Active-set and intercept-only refits now divide estimates by their actual positive standard errors, preserving the test when estimates and standard errors are rescaled together; the existing exactly-zero-SE convention is unchanged.
  - Nonrobust oracle inference now raises `RuntimeError` if estimated dispersion is zero despite a nonzero evaluated response-minus-mean residual on any positive-weight row, including zero-design rows. At positive dispersion, all active model-based variances must be positive; zero off-diagonals remain valid. Genuine evaluated zero-residual fits may retain zero estimated dispersion; logistic, Poisson, and negative-binomial refits retain fixed dispersion one. These are numerical failure checks, not a general covariance-conditioning guarantee.
  - Dividing robust-inference weights by their maximum could round a positive low-weight observation to zero even when its large residual supplied a substantial covariance contribution. Common weight scaling now preserves positive weights and material score contributions. Conservative bounds check both underflow and overflow when choosing a common scale. An empty conservative scaling interval can still use unchanged weights when actual aggregate products are finite, required nonzero squared-score averages remain normal, and bounded omitted tiny terms are below ordinary aggregate roundoff. If neither stable scaling nor bounded raw evaluation can be certified at working precision, or robust scores, curvature, or active covariance are nonfinite, inference raises `RuntimeError` instead of publishing invalid uncertainty; a finite exact covariance may still exist. Scale invariance remains subject to representable weighted products and their precision, including intrinsic subnormal limits; genuine zero scores, individual zero variances, and inactive `NaN` uncertainty remain allowed. An entirely zero active HC0/HC1 covariance now raises `RuntimeError` when effective-row score factors prove the score matrix is nonzero. Zero-weight and zero-design rows do not trigger this check; it does not detect every partially underflowed covariance entry or change nonrobust behavior. Child refit weights are unchanged.
  - Unknown-dispersion `nonrobust` oracle fits (Gaussian, both Gamma links, inverse Gaussian and Tweedie) now raise `ValueError` when positive-weight observations do not outnumber selected coefficients, including the effective intercept. This prevents saturated or zero-weight-padded samples from reporting zero SEs or point-width intervals. Residual-degree counting, fixed-dispersion Poisson/NB covariance, and existing HC0/HC1 behavior are preserved; the Gamma/Tweedie Pearson correction is described above.
  - Constructor reconstruction previously lost family-specific settings and could retain default regularization or select an unintended device. Non-Gaussian oracle refits now use exactly `C=0`, preserve the fitted Gamma link, negative-binomial dispersion and Tweedie power, forward convergence controls, and explicitly use CPU with IRLS for logistic/Poisson or Newton for the other supported non-Gaussian families.
  - Prediction and scoring also use the fitted negative-binomial/Tweedie family settings, including typed defaults and explicit overrides; Gamma prediction respects the fitted inverse or log link. Prediction continues to use the penalized coefficients.
  - Adaptive-L1 initialization uses the same resolved negative-binomial/Tweedie family settings, including typed defaults and explicit overrides. Negative-binomial and inverse-Gaussian oracle HC0/HC1 covariance now uses the observed-Hessian sandwich bread; nonrobust covariance retains Fisher information, and HC1 retains the original observation count, including zero-weight rows. Oracle HC0/HC1 covariance now rescales weights before squared score products to avoid underflow or overflow from their common magnitude; refit weights and nonrobust conventions remain unchanged.
  - Errors during oracle computation now clear fitted, prediction, and inference-report state for generic and typed estimators, including state from an earlier successful fit. A successful fit is required before reuse. Child non-convergence raises `RuntimeError` before an oracle table is published. All-zero effective Poisson/NB/Tweedie responses and single-class logistic responses are checked for a finite unpenalized target; a free intercept permits a strictly improving direction and these fits raise `ValueError` instead of publishing finite uncertainty. Zero-weight rows do not alter this boundary. Without an intercept, a selected-design direction is checked so valid mixed-sign finite targets remain available; numerically uncertifiable boundary designs fail explicitly with `RuntimeError`. This is not a general separation detector. Empty slope support with an intercept receives an intercept-only refit when its target is finite; an empty inverse-link Gamma model without an intercept is rejected. Prediction coefficients remain penalized, inactive uncertainty remains `NaN`, and ordinary normal-reference intervals do not adjust for selection. Oracle inference remains CPU-only.
  - Implementation and regression coverage: `statgpu/linear_model/penalized/_inference_mixin.py`, the typed negative-binomial/Tweedie family resolvers, and `dev/tests/test_issue227_oracle_refit_contract.py`.
- **Whole-cluster bootstrap (#199)**:
  - Made nested-label test oracles portable across NumPy coercion versions by independently accounting for the existing estimator finite-input guard and requiring exactly one additional snapshot conversion. Missing-label rejection, zero callbacks, dtype/group/sample assertions and production grouping are unchanged; mutation controls detect missing or redundant snapshots.
  - Structured masked-array cluster labels now reduce masks field by field, including nested records and subarray fields. All-false masks retain the previously accepted grouping; an actually masked field raises the existing missing-label `ValueError` rather than a structured-dtype reduction error.
  - Missing-value checks now inspect dtype-inferred nonscalar results from nested array converters before mixed string siblings can stringify a numeric NaN. Each nested conversion result is reused for grouping. Actual masked host entries also raise; all-false masks and literal missing-looking string IDs remain valid. A custom converter that itself returns strings still cannot reveal discarded missingness.
  - Preserved the existing top-level backend-transfer conversion for sequence wrappers, so a `.get()` or `.cpu()` conversion remains authoritative for grouping. Label prevalidation does not replace those hooks with sequence iteration or evaluate them an extra time; nested ordinary sequences retain missing-value checks.
  - Nested custom array conversions no longer receive an extra object-dtype conversion during label validation. Their ordinary conversion behavior is preserved, including existing errors when an array protocol is explicitly declared as `None`. Object results are still checked for missing labels; a custom conversion that has already replaced missing values with strings retains the documented limitation.
  - Mixed string sequences could convert actual NaN values to the literal string `"nan"` before validation. Both module and estimator helpers now check host labels before conversion, including `None`, NaN, NaT, and pandas missing scalars in object arrays and supported host sequences such as list, tuple, deque, UserList, and legacy length/index sequences. Missing-label validation and grouping reuse the same captured host-sequence labels without the extra source reads introduced by validation; the estimator’s existing finite-input checks remain. Prevalidation does not consume iterators or add calls to custom array/device conversion protocols, including dynamically exposed array converters. Genuine string IDs such as `"nan"`, `"None"`, and `"NaT"` remain valid; native numeric backend grouping and seeded samples are unchanged. Regression tests cover both entry points, lossless object arrays, literal-string negative controls, accepted sequence forms, caller dtype/shape preservation, and unchanged custom conversion counts.
  - The previous row-count stopping rule could truncate a sampled cluster. Each resample now draws the original number of clusters with replacement and concatenates whole groups, preserving aligned arrays. Unequal cluster sizes yield variable row counts and scalar callbacks without batch probes; equal-size dense seeded sampling is unchanged.
  - Unequal-size groups reject `force_vectorized=True` except for `statistic_hint="mean"` on exactly one one-dimensional array. Missing cluster labels raise `ValueError`; the separate stratum/group-label limitations remain. Metadata includes `resampling_unit`, `n_clusters`, `clusters_per_resample`, and `variable_sample_size`.
  - Implementation and regression coverage: `statgpu/inference/_resampling.py`, `dev/tests/test_issue199_whole_cluster_bootstrap.py`, and executable bilingual resampling examples.
- **LinearRegression score shapes (#178)**:
  - Subtracting a column response from one-dimensional predictions could silently broadcast a residual matrix. Single-target scoring now accepts `(n,)` or `(n,1)` equivalently, while genuine multi-target responses must match prediction shape exactly. Incompatible rows or targets raise `ValueError` before residual arithmetic on the selected backend.
  - Formula prediction still drops missing query rows; scoring now rejects a shortened prediction against the full response rather than broadcasting it. Constant targets retain a score of 0, and multi-target scoring retains the mean of target-wise R² values.
  - Implementation and regression coverage: `statgpu/linear_model/wrappers/_linear.py`, `dev/tests/test_issue178_linear_score_shape.py`, the column-response learner example, and the formula-score regression in `dev/tests/test_pr168_linear_cycle8.py`.

#### Validation (2026-10-17)

- Added focused numerical and documentation regressions without relaxing the independent references or numerical tolerances. Previously expected failures are enabled only for the repaired oracle, RidgeCV-subset, and formula-score cases; unrelated prediction and numerical limitations remain tracked separately. Physical CUDA checks are conditional on available hardware; CPU execution does not establish physical-GPU validation or performance.

### Multiple-testing Torch devices (PR #171)

#### Fixed (2026-10-15)

- **Torch input-device preservation in multiple testing**:
  - The module functions `adjust_pvalues`, `multipletests`, and `combine_pvalues` previously selected Torch without binding its allocations to the input tensor's concrete device. Conversion could move CPU inputs when CUDA was available, or create ranks, adjustment factors, or weights on a different CUDA device.
  - Torch `pvalues` tensors now retain their CPU or exact indexed CUDA device with `backend="auto"` or explicit `"torch"`. Float32 inputs are promoted to float64 on that device; temporary arrays, supplied/default weights, and outputs remain together. Weights do not select the backend or device.
  - Explicit `backend="torch"` with non-Torch p-values uses the current concrete CUDA device when available, otherwise CPU. Unsupported Torch p-value devices, including MPS, raise `ValueError` because these functions support float64 execution on CPU and CUDA; no silent device fallback is introduced.
  - Module functions still ignore `set_device`. Estimator helpers retain their separate CUDA requirement and input conversion when they select Torch; shared backend-factory behavior is unchanged. Existing numerical warnings for NaN `alpha`, overflowing weight sums, and extreme probability tails still apply.
  - Updated the bilingual multiple-testing API, device guide, and estimator helper reference. Added device-ownership regressions in `dev/tests/test_multiple_testing_device_ownership.py`, including conditional physical CUDA checks. No physical GPU validation is claimed for this local update.

### Learner guides and API references (PR #168)

#### Improved (2026-10-10)

- Reused the covariance finite guard’s normalized working input within each public call. Supported NumPy-bound Torch `fit`, `score`, `predict`, and `mahalanobis` calls perform one Torch-to-NumPy normalization, including the nested prediction path; native Torch FP8 calls reuse their source-device widening. Validation still checks every supplied numerical argument, including ignored `y`, and no input cache is retained on the estimator. Numerical results, backend selection, input ownership, and exception behavior are unchanged. `dev/tests/test_covariance_normalization_reuse.py` checks conversion counts and preservation contracts; these counts do not establish a wall-clock speedup.

#### Fixed (2026-10-10)

- **Covariance Torch conversion and FP8 validation**: all seven covariance estimators normalize supported real Torch inputs before public finite validation. NumPy-bound inputs move to CPU before conversion to `float64`, preserving `bfloat16`, `float8_e5m2`, lazy-negative views, and MPS-to-CPU conversion. Native Torch `float8_e5m2` validation widens a temporary on the original device rather than relying on a raw FP8 finite-check kernel. Previously Torch 2.5.1 CUDA rejected those FP8 inputs before the covariance conversion helper ran. NaN and positive/negative infinity still raise; fitted backend ownership, original input/storage, gradients, explicit unavailable-device errors, and existing CuPy/pandas handling are preserved. Coverage is in `dev/tests/test_covariance_torch_inputs.py` and `dev/tests/test_covariance_fp8_prevalidation.py`, including a missing-kernel ordering emulator. CPU results do not establish physical CUDA or MPS acceptance; hardware reruns remain required. Other float8 formats, complex, sparse, and quantized input support is unchanged.

- Restored numeric pandas `DataFrame` inputs for fitting, scoring, prediction, and Mahalanobis distances across all seven covariance estimators, plus single-feature `Series` fits. Device preparation had confused pandas' keyed `get` method with a CuPy transfer method. Ordinary array-like inputs are now normalized before device conversion, while actual CuPy and Torch arrays retain the existing device-policy handling. Regression coverage checks these inputs against NumPy results, validates malformed inputs, and exercises real Torch CPU queries; physical CUDA coverage is conditional on available hardware.

#### Fixed (2026-10-09)

- ANOVA now rejects NaN and positive or negative infinity in observations passed to `f_oneway`, `cohens_f`, and `f_twoway`, consistently with the other maintained ANOVA procedures. This does not change the documented results for finite constant groups or insufficient residual degrees of freedom.
- Covariance estimators now honor explicit computation devices even when the input is a Torch tensor or CuPy array: `cpu` uses NumPy, `cuda` requires CuPy CUDA, and `torch` requires Torch CUDA. Unavailable explicitly requested backends raise. When both the estimator and global policy are `auto`, native input-backend behavior is retained. Scoring, Mahalanobis distances, and cross-validation use the fitted computation backend.
- Repaired the shared GLM `C` parameter tables in both languages so the rendered cells retain the complete penalty normalization, `C=0` behavior, and solver exceptions.

#### Fixed (2026-10-06)

- Fixed native generated Torch model-X device and seed handling: its local random generator and noise use the exact input tensor device, including CPU when CUDA is also available, and construction honors `random_state` without consuming the global Torch RNG. With the existing `None`-to-zero fallback, multiple Torch construction draws repeat the same noise; use an explicit integer seed for distinct, reproducible per-draw seeds. This does not imply cross-backend/device bitwise parity or repair the separate Lasso cache/device-routing limitations.

#### Improved (2026-10-09)

- Separated learner guidance from panel/model validation provenance and benchmark-dashboard maintenance instructions. Kept statistical assumptions, numerical warnings and comparison settings in the user pages; documented the custom RidgeCV training-row defect and its external-CV workaround before PR #249 fixed it for 0.3.0. Moved the historical PR78 dashboard plan to developer plans, replaced brittle current-data counts with generated-data links, and qualified older speed measurements by their actual workload and available provenance.

- Reorganized model tutorials into short, explained steps: imports and input meaning, data preparation, fitting, prediction, and result interpretation. Later inference, tuning, and GPU examples name and reuse their earlier setup where appropriate, instead of repeating whole datasets or CPU fits. English and Chinese pages follow the same progression while retaining model-specific formulas, assumptions, and limitations.
- Added explicit data preparation to examples that previously relied on undeclared variables. Executable-example tests now follow the same ordered blocks and declared setup dependencies as readers, while preserving the checks on predictions, inference, and cross-validation results.

#### Improved (2026-10-06)

- Tightened the UMAP missing-neighbor xfail to the exact finite graph signature; malformed graphs fail normally, and a correct UMAP repair reaches strict XPASS.

- Documented silent missing-row removal in ordinary formula-based LinearRegression/GLM prediction, including invalid finite LinearRegression scores after broadcasting. Added explicit complete-query/index-preserving checks and distinguished the penalized wrappers, which already reject missing prediction rows. PR #249 fixes the invalid LinearRegression score broadcasting; ordinary formula prediction can still drop missing query rows, so check query completeness before prediction.

- Rebuilt Poisson and shared CV learning examples as standalone seeded CPU workflows with held-out data, ordered/gapped folds and interpretation of expected counts, rate multipliers and uncertainty. Completed typed GLM constructor references and CV installed help, including family-specific controls and actual public methods.
- Corrected UMAP's reference cross-entropy to exclude self-pairs, defined TSNE's conditional/joint probabilities and feasible perplexity limits, and explained the host work involved in UMAP attraction-curve fitting. A finite TSNE embedding does not establish that an impossible perplexity target was met.
- Linked the complete kernel and spline API inventories directly from the shared reference. Strengthened known-defect regressions with independent numerical signatures and mutation checks so unrelated NaNs, finite corruption and runtime failures are not mislabeled as expected failures.

- Completed the Ridge, SCAD and MCP learner journeys with standalone CPU examples, explicit unpenalized intercepts, model-specific method/formula/weight references and runtime help. Clarified generated Torch model-X seed semantics and sampler-option dispatch, and warned that a rejected KernelPCA refit can leave mixed state. The custom RidgeCV training-subset defect described during documentation work is fixed by PR #249 in this release. RidgeCV retains the numerical range limits described above; KernelPCA failed-refit limitations remain.
- Expanded bilingual explanations, mathematical definitions and runnable examples for linear models, inference, feature selection, survival, smoothing and twelve unsupervised estimators. The [complete API index](README.md#complete-api-references) links constructor options, methods and fitted-result references.
- Clarified statistical inference targets, selection/tuning conditioning, weighting, resampling assumptions and available reporting methods. Connected Ridge, SCAD, MCP and Poisson explanations now distinguish their actual defaults and supported interfaces. Cox-family documentation identifies the difference between summed and row-averaged partial likelihood.
- Documented current limitations and practical checks for failed refits, nonfinite outputs, data geometry and device placement. The new [coordinate-scaling guidance](models/nonparametric/nonparametric.md) and [NMF guide](unsupervised/nmf.md) explain how very small measurement units can change current results, how to prepare and reuse a consistent scale, and how to map predictions or densities back. The [Ridge guide](models/regression/ridge.md) also shows training-derived centering for large coordinate offsets.
- Clarified that a shared estimator's `device="auto"` inherits the global device policy, and completed public constructor help. Retained benchmark reports are identified as historical measurements of their recorded source and hardware.
- In addition to documentation, docstrings and regression coverage, this PR fixes native Torch model-X random allocation and local-generator use, ANOVA nonfinite-input rejection, and covariance device selection. Other separately tracked numerical/state limitations remain unresolved; UMAP production code is unchanged. CPU examples and routing inspections do not establish physical-GPU execution or performance.

### Logistic classifier solver selection (PR #172, 2026-10-03)

#### Added

- `LogisticRegression` and `LogisticRegressionCV` now accept `solver="auto"`, `"irls"`, and `"lbfgs"`; `auto` preserves the existing IRLS route. This API is new in the upcoming 0.3.0 release and is not present in 0.2.5.
- L-BFGS retains the summed weighted Bernoulli objective plus `||coef||**2/(2*C)`, excludes the intercept from ridge, and preserves the legacy unpenalized `C=0` direct fit. Its implementation scales the entire objective by sample count or total weight, leaving the meaning of `C` unchanged.
- NumPy, CuPy, and Torch use backend-native fitting. GPU CV keeps batched IRLS for `auto`/`irls`; explicit L-BFGS uses float64 candidate fits on the selected backend and the same solver for the final refit. `gpu_cv_mixed_precision` affects only batched IRLS.
- Fitted `solver_`, requested/resolved summary output, transactional validation, and solver-specific convergence reporting make the executed route visible. Existing covariance inference is retained without correcting shrinkage or CV tuning uncertainty.
- Changed implementation: `statgpu/linear_model/wrappers/_logistic.py`, new `_logistic_solver.py`, `statgpu/linear_model/cv/_logistic_cv.py`, and the opt-in gradient-convergence hook in `statgpu/solvers/_lbfgs.py`. Regression coverage: `dev/tests/test_logistic_solver_contract.py`. These test definitions and CPU/Torch-CPU runs do not by themselves establish physical CUDA execution; final-release validation is described above.

```python
from statgpu.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], solver="lbfgs", device="cpu")
model.fit(X, y)
print(model.C_, model.solver_)
```

### Quantile solver and inference updates (PR #166)

#### Fixed

- **Public Quantile solver boundaries**: direct, CV, and low-level consumers now apply the same response/design shape, stopping, continuation-path, sample-weight, and solver-compatibility rules. Ordinary FISTA and the historical omitted/uniform-weight low-level L-BFGS behavior remain available where documented; FISTA-BB, ADMM, Newton, Proximal Newton, and L-BFGS-B reject Quantile before numerical work. The historical `quantile_cd_solver` implementation ignored `sample_weight` and could not reliably represent an unpenalized intercept, so the public name is retained only for import compatibility and raises when called; scalar SCAD/MCP fitting uses Proximal IRLS-CD.
- **Direct/CV input alignment**: Quantile response validation preserves the active backend and is shared by generic/typed direct fits and `PenalizedGLM_CV`; public low-level Quantile routes also reject non-finite `X/y` before numerical work. Malformed custom folds and zero fold weight mass are rejected before automatic alpha-grid numerical work; continuation scales and adaptive/group penalty ownership stay aligned with the fitted objective. Prediction and score reject malformed shapes instead of allowing NumPy broadcasting to return plausible but invalid results.
- **Standalone `QuantileRegression` inference correctness**: fitted residuals are centered at their empirical target-quantile so the bootstrap error distribution has empirical τ-quantile zero, then child refits solve the requested Quantile objective with backend-native batched IRLS/MM. Multifeature non-median child objectives are regression-checked against independent Quantile LP solutions, avoiding the earlier batched subgradient/FISTA path that could stop above the pinball optimum. Standalone non-uniform weighted inference is rejected because weighted kernel/bootstrap inference is not implemented; uniform weights retain the equivalent unweighted inferential target. The bootstrap is an i.i.d. residual bootstrap with exchangeable residual draws, not a heteroscedastic-robust wild/multiplier bootstrap.
- **Standalone inference lifecycle and controls**: bootstrap requires at least two resamples; invalid inference method/kernel/bandwidth/stopping/boolean/quantile controls raise before backend work; Hall-Sheather/Bofinger/Chamberlain bandwidth evaluation rejects `q ± h` outside `(0,1)`, non-finite/non-positive bandwidths, and non-finite/non-positive residual-density estimates at zero before covariance publication. Standalone `score()` returns the documented negative pinball loss with optional analytic weights. Failed fits do not retain partial inference results, and `gpu_memory_cleanup=True` recognizes the recorded `cupy` backend on success and failure paths.
- **Runtime/API documentation**: Quantile subgradient help matches the executable derivative, generic penalized runtime help names the public Quantile loss surface, and solver wrappers remain reload/import-order safe while preserving historical public-module identities.
- **Smooth/async Quantile FISTA backend parity**: NumPy, CuPy, and Torch use CPU-equivalent every-iteration best-accepted/objective-stability semantics on the backtracking L2/no-penalty Quantile route. Accelerator execution batches the accepted trial's objective/coefficient-change and L2-tracking scalars into the Armijo synchronization that already occurs for that trial, avoiding extra convergence or penalty-tracking synchronization. The asynchronous non-smooth Quantile CV path keeps deferred checks, caps Nesterov momentum at `0.5`, and halves the fixed step with a momentum restart only after two consecutive post-burn-in objective checks fail to make a tolerance-scale improvement. The adaptive control reuses the existing deferred synchronization and batches L1 penalty tracking into the same check, so it adds no host sync. For physical validation, the async weighted-L1 fixture now uses an independent SciPy/HiGHS linear-program optimum: the deterministic CPU generic-FISTA diagnostic is about `8.939e-4` above that convex optimum, so CPU-FISTA similarity is no longer treated as correctness. The unchanged `5e-4` threshold is measured against the LP objective. Accelerator Quantile sparse CV now uses the same maintained async/adaptive FISTA route for both per-fold L1/ElasticNet/Adaptive-L1 candidates and the selected full-data refit; previously those internal `model.fit()` calls reverted to generic pinball backtracking and could fail Armijo at a non-smooth kink. Ordinary direct Quantile FISTA behavior is unchanged. Physical schema v23 also uses independent HiGHS LP references for L1 CV fold/alpha validation scores, selected alpha, and the selected full-data penalized objective. On the deterministic fixture, CPU CV scores differ from the LP reference by up to `3.1335e-3`, so CPU CV is retained only as a diagnostic; the existing accelerator score and final-refit objective tolerances are unchanged.
- **Low-level scalar Quantile SCAD/MCP accuracy (Issue #169)**: the public low-level `fista_lla_path` route now delegates scalar Quantile SCAD/MCP to the same dedicated Proximal IRLS-LLA engine used by direct and CV fits. The previous fused fixed-step FISTA inner solve could stall above the convex weighted-L1 optimum of the non-smooth pinball surrogate while still reporting convergence; group penalties and warm-started/path-reporting low-level calls keep the fused FISTA-LLA engine. Scalar physical-gate schema v3 replaces the zero-versus-zero step-scale probe with a nontrivial converged fixture checked against an independent SciPy/HiGHS LP fixed point (observed gap about `4.6e-11`, tolerance `1e-8`), plus a dedicated budget-exhaustion warning case.
- **Low-level Quantile validation and warning seams**: adaptive/group adaptive weights and `AdaptiveL1Penalty` controls now reject complex inputs instead of silently truncating them; direct `QuantileLoss.irls()` rejects zero-feature designs; the single-step marked continuation path now solves the requested target alpha; the zero-budget fused LLA fallback no longer raises `UnboundLocalError`; and re-emitted Quantile convergence warnings point at the caller. Torch kernel-based sandwich inference keeps its covariance weights in the working tensor dtype instead of a hidden float32 default, restoring tight cross-backend bandwidth parity.

#### Validation

- Added focused regressions for Python-version collection safety, public solver aliases/reload idempotence, low-level shape/path/weight rejection, direct/CV response validation, prediction/score broadcasting guards, Torch response-backend preservation, standalone failure transactions/controls/cleanup, kernel-bandwidth domain failures, and an integrated `tau=0.2` batched-bootstrap check that distinguishes the requested quantile from the incorrect complementary-quantile direction.
- Extended CUDA regression coverage for non-uniform weighted L1 FISTA, standalone bootstrap execution, automatic Group SCAD/MCP direct/CV fits, weighted scalar low-level Quantile LLA accuracy against an independent LP fixed point, dedicated budget-exhaustion diagnostics, flat-IRLS behavior, and Group LLA derivative residency on CuPy/Torch.

### Quantile solver provenance reconciliation (PR #164 / Issue #163)

#### Fixed

- Penalized Quantile solver identity is now consistent with the algorithm that actually executes. L2/no-penalty `solver="auto"` resolves to ordinary Quantile IRLS, L1/ElasticNet remain on the maintained FISTA family, and SCAD/MCP resolve to the dedicated Proximal IRLS-CD implementation. `proximal_irls_cd` remains an internal resolved/executed provenance label rather than a public explicit `solver=` keyword.
- Incompatible explicit Quantile solver requests now fail before numerical dispatch or CV grid work instead of silently executing another algorithm. Direct fits, CV candidate/fold execution, and the selected full-data refit therefore agree on requested/resolved/executed solver identity.
- Non-median Quantile CV scoring preserves the caller's configured quantile level, including analytic validation weights. Typed `PenalizedQuantileRegression(quantile=q)` also preserves that level through clone-safe construction, adaptive-L1 initialization, and scoring.
- SCAD/MCP Quantile intercepts are optimized as unpenalized coordinates of the pinball objective. A fully flat LLA surrogate closes through the maintained full `QuantileLoss.irls()` kernel rather than continuing a diagonal approximation.
- Torch Quantile execution now keeps L2 penalty diagonals, IRLS warm starts, Proximal IRLS-CD epsilon/threshold/tolerance scalars, and fallback weights on the active tensor dtype/device. Warm starts use Torch-native cloning rather than the NumPy/CuPy copy path.

#### Validation

- Exact clean numerical source `2de971402004efc703dc98f510942db3980988e4` passed the frozen schema-v1 physical gate on Tesla P100-SXM2-16GB with CuPy 13.6.0, Torch 2.0.0+cu117, NumPy 1.24.2, and Python 3.9.16. All **12/12** direct/CV CuPy and Torch CUDA cases passed with concrete `cuda:0` provenance and the expected IRLS/FISTA/`proximal_irls_cd` identities.
- Frozen parity tolerances were not loosened. The maximum L2 coefficient/intercept error was `6.938893903907228e-15` against `2e-5`, the maximum L2 CV-score error was `7.965850201685498e-15` against `2e-5`, and the maximum SCAD penalized-objective error was `1.4085439563257807e-06` against `2e-4`.
- The canonical exact-source artifact is `dev/reviews/pr164_quantile_solver_provenance_gpu.json`. Commit `230eaebbd4dfaf090e84011fe0eb190339389411` adds only that artifact directly above the validated numerical source. This records PR #164 acceptance at that source. Later releases require their own validation; the artifact does not establish acceptance of changed numerical code or validators.

### Quantile IRLS penalty contract (PR #162 / Issue #161)

#### Fixed

- Low-level `QuantileLoss.irls()` now accepts only no penalty or L2. ElasticNet, L1, SCAD/MCP, group/adaptive penalties, and unknown penalty objects fail before numerical iteration instead of allowing an IRLS solve that applies only the smooth L2 component of a declared non-smooth objective.
- The public estimator contract is unchanged: explicit Quantile `solver="irls"` remains a maintained L2/no-penalty route, while non-smooth penalties continue through FISTA-family or dedicated Quantile non-convex algorithms.
- The IRLS docstring now matches the executable low-level contract. Unsupported direct low-level calls receive a precise fail-closed error rather than a plausible-looking partial-penalty result.

#### Validation

- Added focused regressions for direct ElasticNet/L1/SCAD/unknown-penalty rejection before the linear solve, retained unpenalized/L2 execution, positive global analytic-weight scale invariance, the existing estimator-level fail-closed boundary, and Torch CPU L2 backend preservation.
- This repair removes an unsupported route and does not alter the maintained None/L2 IRLS numerical algorithm, so it does not by itself create a new physical-CUDA acceptance requirement.

### Weighted explicit Newton/L-BFGS GLM fits (PR #151 / Issue #150)

#### Changed

- Ordinary `GeneralizedLinearModel` explicit `solver="newton"` and `solver="lbfgs"` now accept genuine non-uniform analytic `sample_weight` on supported GLM rows without silently substituting IRLS/FISTA or changing an explicit NumPy/CuPy/Torch execution request.
- Maintained GLM smooth solvers use one normalized analytic-weight objective across the complete solve. Review/fix passes made effective-uniform classification scale- and permutation-invariant, normalize before execution-dtype casting, preserve fractional weights for integral designs, avoid raw float32-weight-sum overflow before normalization, and retain the exact solver-prepared weight identity for ordinary post-fit diagnostics/inference. On Newton/L-BFGS M-estimation routes, that identity is rescaled to an equivalent mean-one representation before covariance evaluation, so positive global analytic-weight rescaling preserves coefficients and nonrobust/HC0/HC1 inference up to numerical solver tolerance without redefining unrelated IRLS/FISTA weighting semantics. `ParameterInferenceResult` also publishes the actual M-estimation `cov_type` when a constructor omitted that standard result field. L-BFGS preserves visible line-search failure while allowing only a bounded floating-point Armijo acceptance when both the requested objective decrease and parameter displacement are below maintained numerical resolution.
- `GammaRegression(link="inverse_power")` now uses a loss-owned smooth-domain contract for explicit Newton/L-BFGS rather than a blanket no-intercept rejection. A domain-capped tiny L-BFGS step does not count as convergence by itself; when a domain-aware quasi-Newton Armijo search is exhausted, the solver retries with steepest descent and a freshly computed domain cap, then fails closed if recovery also fails before gradient convergence. Feasible no-intercept designs fit normally, while infeasible, uncertifiable, or domain-pinned fits fail visibly.
- The same inverse-Gamma domain contract is closed through penalized L2 Newton/L-BFGS and `PenalizedGLM_CV`: framework warm starts are reused only when domain-feasible, validation scoring uses the declared `inverse_power` objective, and selected full-data refits preserve the link. Typed `PenalizedGammaRegression` preserves historical `loss_kwargs["link"]` precedence while remaining sklearn-clone safe.
- Existing `solver="auto"`, IRLS/FISTA, explicit smooth-solver `C`, Ordered GLM, standalone LogisticRegression behavior, and non-inverse Gamma links are preserved unless explicitly covered above. Ordinary GLM fits continue to record the solver/backend/device that actually completed successfully.
- Ordinary explicit Newton/L-BFGS refits are transactional across validation, optimization, inference, and provenance publication. A failed refit atomically restores the complete last successful estimator state instead of exposing old coefficients with new-attempt metadata; a first failed fit remains unfitted. The previous successful execution provenance remains available without advertising the failed attempt as new evidence.

#### Validation

- Historical hosted/physical evidence remains exact-source-specific. Numerical/validator source `9eb39cee2c0e691160f528ab687b7379d26f3e42` passed all seven hosted PR workflows and the Tesla P100 schema-v5 gate; the complete CPU suite on that source reported **3395 passed / 831 skipped / 0 failed**. Later review changed production inference/result provenance, so neither that hosted run nor schema v5 accepts later source.
- The Tesla P100 schema-v3 artifact at `c6781cb6a2e1fe500f325e832d23cdc80a99b564`, schema-v4 inverse-Gamma artifact at `9ef5b34ffc8abf133bc6262e98a855d5efe37d3c`, and schema-v5 artifact at `9eb39cee2c0e691160f528ab687b7379d26f3e42` remain immutable **historical exact-source evidence only**. Their retained files are `dev/reviews/pr151_glm_weighted_explicit_solvers_gpu.json`, `dev/reviews/pr151_inverse_gamma_domain_gpu_v4.json`, and `dev/reviews/pr151_final_gpu_v5.json`.
- Schema v6 is an immutable **failed historical validator**. Its float32-overflow inference fixture accidentally cast `X` and `y` to float32 in addition to the intended float32 analytic weights, adding an unrelated float32-design CuPy L-BFGS cross-backend parity requirement. Issue #160 tracks that separate precision question; PR151 does not weaken its frozen tolerances or broaden generic L-BFGS to make that fixture pass.
- Corrected schema v7 records the historical physical CUDA acceptance for PR #151. Exact clean numerical/validator source `0302262ef8242b31f2f17b7835b6c08aeda89904` passed the complete v3→v4→v5→v7 chain on Tesla P100-SXM2-16GB with CuPy 13.6.0, Torch 2.0.0+cu117, NumPy 1.24.2, and Python 3.9.16, recording `status: success` and `source_clean: true`. The canonical artifact is `dev/reviews/pr151_final_gpu_v7.json`; the earlier accepted `e1cbf3756d269a6073461de8331da11c6235fb4e` schema-v7 artifact is preserved as `dev/reviews/pr151_final_gpu_v7_e1cbf375.json`. Numerical parity with the earlier run is unchanged: the maximum review-closure error remains `9.719913152128612e-09` at the same penalized/Newton/Torch p-value comparison. Commit `29b8e8056fb9bdfb34e2b76982d433c0be3d844f` records the final evidence directly above the validated numerical source; that historical documentation update did not alter the validated implementation. PR #151 is merged. PR #165 subsequently documented the float32 precision behavior investigated in Issue #160; it did not replace the production L-BFGS algorithm. These artifacts do not certify the final 0.3.0 source.

### Backend-native Gaussian residual bootstrap (PR #147 / Issue #145)

#### Changed

- The existing unweighted Gaussian `residual_bootstrap` row now executes on the backend and concrete device recorded by the successful fit: NumPy/CPU, CuPy CUDA, or Torch CUDA. The statistical procedure is unchanged: residuals are resampled around the fixed fitted values and every bootstrap draw refits the same penalized Gaussian estimator.
- For a fixed `bootstrap_random_state`, NumPy, CuPy, and Torch consume the same deterministic residual-index schedule. GPU fits keep response construction and child optimization on the selected GPU device; only the small index/control data, the already-established small parent parameter snapshot used to reconstruct fitted values, and the completed reporting snapshots cross the host/device boundary.
- Bootstrap child refits preserve the fitted penalty family/object, `alpha`, `l1_ratio`, penalty kwargs, intercept semantics, solver/stopping controls, `lipschitz_L`, applicable loss kwargs, and SCAD/MCP LLA controls. Child inference remains disabled so bootstrap never recursively invokes itself.
- `PenalizedGLM_CV` continues to run bootstrap only once, after selecting the penalty and refitting on the full dataset. Results therefore condition on the selected penalty; tuning-selection uncertainty is not corrected.
- Weighted residual bootstrap, robust/HC bootstrap, HAC/block bootstrap, non-Gaussian bootstrap, Cox bootstrap, and batched bootstrap refits remain outside this procedure and fail closed rather than switching to a different resampling method.

#### Validation

- Hosted coverage includes deterministic NumPy preservation, L1/ElasticNet/SCAD/MCP refit ownership, exact-device child contexts, formula and typed-wrapper consumers, failure transactions, CV final-refit-only behavior, runtime-installer idempotence, and the unchanged fail-closed rows.
- `dev/benchmarks/validate_gaussian_residual_bootstrap_gpu.py` is the frozen schema-v1 physical CUDA gate for CuPy/Torch L1 and ElasticNet parity, deterministic same-draw identity, concrete device provenance, Torch↔CuPy heterogeneous-container routing, and no CPU numerical fallback. PR #147 was merged on 2026-09-12. This describes the feature validator, not a completed validation of the final 0.3.0 release source.

### Penalized GLM inference contract (PR #142)

#### Changed

- Generic and typed penalized GLMs now recommend public `inference_method="auto"` while preserving specialized sparse-Gaussian wrapper defaults. Successful inference-enabled fits expose the requested method, resolved numerical method, reported result method, inferential target, and penalty/tuning conditioning instead of inferring those semantics from a single `inference_method_` string.
- Supported smooth non-Gaussian L2/no-penalty rows use fixed-penalty `m_estimation` with nonrobust/HC0/HC1 covariance. Positive L2 inference targets the penalized estimating equation; zero-penalty aliases target the unpenalized population parameter. Non-Gaussian L1/ElasticNet coefficient inference now fails closed instead of publishing the historical partial sandwich approximation.
- `inference_method="bootstrap"` is intentionally narrow: it resolves only to an unweighted Gaussian residual bootstrap with `cov_type="nonrobust"` for supported sparse penalties. PR #147 extends this procedure to the fit-recorded NumPy, CuPy, or Torch backend. Refits preserve the fitted penalty family and tuning (`alpha`, `l1_ratio`, penalty kwargs, intercept, solver/stopping and LLA controls), require at least two draws, and never silently transfer CuPy/Torch fits to CPU. Weighted, robust/HC/HAC, non-Gaussian, Cox, and Gaussian L2 residual-bootstrap requests remain unsupported.
- Supported non-Gaussian M-estimation now treats the fit-recorded numerical backend and concrete device as authoritative. CuPy and Torch inference stays on that backend/device, including heterogeneous input containers; there is no inference-time re-resolution that can silently switch a fit to another numerical engine.
- `PenalizedGLM_CV` exposes the same inference controls. CV folds/candidates remain estimation-only; inference runs once on the selected full-data final refit. Successful CV results report conditioning on the selected penalty with `penalty_selection_adjusted_=False`, so tuning-selection uncertainty is not implied to be corrected.
- The generic/typed GLM pages now distinguish these current methods from reserved-but-unimplemented robust/HAC/bootstrap combinations and keep EN/CN support claims aligned.
- Genuine non-uniform analytic `sample_weight` is now supported by the shared Newton solver for losses that expose weighted value/gradient/Hessian primitives. Weighted smooth non-Gaussian L2 fits no longer need an inference-only FISTA override: public `solver="auto"` keeps the canonical dispatch and applicable Logistic/Poisson L2 rows resolve to backend-native Newton across direct fit, CV, selected final refit, and M-estimation inference. The public request remains `auto`; the executed solver is recorded separately. PR #151 subsequently made effective-uniform classification invariant to common weight rescaling and permutation; see its weighting details above.

#### Validation

- Hosted validation covers API/clone compatibility, formula/categorical/missing-row alignment, failure transactions, NumPy/CuPy/Torch provenance guards, deterministic residual-bootstrap ownership, final-refit-only CV inference, and targeted external references.
- Exact clean numerical source `db448d718f523eacf97bcb3c419e376c9812362d` passed the maintained Tesla P100 schema-v4 physical validator with CuPy 13.6.0, Torch 2.0.0+cu117, NumPy 1.24.2, and Python 3.9.16. The four maintained family/weight rows covered Logistic explicit-FISTA unweighted plus canonical `solver="auto"` → Newton for Logistic weighted and Poisson weighted/unweighted. Every CuPy, Torch, Torch→CuPy, and CuPy→Torch route passed the unchanged coefficient/intercept (`2e-6`) and inference (`1e-5`) tolerances with concrete `cuda:0` provenance; the retained artifact is `results/pr142_penalized_glm_inference_gpu/pr142_penalized_glm_inference_gpu.json` (15944 B).
- The PR #142 acceptance record includes passing hosted CI and independent review on its own reviewed source. The P100 artifact remains anchored to the immutable numerical source above; subsequent changelog/evidence-only closure reuses it by explicit approval, while any numerical, validator, backend, solver, inference, or tolerance change would require a fresh physical run.

### Node-wise Lasso inference tuning contract (PR #139)

#### Changed

- Added public `nodewise_alpha=None` control for debiased sparse-Gaussian inference on `PenalizedGeneralizedLinearModel`, `PenalizedLinearRegression`, `Lasso`, `ElasticNet`, `LassoCV`, and `ElasticNetCV`. Explicit finite positive real values are authoritative; `PenalizedGLM_CV` keeps this control at its current unsupported/default boundary.
- Replaced the response-scale-dependent internal node-wise penalty with a response-independent design rule. Omitted `nodewise_alpha` now resolves from the standardized centered/weighted working design as `sqrt(2 * log(max(p, 2)) / n_nodewise)`, with a Kish-style effective sample size under non-uniform analytic weights.
- Node-wise precision is solved in standardized design coordinates, validated by a full KKT certificate, and back-transformed as `D^{-1} Theta_Z D^{-1}`. The analytic `p=1` case does not consume a node-wise penalty.
- Added fitted/provenance reporting for requested/resolved node-wise tuning and cache keys that distinguish numerical node-wise precision identity from the main-model tolerance. Explicit/CV/formula/GPU paths therefore cannot silently reuse stale precision from another node-wise request.
- Preserved the existing debiased inferential target, covariance/reference-distribution semantics, active prediction coefficients, and post-selection OLS behavior. Legacy response-scale-dependent node-wise tuning is intentionally not exposed as a new public mode.

#### Validation

- Added deterministic fixed-design simulation evidence comparing the old response-dependent rule, the new automatic rule, and a representative explicit rule at `n=320`, `p=18`, 120 repetitions and two response-noise scales. The omitted/default rule keeps its node-wise penalty constant across noise scales while the legacy rule changes by exactly 8×; the new rule is non-inferior on coverage error and simultaneously improves average interval width in that migration design.
- Hosted tests cover low-level scalar/array node-wise tuning, KKT validation, cache isolation, public API/clone/refit behavior, CV selected-final-refit inference, formula routing, unavailable GPU/Torch-CUDA behavior, and the committed migration simulation artifact.
- `dev/benchmarks/validate_nodewise_alpha_gpu.py` is the maintained physical CUDA gate for CuPy/Torch execution. It requires both frameworks on a real CUDA device and checks automatic/explicit node-wise tuning, NumPy parity, response-scale invariance of the automatic penalty, weighted rescaling, and cross-container provenance.

### Post-selection OLS inference API (PR #138 / Issue #137)

#### Changed

- Added canonical hardware-neutral `inference_method="post_selection_ols"` for sparse Gaussian `Lasso`, `ElasticNet`, and the public generic `PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")` surface. The historical `cpu_ols` / `gpu_ols` strings remain one-cycle compatibility aliases and emit `FutureWarning`; `LassoCV` keeps `cpu_ols_inference` / `gpu_ols_inference` at its CV-specific compatibility boundary.
- Statistical method selection is now orthogonal to hardware. Explicit `device="cpu"`, `"cuda"`, and `"torch"` remain authoritative even for heterogeneous input containers, while genuine AUTO may preserve native CuPy/Torch-CUDA input. LassoCV now uses one resolved backend for folds and the selected-alpha final refit and preserves concrete CuPy device affinity.
- `post_selection_ols` refits an unpenalized OLS/WLS model on the selected active set using the fit-recorded NumPy/CuPy/Torch backend, but keeps the original penalized `coef_` / `intercept_` as the prediction model. Rank-deficient active designs use effective rank for residual degrees of freedom and the same Moore-Penrose/SVD design geometry for the refit and covariance bread.
- Post-selection diagnostics now belong to the refit rather than overwriting penalized-fit R-squared, F, log-likelihood, AIC, or BIC. Summary output labels effective residual DoF separately and preserves the penalized fit diagnostics on the estimator.
- Formula-based sparse fits retain categorical/interaction/missing-row semantics and align array-like sample weights to Patsy-retained rows. Rejected or post-fit-inference-failed refits clear stale result-bearing state and fail closed.
- Unified sparse-Gaussian analytic weights across direct NumPy/CuPy/Torch fits and weighted LassoCV: weighted centering precedes the equivalent `sqrt(w * n / sum(w))` transform; default alpha grids, fold objectives, validation MSE, and final refits use the same convention. Positive constant weights are the exact unweighted CV problem.
- Unified debiased inference around the same centered average-loss working problem. Intercept-inclusive simultaneous max-|Z| inference now includes the original-coordinate intercept influence, and successful refits clear stale simultaneous/precision state before publication.

#### Validation

- Hosted coverage includes Python 3.9/3.12, Torch 2.0 CPU, the full CPU suite, maintenance compatibility, static/documentation checks, release packaging, and benchmark-frontend contracts.
- `dev/benchmarks/validate_post_selection_ols_gpu.py` is the final schema-v7 22-case CuPy/Torch physical gate. Historical P100 artifacts remain evidence only for their recorded source SHAs; PR #138 has since merged. These historical checks are not acceptance of the final 0.3.0 source. No GPU speedup claim is made.

### Penalized solver API cleanup (PR #135)

#### Changed

- Made backend-neutral `solver` the authoritative direct-fit solver selector across the public penalized estimator family.
- Retained deprecated `cpu_solver` for compatibility; removal is reserved for a future breaking release. Meaningful legacy use emits `FutureWarning`; omission remains quiet and preserves historical behavior.
- Added stage-specific `LassoCV.cv_solver` for fold/path execution and fitted `cv_solver_` for the algorithm that actually ran. `solver` continues to control the final full-data refit.
- Preserved legacy CUDA/Torch FISTA behavior during the migration window instead of silently remapping `cpu_solver` into a different direct numerical path.

#### Validation

- Added constructor/signature/get-params/set-params/clone coverage across maintained sklearn versions and caller-facing warning-location checks.
- Added CV regressions proving `cv_solver` belongs to folds/path while `solver` belongs to the final refit.
- Updated English and Chinese migration, solver, CV, and model documentation to distinguish behavior-preserving API migration from an intentional solver change.

### Gaussian linear-model backend-native inference (PR #129 / Issue #127)

#### Improved

- Migrated maintained Gaussian linear-model covariance, standard-error, statistic, p-value, and confidence-interval numerical work to the executed NumPy/CuPy/Torch backend while preserving the established final NumPy reporting snapshot.
- Routed normal/Student-t reference-distribution work through the maintained shared inference layer, including stable df=1/df=2 extreme-tail handling. Missing or invalid executed-backend provenance now fails closed instead of silently routing inference through NumPy.
- Preserved Ridge/L2 average-loss scaling, analytic weights, `RidgeCV` selected-final-refit inference, formula row alignment, rank-deficient designs, multi-target inference, and the public NumPy result/reporting contract.
- Added exact-backend provenance checks and no-host-transfer regressions for direct Gaussian and Ridge/CV consumers.

#### Validation

- The [final acceptance record](https://github.com/TheHiddenObserver/statgpu/pull/129#issuecomment-5479959733) reports seven passing hosted workflows and final review on exact numerical source [`8d9f422e2fcf648e412838747b9ab5881b3fb7ef`](https://github.com/TheHiddenObserver/statgpu/commit/8d9f422e2fcf648e412838747b9ab5881b3fb7ef). Both physical CUDA validators exited 0 from a clean checkout on Tesla P100-SXM2-16GB `cuda:0`: the backend-native matrix used schema 5, CuPy 13.6.0 and Torch 2.0.0+cu117; the schema-1 CuPy rank-recovery validator passed 5/5 cases. Coverage included covariance/BSE/statistic/p-value/CI parity, weighted/rank/multi-target/small-df cases, backend/device provenance, and `RidgeCV` final-refit inference.
- That record identifies the historical artifacts as `results/pr129_gaussian_gpu_8d9f422e/gaussian_backend_native_8d9f422e.json` and `results/pr129_gaussian_gpu_8d9f422e/gaussian_cupy_rank_recovery_8d9f422e.json`. These results apply to the cited source, not to later versions; no GPU speedup claim is made.
- PR #129 was subsequently merged on 2026-08-31 as [`b17c8a6c161065bfbe9efa9f60d0fad69a464773`](https://github.com/TheHiddenObserver/statgpu/commit/b17c8a6c161065bfbe9efa9f60d0fad69a464773). Earlier pending-acceptance records describe their own checkpoints rather than the final state.

## 0.2.5 — 2026-08-26

### Panel Tier-1 Stage C covariance and release closure

- Shipped HC0/HC2/HC3, legacy HC1 where defined, one-/two-way clustering with optional `group_debias`, Driscoll-Kraay with Bartlett/Parzen/QS kernels, RandomEffects robust/HC inference, and legacy row-order HAC with ordered-categorical chronology.
- Added transactional panel fits, row-preserving formula prediction, classical Hausman/pooling-F/Breusch-Pagan LM diagnostics, and overflow-safe fit statistics.
- Hardened fixed-effect nuisance-rank residual df, rank-deficient HC2/HC3 behavior, extreme-scale covariance arithmetic, level prediction, panel diagnostics, Fama-MacBeth rank handling, and concrete-device CuPy/Torch affinity.
- Kept ordinary panel workloads on vectorized GPU paths while reserving exact accumulation for genuine cancellation risk; maintained 10k-row CuPy two-way covariance dropped from roughly 1000 s to about 1.3 s on the recorded Tesla P100 benchmark workload.
- Final release acceptance on numerical source `697de113` passed all 12 maintained physical runners; artifacts are retained under `results/pr126_release_697de113/`.

### Distribution inference and packaging

- Fixed CuPy inverse Beta/Gamma LUT cache tuple ordering, restoring Student-t/Beta/F/Gamma/chi-square inverse quantiles and downstream confidence-interval widths.
- Released the corresponding panel/distribution regression coverage and universal `py3-none-any` package artifacts for `0.2.5`.

<a id="unreleased--2026-08-09"></a>
## 2026-08-09 — Historical Panel development record

The related Stage C capabilities were subsequently released in 0.2.5; the entries below retain the earlier development context.

### PR #126 — Panel Tier-1 Stage C covariance

- Finalized panel Tier-1 Stage C covariance and inference across `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, and `FamaMacBeth`.
- Added HC0/HC2/HC3, clustered, two-way clustered, Driscoll-Kraay, robust RandomEffects inference, stable fixed-effect level prediction, and hardened Hausman/pooling-F/Breusch-Pagan diagnostics.
- Hardened the shared panel numerical layer against extreme float64 dynamic range, recoverable cancellation, subnormal designs, singular/rank-deficient systems, unsafe CuPy scatter reductions, and Torch CUDA SVD driver instability.
- Preserved existing HC1/legacy HAC/public coefficient inference where applicable while making unsupported or numerically unresolved covariance cases fail closed.
- Completed exact-source physical CUDA validation and retained the numerical evidence under the corresponding `results/pr126_*` artifacts.

### PR #122 — Panel Tier-1 diagnostics Stage B

- Added structured `fit_statistics_` for within/between/overall and adjusted R², classical model F, pooling F, one-way entity Breusch-Pagan LM, and classical FE-vs-RE Hausman diagnostics.
- Preserved Stage-A coefficient inference and legacy df/R² attributes while adding NumPy/CuPy/Torch, formula-row-alignment, linearmodels-definition, and physical-GPU evidence.

### PR #121 — CuPy inverse-quantile LUT correctness

- Fixed CuPy `betaincinv` / `gammaincinv` lookup-table cache ordering and restored public CuPy Student-t, Beta, F, Gamma, and chi-square inverse quantiles.
- Validated the unchanged two-line numerical repair on Tesla P100 and added maintained public-distribution plus downstream Panel inference regressions.

### PR #119 — Panel Tier-1 shared framework Stage A

- Added internal `BasePanelModel`, shared panel index/balance metadata, and the shared residual-based covariance/inference substrate without expanding public diagnostics beyond Stage A.
- Migrated the maintained panel estimators to the shared lifecycle while preserving formula, prediction, summary, fixed-effect recovery, RandomEffects GLS, Fama-MacBeth chronology, backend output, and no-fallback behavior.
- Added pre-refactor golden regression coverage and Python 3.9 + Torch 2.0 CPU execution for shared panel metadata/covariance/inference tests.

Stage B diagnostics and Stage C covariance expansion remain pending under Issue #93; this Stage-A refactor does not advertise them as public capabilities.

### PR #116 — Torch LogisticRegressionCV strict-CUDA repair

- Fixed the maintained Torch strict-CUDA `LogisticRegressionCV` failure in the batched GPU IRLS path. Mixed-precision CV now allocates parameters and ridge diagonals in the active working dtype, and coefficient/intercept paths remain backend-native through validation scoring.
- Added regression coverage for float32 and float64 CV, weighted/unweighted execution, intercept/no-intercept paths, and the full selector consumer. A dedicated Python 3.9 + Torch 2.0 CPU CI job prevents the optional Torch regression suite from silently skipping.
- Physical validation ran on exact numerical implementation head `e6e4846b06604ed53e65fc9afd9054bd5777098f` using Tesla P100-SXM2-16GB, Python 3.9.16, PyTorch 2.0.0+cu117 / CUDA 11.7, and CuPy 13.6.0. All four focused Torch CUDA cases selected the same `C=0.2` as the CPU reference; the largest mean-loss difference was below `6.2e-8`, with the float64 path agreeing to machine precision.
- The canonical six-family rerun recorded all 18 statgpu NumPy/CuPy/Torch backend rows as successful, with zero failed candidates/folds and converged final refits. `LogisticRegressionCV` selected `C=0.1` on NumPy, CuPy, Torch, and sklearn; the Torch/NumPy validation-loss difference was below `4.7e-8`.
- The historical pre-fix P100 source remains immutable and registered. The post-fix exact-head source is registered separately from `results/pr116_p100/cv_benchmark_pr116_p100.json`; `focused_validation.json` remains validation-only evidence rather than dashboard timing data.

Related: Issue #112 and pull request #116.

## 0.2.4 — 2026-08-06

### Logistic regression and GLM correctness

- Corrected arbitrary-link Binomial IRLS Fisher weights, working responses, line-search objectives, backend-native warm starts, and quadratic-penalty validation.
- Hardened direct `LogisticRegression` validation, transactional refits, convergence reporting, integer hard predictions, single-column response handling, and finite decision thresholds.
- Unified fitted logistic likelihood diagnostics across NumPy, CuPy, and Torch with the registered stable `LogisticLoss` objective. Likelihood, AIC, BIC, pseudo-R², and convergence remain available independently of covariance inference.
- Kept confusion-matrix metrics available for one-class targets while retaining explicit class-support errors for ROC-AUC and average precision.
- Kept analytic weights device-native on CuPy/Torch fits and corrected weighted IRLS curvature, likelihood, dispersion, and sandwich-inference semantics.
- Standardized GLM analytic-weight behavior across fitting, line search, diagnostics, and covariance. Globally rescaling analytic weights does not change fitted parameters or reported diagnostics.
- Added backend-native response-domain, real-valued, finite, shape, and length validation for scalar GLMs, including penalized and CV entry points.
- Aligned formula sample weights after Patsy row filtering and corrected weighted Gaussian FISTA centering.

### Cross-validation, inference, and estimator contracts

- Made `RidgeCV`, `ElasticNetCV`, and `LogisticRegressionCV` failure-safe: stale state is cleared before fitting and selected parameters are published only after the final full-data refit succeeds.
- Preserved explicit Torch/CuPy requests and pinned `device="auto"` final refits to the backend selected during cross-validation.
- Updated Logistic and Elastic Net default regularization grids to incorporate analytic weights and satisfy integer-weight row-replication equivalence.
- Preserved declared validation losses and analytic weights in penalized CV; programming, shape, CUDA OOM, and device errors are no longer converted into candidate `NaN` values or unrelated MSE fallback.
- Completed standalone `ElasticNet` and final-refit `ElasticNetCV` inference across NumPy, CuPy, and Torch. Fold models remain estimation-only.
- Corrected ElasticNet/Ridge scaling documentation: under the shared average-loss convention, `ElasticNet(alpha, l1_ratio=0)` matches `Ridge(alpha)`.
- Made public finite-input guards, cloning, sklearn tags, nested `set_params`, and fitted-state invalidation transactional, including legacy scikit-learn clone identity checks.

### Solver and backend safety

- Corrected the solver matrix so Newton, L-BFGS, and L-BFGS-B reject unsupported non-smooth penalties rather than optimizing only the smooth component.
- Removed the incorrect Euclidean-prox Newton shortcut. Smooth L2/no-penalty objectives retain Newton; non-smooth proximal-Newton requests delegate visibly to backend-native FISTA until a Hessian-metric proximal solver exists.
- Narrowed Armijo, linear-solve, CV-grid, and inference fallbacks to recognized numeric or rank failures. CUDA OOM, device, index, contract, and unrelated runtime failures propagate.
- Normalized warm starts for FISTA, Newton-family, L-BFGS-family, and ADMM solvers to the preprocessed design backend, device, and dtype.
- Completed ADMM's legitimate Cholesky fallback and hardened L-BFGS-B directions, backend-native bounds, and NaN-bound validation.
- Added a centralized, observable Torch compile policy: eager remains the default for unset, `auto`, and `disable`; `default` and `reduce-overhead` are explicit opt-ins. Only the known CUDA Graph output-lifecycle failure becomes a permanent eager fallback.
- Removed the package-initialization cycle between `statgpu.glm_core` and the Cox loss export by lazily exposing `CoxPartialLikelihoodLoss`; fresh-interpreter imports no longer require a particular order.

### Documentation and release preparation

- Reconciled the English and Chinese LogisticRegression, ElasticNet, cross-validation, solver-algorithm, and solver/penalty documentation with the maintained implementation.
- Removed unsupported universal GPU speedup, backend-threshold, and coefficient-tolerance claims. Performance guidance now requires workload-specific benchmarking.
- Documented ownership boundary between maintained pytest coverage and manual physical-GPU diagnostics.
- Bumped package metadata to `0.2.4` and added `.github/releases/v0.2.4.md` as the authoritative GitHub Release body.

### Validation

- The final PR #87 implementation head passed 2239 tests with 719 skipped, static and documentation contracts, Python 3.9–3.12 regression jobs, scikit-learn 1.2.2/1.3.2/latest compatibility, and release-package validation.
- Physical NVIDIA validation passed on the unchanged numerical implementation: RTX 4090 with PyTorch 2.8.0+cu128 passed the selected compile/CUDA Graph matrix 9/9 and runtime assertions; Tesla P100 with CuPy 13.6.0 passed the corresponding runtime assertions.
- The focused release PR changes version metadata and release-facing documentation only. Exact release-head hosted gates must pass before tag `v0.2.4` is created.

## 0.2.3 — 2026-08-04

### Survival analysis

- Completed CoxPH Phase 1 with Breslow, Efron, and Exact ties; delayed entry
  and `(start, stop]` counting-process data; shared-coefficient stratification;
  subject identifiers; and `Surv(start, stop, event)` formula input.
- Added shared NumPy, CuPy, and Torch-CUDA risk-set primitives for objectives,
  gradients, information matrices, and baseline estimation. Exact tied-event
  partitions use backend-native dynamic programming.
- Extended `CoxPHCV` held-out partial likelihood to all supported tie methods,
  delayed entry, start-stop rows, strata, and subject-grouped folds.
- Hardened Cox inference, centered risk-set numerics, log-domain baseline prediction,
  formula NA alignment, singular-information handling, CV cache identity, fold eligibility,
  selected-penalty full-data refit, and failed-fit state resets.
- Hardened L1, L2, Elastic Net, SCAD, and MCP penalized Cox estimation; removed
  the unidentified intercept; corrected Cox-specific warm starts; and made the Torch Efron
  value, gradient, and Hessian paths native.

### Cross-validation and grouped penalties

- Requested CoxPHCV two-stage or successive-halving controls execute one explicit exhaustive
  full-precision candidate pass, preserving deterministic selection while avoiding repeated
  complete-grid fitting.
- One-shot `CoxPHCV.cv_splits` iterators are reusable across repeated fit, scikit-learn clone,
  parameter reconstruction, and pickle.
- Public Group Lasso and Adaptive Group Lasso use the generic loss-gradient and exact
  group-proximal path consistently across supported backends.

### Validation and packaging

- Hosted workflow #960 passed on final reviewed head
  `f05a44ad363b46612e956e137e2f00d040765acb`: documentation, static, full CPU,
  and Python 3.9–3.12 regression jobs all passed; the complete CPU suite reported
  1881 passed and 662 skipped.
- The final exact-head physical-GPU promotion artifact is published as
  [schema-3 evidence](https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4).
  It records 134/134 passing checks, zero child and nested return codes, empty
  gate-failure arrays, clean source state before and after execution, and SHA-256
  `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- The package version is now `0.2.3`. Release-package validation checks version consistency,
  builds the pure-Python wheel and sdist, runs `twine check`, validates artifact contents,
  and smoke-installs both distributions in clean environments.

## Earlier history

Detailed entries through 2026-08-03 are retained in
[the archived changelog](changelog-history-through-2026-08-03.markdown).
