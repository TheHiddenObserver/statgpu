# Canonical cross-validation benchmark source contract

The dashboard accepts cross-validation data only from a current, machine-readable source that validates against `dev/benchmarks/cv_source_schema.json` and the semantic checks in `dev/benchmarks/cv_source.py`.

## Initial model package

The first canonical source must contain exactly one representative case for each maintained family:

- `RidgeCV`;
- `LassoCV`;
- `ElasticNetCV`;
- `LogisticRegressionCV`;
- `PenalizedGLM_CV`;
- `CoxPHCV`.

Additional cases may be added in later source versions, but the initial registration is reviewed as one aligned package.

## Timing contract

Every successful run retains three distinct measurements:

- `cv_evaluation_ms`: candidate-by-fold evaluation, excluding the final full-data refit;
- `final_refit_ms`: the selected configuration fitted on all training observations;
- `total_fit_ms`: the complete public `fit` operation, including orchestration overhead.

The benchmark runner must measure these regions directly. It must not derive the first two fields by subtracting rounded aggregate values. GPU measurements synchronize the resolved backend immediately before and after each measured region. Transfer-inclusive and device-only timings are separate protocols and may not be mixed within a comparison.

## Backend disposition

Each representative case must explicitly record a statgpu disposition for `numpy`, `cupy`, and `torch`:

- `success` only when the run was executed and measured;
- `unavailable` when the required runtime or physical device is absent;
- `unsupported` when the maintained API does not implement that combination;
- `failed` when an attempted maintained combination raises or produces invalid evidence.

A non-success row contains a reason and no timing, score, selection, or convergence measurements. This prevents unavailable or failed GPU runs from being presented as measured zero-time or CPU-fallback results.

A truthful `failed` disposition does **not** make an otherwise complete source non-canonical. It is benchmark evidence about the maintained backend matrix and should be linked to a separately triaged production defect when appropriate. The initial P100 validation of PR #111 exposed exactly this situation for `LogisticRegressionCV` on Torch strict CUDA; the estimator defect is tracked in #112 and the benchmark row must remain failed until a later rerun demonstrates otherwise.

## Statistical alignment

A case records its dataset generator, exact scale, folds, split strategy, grid identity, candidate count, scoring direction, and normalization. External references are included only when objective scaling, regularization mapping, weighting, folds, and scoring are aligned.

For Ridge, statgpu uses an average-loss objective while scikit-learn uses an unnormalized residual sum of squares. Therefore an aligned unweighted reference applies `sklearn_alpha = n_fit * statgpu_alpha` separately for each training fold and again for the final full-data refit. The source retains the canonical statgpu `alpha` rather than replacing it with the mapped reference value.

The aligned regression score is mean squared error; the aligned binary-classification score is binary log-loss; Cox uses held-out partial log-likelihood with the declared ties semantics. Estimator-specific aliases such as R-squared, accuracy, or C-index must not be mixed into the same score field.

Cox cases additionally require subject-preserving folds and explicit survival scoring semantics. Unsupported delayed-entry, strata, weighting, or ties combinations must remain explicit dispositions rather than silently simplified cases.

## Provenance and raw repeats

The source records the statgpu commit, source/result date, generation time, host, CPU/GPU identity, Python/package versions, seeds, warmup count, repeats, synchronization policy, timing scope, and transfer policy. Successful rows retain raw per-seed timing samples in addition to aggregates.

The historical `lassocv_combined_20260409.json` file remains audit-only. It predates the canonical minimum date and does not provide the six-family timing decomposition required by this contract.

## Registration sequence

1. Run the benchmark on the declared CPU/GPU environment.
2. Validate the raw file with `python dev/benchmarks/cv_source.py <source.json>`.
3. Independently review objective and regularization alignment.
4. Copy the exact immutable raw file under `results/benchmark_frontend_sources/` without reconstructing or editing measured values.
5. Register its SHA256, parser version, comparison, environment, original path, and source date in `dev/benchmarks/frontend_sources.json`.
6. Update the source catalog and the six CV capability rows in `dev/benchmarks/benchmark_coverage_matrix.json`.
7. Generate the dashboard bundle and verify schema, semantic, staleness, TypeScript, and browser gates.
8. Add browser assertions against real parsed CV rows, including preservation of any explicit failed backend disposition.

No source is registered merely to enable the CV tab. Real measured evidence is a prerequisite.
