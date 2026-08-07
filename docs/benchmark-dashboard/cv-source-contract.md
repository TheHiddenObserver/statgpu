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

The runner profiles these regions inside the same public `fit` call. It does not derive component timings by subtracting rounded or separately executed measurements. GPU measurements synchronize the resolved backend immediately before and after each observed region. Input conversion occurs before the measured public fit, so the initial source uses a device-resident timing scope.

Warmup runs are executed but excluded from aggregates. Global CV caches are cleared before every warmup and measured repeat so repeated timings cannot silently become cache-hit timings.

## Backend disposition

Each representative case explicitly records a statgpu disposition for `numpy`, `cupy`, and `torch`:

- `success` only when the run was executed and measured;
- `unavailable` when the required runtime or physical device is absent;
- `unsupported` when the maintained API does not implement that combination;
- `failed` when an attempted maintained combination raises or produces invalid evidence.

A non-success row contains a reason and no timing, score, selection, or convergence measurements. This prevents unavailable GPU runs from being presented as measured zero-time or CPU-fallback results.

## Statistical alignment

A case records its dataset generator, exact scale, folds, split strategy, grid identity, candidate count, scoring direction, and normalization. External references are included only when objective scaling, regularization mapping, weighting, folds, and scoring are aligned.

The runner provides aligned manual sklearn references for `RidgeCV`, `LassoCV`, `ElasticNetCV`, and `LogisticRegressionCV`. These references use the declared deterministic folds, candidate grid, objective, and final full-data refit rather than relying on a library default CV configuration. No sklearn substitute is asserted for `PenalizedGLM_CV` or `CoxPHCV`.

Cox cases additionally require subject-preserving folds and explicit survival scoring semantics. Unsupported delayed-entry, strata, weighting, or ties combinations must remain explicit dispositions rather than silently simplified cases.

## Provenance and raw repeats

The source records the statgpu commit, source/result date, generation time, host, CPU/GPU identity, Python/package versions, seed, warmup count, repeat count, synchronization policy, timing scope, and transfer policy. Successful rows retain every measured repeat in addition to aggregate values.

All source numbers must be finite. The writer uses strict JSON serialization and the semantic validator recursively rejects `NaN` and infinities.

The historical `lassocv_combined_20260409.json` file remains audit-only. It predates the canonical minimum date and does not provide the six-family timing decomposition required by this contract.

## Execution

Install the repository with the validation and appropriate GPU extras. On a CUDA 12 environment with Torch available, for example:

```bash
python -m pip install -e '.[validation,gpu12,torch]'

python dev/benchmarks/benchmark_cv_models.py \
  --out results/cv_benchmark_candidate.json \
  --env-id remote-p100 \
  --backends numpy,cupy,torch \
  --n-samples 240 \
  --n-features 16 \
  --seed 20260807 \
  --repeats 3 \
  --warmup 1 \
  --include-sklearn

python dev/benchmarks/cv_source.py results/cv_benchmark_candidate.json
```

The environment identifier and package extra must match the actual machine. Do not use a CPU-only result to claim CuPy or Torch coverage. A candidate with unavailable GPU dispositions is useful for runner verification but is not sufficient for the initial canonical package when those backends are maintained.

## Dashboard integration

The canonical parser emits a normalized `metrics.cross_validation` object containing:

- CV evaluation, final refit, and total fit timing;
- selected parameters;
- validation and final scores with scoring identity/direction;
- candidate and fold counts;
- failed candidate/fold counts;
- final-refit convergence.

The frontend renders these fields in a dedicated Cross-validation Metrics panel. The panel remains absent while the canonical bundle contains no real CV rows.

## Registration sequence

1. Run the benchmark on the declared CPU/GPU environment.
2. Validate the raw file with `python dev/benchmarks/cv_source.py <source.json>`.
3. Independently review objective and regularization alignment.
4. Copy the immutable source under `results/benchmark_frontend_sources/`.
5. Register its SHA256, parser version, comparison, environment, and source date in `frontend_sources.json`.
6. Update the coverage matrix from gap to canonical or partial canonical coverage.
7. Generate the dashboard bundle and verify schema, semantic, staleness, TypeScript, build, and browser gates.

No source is registered merely to enable the CV tab. Real measured evidence is a prerequisite.
