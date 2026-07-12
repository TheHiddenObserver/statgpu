# PR #79 Full Repository Review

Date: 2026-07-12  
Branch: `agent/code-review-fixes`  
Base: `master`

## Scope

This review followed `dev/AGENTS.md`, `.claude/workflows/new-module-dev.md`, and
`.claude/skills/code-review.md`. The review covered:

- correctness and statistical/API contracts;
- NumPy, CuPy, and Torch backend routing;
- readability, maintainability, and extensibility;
- input validation and sklearn-style estimator behavior;
- performance and memory-risk hot paths;
- test quality, test discovery, documentation, and CI gates.

The package inventory at the start of the repository-wide pass contained 201
Python source files and approximately 77,900 source lines. The audit combined
manual review, package compilation, high-signal Ruff rules, dead-code scanning,
full pytest collection, selected multi-version regression tests, and targeted
regression tests for every accepted fix.

A subsequent focused pass traced Ridge through the optimized wrapper, generic
penalized estimator, exact and iterative solvers, formula handling, weighted
inference, `RidgeCV`, `PenalizedGLM_CV`, documentation, validation scripts, and
benchmark scripts.

## Fixed findings

### Correctness and API contracts

1. Backend factory arguments are validated instead of silently treating typos as
   automatic selection.
2. Explicit `device="cuda"` and `device="torch"` warnings check the requested
   backend, rather than merely checking whether any CUDA backend exists.
3. `BaseEstimator.set_params()` rejects unknown parameters and supports nested
   `name__parameter` updates.
4. Model-context inference now resolves explicit Torch device selection to the
   Torch backend and casts all p-value/resampling inputs consistently.
5. Adaptive L1 gradient evaluation no longer raises `NameError` because the
   backend array resolver is now imported.
6. The Torch Newton fallback no longer references an unbound `torch` name.
7. The CuPy knockoff path no longer calls an undefined array-type helper.
8. The Cox Torch Hessian path uses `n_samples` instead of an undefined `n`.
9. UMAP constructs the actual fuzzy union `W + W.T - W * W.T`.
10. UMAP and resampling treat `random_state=None` as fresh entropy while fixed
    seeds remain reproducible.
11. Shared random seeds are normalized to the unsigned 32-bit range accepted by
    NumPy/CuPy `RandomState` and Torch generators.
12. NNDescent initializes its convergence counter, validates inputs, fixes
    `argpartition` kth semantics, excludes duplicate/self neighbors, and returns
    consistent float64 squared distances.
13. Small-sample spectral UMAP initialization returns exactly the requested
    number of components.
14. `KMeans.score()` now applies the same sparse, dimensionality, and feature
    count checks as `predict()` and `transform()`.
15. Cross-validation cache size, sample weights, chunk size, intercept vectors,
    and weighted MSE inputs now have explicit contracts.
16. Mixed NumPy/GPU CV inputs are converted together instead of returning a
    NumPy backend label with an unconverted GPU object.
17. Cox inference normalizes the legacy Breslow/Efron Hessian orientation at the
    observed-information boundary, preventing Efron standard errors from being
    clipped to zero while preserving coefficient estimates.

### Ridge objective and weighted-path consistency

1. Ridge preserves the package-wide objective `average data loss + penalty`.
   For L2 regression this yields:

   - unweighted: `(Xc'Xc + n*alpha*I) beta = Xc'yc`;
   - weighted: `(Xc'W Xc + sum(w)*alpha*I) beta = Xc'W yc`.

   scikit-learn comparisons use `alpha_sklearn = n*alpha_statgpu`, or
   `sum(w)*alpha_statgpu` for weighted fits, rather than changing statgpu's
   internal objective.
2. Weighted centering is performed before multiplying by `sqrt(sample_weight)`.
   The optimized Ridge wrapper, generic exact solver, and CPU FISTA path now
   solve the same weighted objective.
3. Explicit CuPy/Torch exact paths use weighted means and `sum(w)` normalization,
   matching the CPU objective instead of centering already weighted arrays with
   an ordinary mean.
4. IRLS receives the same `sum(w)*alpha` ridge curvature when sample weights are
   present.
5. Weighted Gaussian inference uses design `[sqrt(w), sqrt(w)*X]`, response
   `sqrt(w)*y`, and residual `sqrt(w)*(y - intercept - X beta)`. The intercept
   column, bread, meat, scale, and ridge curvature therefore follow one weighting
   convention.
6. `RidgeCV` default alpha grids use weighted-centered cross-products divided by
   total weight. Both the alpha grid and the full CV fit are invariant to global
   positive rescaling of sample weights.
7. `PenalizedGLM_CV(loss="squared_error", penalty="l2")` now generates its
   default alpha grid from the normalized weighted null gradient. Its alpha grid,
   selected alpha, and final fit are invariant to global positive rescaling of
   sample weights.
8. The default explicit-GPU Newton Ridge CV fallback no longer constructs a
   host-side Gram cache that the Newton solver does not consume. Explicit exact
   Ridge and sparse squared-error paths retain the cache where it is used.
9. Formula parsing records the retained row positions after Patsy missing-value
   filtering, allowing full-length side arrays such as sample weights to be
   aligned with the fitted rows.
10. Sample-weight validation rejects wrong length, non-finite values, negative
    values, and non-positive totals. CuPy/Torch validation performs reductions on
    the selected device and synchronizes only scalar results, avoiding a full
    weight-vector transfer to CPU.
11. English and Chinese Ridge documentation now states the actual average-loss
    and weighted objectives, estimating equations, alpha mappings, and inference
    convention. Maintained validation and benchmark scripts use the same mapping.

### Post-Ridge public module audit

1. **ANOVA and post-hoc inference**
   - Additive two-way ANOVA now absorbs omitted interaction variation into the
     residual instead of inflating main-effect F statistics.
   - The balanced-design sums-of-squares implementation rejects unbalanced cells
     until the API exposes an explicit Type I/II/III convention.
   - Welch ANOVA preserves fractional denominator degrees of freedom and rejects
     mixed zero-variance groups rather than silently changing the null hypothesis.
   - Tukey and Bonferroni comparisons handle identical constant groups without
     NaN/Inf artifacts and accept optional GPU inputs through an explicit boundary.
2. **Kernel methods**
   - The chi-square kernel rejects negative features and its chunked NumPy fallback
     matches the reference definition.
   - KernelRidge validates inputs, uses stable solve fallback, and implements
     force-finite uniform-average multi-output R-squared.
   - KernelRidgeCV validates folds/grids, avoids unused eigenvectors, and reports
     actual mean fold R-squared.
   - KernelPCA uses the unregularized centered-kernel eigenvalues for embeddings so
     training `fit_transform` and `transform` agree.
   - Nystroem uses SVD normalization for indefinite kernels instead of converting
     negative eigenvalues into enormous artificial features.
3. **Covariance estimators**
   - EmpiricalCovariance computes the exact precision when possible and adds jitter
     only as a singular fallback.
   - GraphicalLasso uses covariance block-coordinate descent, leaves the precision
     diagonal unpenalized, preserves the empirical covariance diagonal, and returns
     mutually consistent covariance/precision matrices.
   - GraphicalLassoCV validates folds and alpha grids; MinCovDet validates support
     fractions and honors `assume_centered` throughout its C-steps.
4. **Panel estimators**
   - Cluster labels are factorized before GPU conversion; clustered/HAC covariance
     validates labels, lengths, kernels, and bandwidths.
   - Formula-side entity/time/cluster arrays follow Patsy's retained rows after
     missing-value filtering.
   - Pooled, between, first-difference, fixed-effects, and Fama-MacBeth paths use
     stable pseudoinverse fallbacks and explicit residual-degree/period checks.
5. **Smoothing, splines, GAM, and metrics**
   - SplineTransformer now implements real `error`, `constant`, `linear`, and
     `continue` extrapolation rather than silently returning zero/ignoring modes.
   - B-spline, KDE, and kernel-regression shared utilities reject non-finite inputs,
     invalid knots, and non-finite weights.
   - GAM validates smoothing parameters, shapes, finite data, and constant features,
     and accepts one-dimensional prediction points for one-feature fits.
   - Binary evaluation rejects non-finite decision thresholds.

### Test and CI quality

1. A remote GPU runner was moved out of `dev/tests`, so CPU-only pytest
   collection no longer imports CUDA-only dependencies.
2. ElasticNetCV tests no longer import Torch unconditionally.
3. GPU and optional-Torch tests now skip only when their backend dependency is
   unavailable; unexpected backend failures are no longer swallowed as passing
   tests.
4. Stale tests were aligned with the public `statgpu.losses` namespace and the
   benchmark-backed auto-solver dispatch table.
5. RidgeCV helper-style tests now contain explicit assertions and backend skips.
6. Focused repository-review suites cover backend validation, estimator
   parameters, RNG semantics, UMAP fuzzy union, NNDescent neighbor validity, CV
   validation, KMeans input contracts, small-sample spectral UMAP, Torch
   inference routing, and Cox/statsmodels parity.
7. Ridge-specific tests cover:
   - unweighted and weighted average-loss closed forms;
   - invariance to global sample-weight rescaling;
   - optimized wrapper versus generic exact estimator;
   - exact versus FISTA equality;
   - formula fits, including rows removed because of missing values;
   - manual weighted inference covariance and weighted design state;
   - weighted default alpha-grid and full `RidgeCV` invariance;
   - weighted default alpha-grid and fit invariance for `PenalizedGLM_CV`;
   - explicit unweighted and weighted scikit-learn alpha mappings;
   - scalar-only NumPy/Torch weight validation;
   - cache-consumer routing for GPU exact versus Newton Ridge CV.
8. Three post-Ridge public-module suites cover:
   - nested additive/two-way ANOVA identities and degenerate post-hoc cases;
   - sklearn/reference parity and invariants for kernel/covariance estimators;
   - formula missing-row alignment and panel covariance/rank-deficiency contracts;
   - spline extrapolation, smoothing/GAM finite-value contracts, and metric edges.
9. CI includes Python 3.9-3.12 regression gates, a complete Python 3.11 CPU
   test-tree job, package and maintained-dev-script compilation, high-signal
   static checks, and complete pytest collection.

### Documentation

- README minimum Python version is aligned with `pyproject.toml` (`>=3.9`).
- Root, English, and Chinese changelogs document PR #79 and its validation
  boundary.
- English and Chinese Ridge model pages document the internal objective rather
  than presenting the unnormalized scikit-learn equation as the statgpu API.
- This report records review scope, accepted fixes, deferred risks, and the
  validation boundary required by `dev/AGENTS.md`.

## Findings intentionally deferred

### Physical GPU validation

The GitHub-hosted jobs are CPU-only. CuPy/Torch routing, type preservation,
scalar-reduction behavior, and error contracts are covered by isolated tests,
but numerical parity, memory usage, and performance have not been revalidated on
physical CUDA hardware. The review status is therefore
`PARTIAL_REMOTE_PENDING`, not `COMPLETE`.

Required remote checks now include:

- run weighted and unweighted Ridge exact fits on both CuPy CUDA and Torch CUDA;
- compare CPU/CuPy/Torch coefficients, intercepts, predictions, and weighted
  inference outputs within documented tolerances;
- verify global sample-weight rescaling invariance for `Ridge`, `RidgeCV`, and
  `PenalizedGLM_CV` on both GPU backends;
- confirm that weight validation transfers only scalar reductions and measure
  peak memory/runtime for large weight vectors;
- confirm that GPU Newton Ridge CV does not construct the unused host Gram cache;
- run the affected UMAP/NNDescent, Cox, knockoff, inference, and ElasticNetCV
  suites on both CuPy CUDA and Torch CUDA;
- validate kernel, covariance, panel, KDE/kernel-regression, spline, GAM, and
  post-hoc paths for numerical parity, output type/device, memory, and runtime;
- verify cleanup hooks and repeated-fit memory behavior.

### Cox Hessian memory optimization

The original Torch Hessian implementation still materializes an
`O(n * p * p)` intermediate. During this review, an attempted broad replacement
was detected by regression tests and fully reverted. Only the definite
undefined-dimension bug was retained. A memory-bounded Hessian rewrite should be
implemented as a separate PR with direct numerical equivalence tests and GPU
peak-memory benchmarks.

### Broader architectural debt

Static scanning identified many direct CuPy imports, explicit host transfers,
and broad exception handlers. These are not automatically bugs: several are
intentional optional-dependency guards or documented CPU fallback boundaries.
Blanket replacement would be higher risk than the current code. They should be
addressed module-by-module with backend parity and performance evidence.

The `statgpu/linear_model/legacy` tree also contains dead and statically invalid
reference code. It is explicitly outside the public API. A separate cleanup PR
should either remove/archive it or make its non-importability mechanically
explicit.

## Validation status

GitHub Actions run **#268** passed all permanent gates on the final branch state:

- Python 3.9, 3.10, 3.11, and 3.12 selected regression matrices;
- the complete `dev/tests` CPU suite on Python 3.11;
- package and maintained validation/benchmark script bytecode compilation;
- high-signal undefined-name/syntax Ruff checks on modified production modules,
  including the Ridge, ANOVA, kernel, covariance, panel, smoothing, spline, GAM, and metrics paths;
- Cox review structure assertions;
- complete pytest collection without optional GPU import failures.

Final status: **PARTIAL_REMOTE_PENDING** until the physical GPU checks above are
completed.
