# PR #79 Full Repository Review

Date: 2026-07-11  
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
17. Ridge's exact CPU solver now uses the same un-normalized `alpha` convention
    as scikit-learn instead of multiplying the penalty by sample count or weight
    sum.
18. Cox inference now normalizes the legacy Breslow/Efron Hessian orientation at
    the observed-information boundary, preventing Efron standard errors from
    being clipped to zero while preserving coefficient estimates.

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
6. Focused regression suites cover backend validation, estimator parameters,
   RNG semantics, UMAP fuzzy union, NNDescent neighbor validity, CV validation,
   KMeans input contracts, small-sample spectral UMAP, Torch inference routing,
   Ridge/scikit-learn parity, and Cox/statsmodels parity.
7. CI now includes Python 3.9-3.12 regression gates, a complete Python 3.11 CPU
   test-tree job, package compilation, high-signal static checks, and complete
   pytest collection.

### Documentation

- README minimum Python version is aligned with `pyproject.toml` (`>=3.9`).
- Root, English, and Chinese changelogs document PR #79 and its validation
  boundary.
- This report records review scope, accepted fixes, deferred risks, and the
  validation boundary required by `dev/AGENTS.md`.

## Findings intentionally deferred

### Physical GPU validation

The GitHub-hosted jobs are CPU-only. CuPy/Torch routing, type preservation, and
error behavior are covered by isolated tests, but numerical parity, memory
usage, and performance have not been revalidated on physical CUDA hardware.
The review status is therefore `PARTIAL_REMOTE_PENDING`, not `COMPLETE`.

Required remote checks:

- run the affected UMAP/NNDescent, Cox, knockoff, inference, and ElasticNetCV
  suites on both CuPy CUDA and Torch CUDA;
- compare CPU/CuPy/Torch numerical outputs within documented tolerances;
- measure peak GPU memory and runtime before and after the changes;
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

GitHub Actions run **#199** passed all permanent gates:

- Python 3.9, 3.10, 3.11, and 3.12 selected regression matrices;
- the complete `dev/tests` CPU suite on Python 3.11;
- full package bytecode compilation;
- high-signal undefined-name/syntax Ruff checks on every modified production
  module;
- Cox review structure assertions;
- complete pytest collection without optional GPU import failures.

Final status: **PARTIAL_REMOTE_PENDING** until the physical GPU checks above are
completed.