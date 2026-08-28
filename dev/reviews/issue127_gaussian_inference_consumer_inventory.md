# Issue #127 Gaussian inference consumer inventory

Baseline: `84f8bc7e17f66466b3a325cbb007b6cb41843821` (`v0.2.5`)  
Implementation branch: `agent/post-v0.2.5-next-phase-plan`

This inventory is the Phase-1 scope record required by
`dev/plans/gaussian_inference_backend_native_plan.md`. It extends the plan's
minimum capability table with consumers discovered while tracing the actual
post-fit data lifecycle.

## Public capability matrix

| Public capability | Fit backend | Inference contract | Formula | CV/final refit | #127 disposition |
| --- | --- | --- | --- | --- | --- |
| `LinearRegression` | NumPy/CuPy/Torch | Supported; single-response GPU inference already backend-native in v0.2.5 | Supported | Non-tunable | Preserve existing GPU path; shared CPU helper migrates to common backend-aware implementation |
| `PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l2")` | NumPy/CuPy/Torch | Supported when `compute_inference=True` | Supported | Direct estimator | **In scope**; public generic consumer discovered during implementation review |
| `PenalizedLinearRegression(..., penalty="l2")` | NumPy/CuPy/Torch | Supported when enabled | Supported | Penalized CV selection exists separately | Inherits the generic PGLM #127 router |
| `Ridge` | NumPy/CuPy/Torch | Supported by default | Supported | `RidgeCV` | Inherits generic router for non-precomputed paths; existing exact-GPU precomputed inference remains native |
| `RidgeCV` | NumPy/CuPy/Torch | Explicit `compute_inference` controls final `Ridge` refit | Not formula-facing | Supported | **In scope** final-refit regression; CV selection/grid semantics frozen |
| `PenalizedGLM_CV` | NumPy/CuPy/Torch CV | No public `compute_inference` option; optimized squared-error L2 refit is estimation-only | Not formula-facing | This is the CV estimator itself | Not promoted to a new inference capability by #127; selection/final coefficients remain regression surface |

## Data-lifecycle findings

### Shared Gaussian helper

The v0.2.5 `statgpu/linear_model/_gaussian_inference.py` path forced `X`, `y`,
parameters, residuals, covariance algebra, and reference-distribution inference
through NumPy/SciPy. The rewritten helper now:

1. resolves the requested/executed numerical backend;
2. keeps design, weighted state, bread/meat covariance algebra, statistics,
   p-values, critical values, and confidence intervals on NumPy/CuPy/Torch;
3. uses the maintained distribution backend for normal/Student-t evaluation;
4. takes one NumPy reporting snapshot only after numerical inference completes;
5. records numerical backend/device and reporting-boundary provenance.

NumPy numerical state intentionally remains float64 to preserve the historical
shared-helper precision contract. CuPy/Torch preserve the fitted floating dtype
and concrete device.

### `LinearRegression`

The audit found that the single-response CuPy and Torch fit paths already
compute their covariance and distribution inference natively before creating
reporting arrays. GPU multi-output inference already fails closed with a clear
`NotImplementedError`; #127 does not silently turn that unsupported surface
into a CPU fallback or advertise new support.

Consequently, #127 does not rewrite the large backend-specific
`LinearRegression` fitting implementation. Its CPU/shared-helper calls benefit
from the common migration while existing GPU numerical behavior remains frozen.

### Penalized Gaussian/L2

The shared penalized inference mixin contains several statistically distinct
branches. Only squared-error L2 is rerouted by #127. L1/ElasticNet debiased
inference and SCAD/MCP oracle/bootstrap behavior continue through the existing
mixin and receive regression guards.

The generic fit path can call post-fit inference with the original NumPy
`X/y` even after a CuPy/Torch fit. Therefore input container type is not valid
execution provenance. The public `PenalizedGeneralizedLinearModel` router uses
`_selected_backend_name` as authoritative, rebuilds numerical state on that
backend, runs inference, and only then creates the established NumPy reporting
state.

The exact-GPU L2 path already precomputed inference natively before #127. It is
left on that implementation and receives additive backend/reporting metadata;
no exact-solver statistical definition is replaced.

### Ridge and RidgeCV

Ridge keeps the average-loss penalty convention. With unnormalized normal
equations the inference penalty contribution is `n_eff * alpha`, where
`n_eff=n` for unweighted fits and `n_eff=sum(sample_weight)` for analytic-weight
fits. Intercept penalty treatment remains unchanged.

`RidgeCV` explicitly exposes `compute_inference` and constructs a final `Ridge`
refit. #127 therefore treats it as a required final-refit inference regression
surface. Fold generation, candidate grid, scoring, selected alpha, tie behavior,
and final coefficients are not implementation scope.

`PenalizedGLM_CV` is different: its public constructor has no inference toggle,
and its optimized squared-error L2 refit intentionally constructs an
estimation-only PGLM. #127 does not add a new CV inference API to that class.

## Validation matrix activated by the inventory

Hosted/locally runnable checks must cover:

- ordinary NumPy closed-form/SciPy parity;
- nonrobust, HC0/HC1/HC2/HC3, and HAC covariance;
- weighted and unweighted state;
- Ridge/L2 `n_eff * alpha` mapping and unpenalized intercept;
- rank-deficient pseudoinverse behavior;
- scalar and multi-target shared-helper behavior;
- small-df/extreme-tail distribution precision;
- Torch CPU backend-native inference;
- public generic PGLM L2 routing from NumPy inputs to the selected Torch backend;
- non-L2/non-Gaussian delegation to the shared penalized inference mixin;
- RidgeCV final-refit inference without changing alpha selection or coefficients;
- physical CuPy and Torch CUDA execution with exact source/device provenance.

Canonical physical evidence must use
`dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py` after that
validator contract is review-clean. Changing its case matrix, acceptance
thresholds, backend/device checks, or pass/fail logic after a physical run
invalidates affected canonical evidence and requires a rerun.
