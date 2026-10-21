# IG / negative-binomial robust inference bread repair

Status: design reviewed; implementation in progress; existing-capability reconciliation, not a release.
Base: efbfb80570eb1632e0605d68d410715dac4d9be8.

## Contract

The existing M-estimation engine labels HC0/HC1 bread observed, but two losses
return expected Fisher curvature from their solver-facing `hessian` method.
The existing active-set CPU oracle already corrects this through an adapter.
Separate inference observed curvature from optimizer curvature, preserve the
objective and fitted coefficients, and unify the inference implementations.
The average-loss convention is H = sum(w_i H_i) / sum(w_i); robust meat uses
squared analytic weights. Penalty curvature is added to the average Hessian,
with intercept unpenalized. HC1 uses observation-count correction, not weight
sum. Model-based inference continues to use Fisher information and dispersion.

For log-link inverse Gaussian, observed eta curvature is (2 y - mu) / mu^2.
For NB2 with fixed alpha, it is mu (1 + alpha y) / (1 + alpha mu)^2.
These are derivatives of the estimating score in the unclipped domain.

## Consumers and gates

- Unpenalized InverseGaussianRegression / NegativeBinomialRegression via
  _GLMBase: HC0/HC1 corrected; nonrobust preserved.
- Generic and typed penalized GLM L2 / supported elastic-net sandwich:
  same corrected observed bread plus unchanged penalty curvature; inference
  targets the penalized estimating equation conditional on fixed tuning.
- Penalized oracle active-set refit: preserve already-correct observed bread,
  selection conditioning, clipping guards, and weight range safeguards.
- PenalizedGLM_CV: candidates unchanged, selected final refit receives fix;
  CV-selected penalty uncertainty remains unadjusted.
- Formula fits: design/intercept/column alignment preserved; test equivalence.
- NumPy / CuPy / Torch: native math, no new host transfers or fallback.
  Test CPU now and conditional physical-device tests; skips are not validation.
- Other losses inherit existing Hessian as observed curvature; no numerical
  changes. HC2/HC3/HAC remain unsupported for non-Gaussian public inference.
- Solver Newton / FISTA / LBFGS, Fisher Hessians, Lipschitz estimates, loss
  normalization and family mean clipping remain unchanged.

## Validation

Independent analytic weights and central finite differences, weighted and
unweighted, prove observed curvature. Independent sandwich algebra proves
HC0/HC1 and penalty scaling. Existing strict statsmodels benchmark compares
all 74 model cases; do not loosen tolerances. Existing penalized inference and
oracle tests guard consumers. Check CV final-refit conditioning and formula
alignment. Record exact source hashes and honest unavailable-device evidence.
Independent design review precedes production edits; final review follows tests.

Design review approved preserving analytic-at-clipped-mean convention; derivative
proof is interior-only. Oracle retains its stronger unclipped-domain rejection.

## Local results (2026-10-20)

- Frozen consumer suite: 135 passed, 8 skipped. Includes the model benchmark
  harness, new observed-bread tests, penalized inference contracts and prior
  review-closure tests. Four new Torch CPU cases actually executed; eight
  physical CuPy/Torch CUDA cases were skipped because unavailable.
- New targeted tests separately: 17 passed, 8 skipped. Six public direct/CV
  cases cover both families and HC0/HC1/nonrobust; two formula cases preserve
  coefficient and covariance alignment. Nonrobust reference respects public
  analytic-weight normalization to sum n and Pearson dispersion.
- Existing oracle observed-bread / clipping characterization: 12 passed.
- Full model benchmark smoke: 74 passed; no source changes during run.
  IG HC0 SE max absolute error 5.4094e-13; NB HC0 8.0918e-9. Existing fixed
  thresholds remain unchanged. No physical CUDA or performance claim added.
- Python 3.9 AST parsing passed for all changed numerical source and new tests.
- Independent design and implementation review found one generic-helper
  fallback issue: corrected so missing Fisher uses solver Hessian for
  nonrobust, not observed curvature. Sentinel regression covers this.
- Full 2,836-case historical oracle suite is additionally running; its final
  outcome must be reported separately from the targeted results above.
