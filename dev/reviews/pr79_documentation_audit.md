# PR #79 Documentation Audit

Date: 2026-07-12  
Branch: `agent/code-review-fixes`  
Base: `master`

## Scope

This audit verifies that the public documentation matches the repository-wide,
Ridge, public-module, and native-backend changes in PR #79.

## Changelogs

The native three-backend follow-up is recorded separately from the earlier
statistical-contract fixes in all maintained changelogs:

- `CHANGELOG.md`
- `docs/en/changelog.md`
- `docs/cn/changelog.md`

Each entry distinguishes backend-native source changes and NumPy/Torch-CPU parity
from the still-pending physical CuPy/Torch CUDA validation.

## Public inventories and indexes

The following files now list the complete affected public API rather than the old
three-covariance/two-panel/one-ANOVA inventories:

- `README.md`
- `docs/en/README.md`
- `docs/cn/README.md`
- `docs/en/guides/implemented-methods.md`
- `docs/cn/guides/implemented-methods.md`

They include the seven covariance estimators, six panel estimators, expanded ANOVA
and post-hoc functions, kernel/nonparametric additions, `SplineTransformer`, and the
intentional metadata/scalar CPU boundaries.

## Model pages

The English and Chinese ANOVA, covariance, panel, and spline pages were synchronized:

- `docs/en/models/anova.md`
- `docs/cn/models/anova.md`
- `docs/en/models/covariance.md`
- `docs/cn/models/covariance.md`
- `docs/en/models/panel.md`
- `docs/cn/models/panel.md`
- `docs/en/models/splines.md`
- `docs/cn/models/splines.md`

The pages now document:

- balanced-only two-way ANOVA and additive-model residual semantics;
- fractional Welch denominator degrees of freedom and post-hoc result contracts;
- off-diagonal-only Graphical Lasso penalty, covariance-update convergence, and
  backend-native Graphical Lasso/CV and FAST-MCD execution;
- all six panel estimators, HAC/cluster metadata alignment, and backend-native
  Fama–MacBeth fitting/inference/prediction;
- backend-native Cox-de Boor evaluation and the `error`, `constant`, `linear`, and
  polynomial `continue` SplineTransformer extrapolation modes;
- NumPy/Torch-CPU parity versus the remaining physical CUDA validation boundary.

## Internal review records

- `dev/reviews/pr79_full_repository_review.md`
- `dev/reviews/pr79_native_backend_followup.md`
- this documentation audit

No temporary documentation synchronization workflow or patch script remains in the
pull request.

## Validation status

Documentation claims intentionally stop short of complete three-backend CUDA
validation. The status remains `PARTIAL_REMOTE_PENDING` until physical CuPy CUDA and
Torch CUDA numerical, device/type, transfer, peak-memory, runtime, convergence, and
repeated-fit checks pass.
