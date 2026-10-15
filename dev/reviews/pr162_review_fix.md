# PR #162 review/fix closure — Issue #161

## Target

- target_kind: PR
- PR: #162
- base: `master`
- base_sha at first review: `38c64bd4f8195358deb12d6e0f40177b89ac3c80`
- reviewed implementation head before this evidence-only note: `b19c0ba629ab0e5bdce48ecca21b1427aba3cf53`

## Change classification

Existing-capability contract repair at the low-level Quantile IRLS boundary. The supported L2/no-penalty algorithm, backend routing, sample-weight normalization, and estimator-level solver validation are preserved. The repair removes an unsupported direct ElasticNet/non-smooth route that previously optimized only part of the declared penalty.

## Findings and disposition

### Closed — low-level ElasticNet partial-objective bug

`QuantileLoss.irls()` previously accepted `ElasticNetPenalty` and added only its L2 diagonal term. It now validates the penalty before initialization/linear algebra and accepts only no penalty or L2. ElasticNet, L1, SCAD, and unknown penalty objects fail closed with a precise error.

### Adjacent finding split to Issue #163

The review also found a pre-existing solver-provenance mismatch for smooth penalized Quantile fitting: the public/auto route can report FISTA while `_fit_loss_backend()` internally invokes IRLS. This is not caused by the #161 patch and the #161 repair does not change that maintained L2/no-penalty path. It is tracked separately as Issue #163 so this PR does not silently expand into a Quantile-wide routing migration.

## Consumer closure

- Direct `QuantileLoss.irls()` None/L2 routes remain executable.
- Direct non-smooth penalties reject before `lstsq`/`solve`.
- Weighted L2 IRLS remains invariant to positive global analytic-weight rescaling.
- Public `PenalizedQuantileRegression(..., solver="irls", penalty="elasticnet")` keeps its existing fail-closed estimator contract.
- Torch CPU direct L2 IRLS remains backend-native.
- Existing learner/reference documentation already states that ordinary Quantile IRLS is L2/no-penalty only and that ElasticNet uses a FISTA-family route; no learner-facing project-tracking text is added.

## Validation

Hosted validation on `b19c0ba629ab0e5bdce48ecca21b1427aba3cf53`:

- Tests workflow: success, including full CPU suite, static contracts, documentation contracts, Python 3.9–3.12 regression matrix, and Torch 2.0 CPU regression environment.
- Maintenance compatibility: success.

No physical CUDA rerun is required for #161 because the change removes an unsupported low-level route and does not alter any maintained L2/no-penalty NumPy/CuPy/Torch numerical path. Any future solver-routing change discovered here belongs to #163 and will receive its own evidence decision.

## Review verdict

For the Issue #161 scope, no actionable finding remains on the reviewed implementation state. This note itself is review/evidence-only and does not alter production behavior.
