# PR #126 — Round 7 pre-physical review checkpoint

Date: 2026-08-13

This checkpoint records the additional exact-head read-only review and fix loop performed after the Round-6 hosted-clean checkpoint. It is deliberately a pre-physical acceptance record: the commit adding this file must itself pass the permanent hosted workflows before the PR may claim exact-head hosted-green status.

## Reviewed technical candidate

Round-7 technical candidate:

`ad42ed423bb9d62d63c04c3a4b8a00eb3cd9f289`

The temporary guarded workflow used to apply and verify the fix is absent from the technical candidate tree. Relative to the Round-6 exact head `e9a8d315ce9cf07cdd366763c635ecb5991489c5`, the final Round-7 tree changes only:

- `statgpu/panel/_fixed_effects.py`;
- `dev/tests/test_panel_stage_c_final_review_fixes.py`.

## Round-7 finding closed

### HIGH — failed refits were not fully transactional after late failures

Round 6 correctly invalidated stale fitted/inference state at fit entry, but a later failure inside the new fit could still leave metadata written by the failed request itself. In particular, `_panel_prepare_numeric()` persists executed backend provenance before later validations, while subsequent fit stages can write `nobs`, panel index metadata, effective formula fixed-effect flags, effect maps, and covariance/inference sidecars. Because the old implementation reset only at fit entry, a late exception could leave an unfitted estimator containing a mixture of constructor state and partially written failed-attempt state.

The public `PanelOLS.fit()` boundary is now transactional:

1. reset derived fit state before the attempt;
2. delegate the numerical/statistical implementation to `_fit_impl()`;
3. on any exception, reset derived fit state again and re-raise unchanged.

This restores constructor-configured entity/time effect flags and clears fitted coefficients, inference/covariance sidecars, formula metadata, backend/prediction provenance, observation/index metadata, and fitted effect maps after every failed refit. `PanelOLS.summary()` also checks fitted state before dereferencing fitted parameters, so a failed refit reports the standard not-fitted error instead of a secondary `None`-state exception.

## Focused evidence

Guarded Round-7 verification succeeded on GitHub Actions run `31677868852`:

- exact checkpoint-lineage guard: PASS;
- exact patch application: PASS;
- `dev/tests/test_panel_stage_c_final_review_fixes.py`: PASS;
- `dev/tests/test_panel_stage_a_golden.py`: PASS;
- panel package compile check: PASS;
- diff/scope guard: PASS;
- temporary workflow self-deletion and final technical commit: PASS.

The strengthened regression now forces a late clustered-covariance refit failure after backend/index state has been written and verifies that the estimator is completely reset. The effects-only formula failure regression also verifies restoration of constructor fixed-effect flags and formula/backend state. Prediction and summary both fail through the standard fitted-state guard afterward.

## Independent re-review

A fresh read-only review of the resulting two-file technical diff found no additional open CRITICAL, HIGH, or relevant MEDIUM issue.

The historical `PanelOLS.df_resid` versus standard diagnostic-df split was explicitly rechecked against `dev/plans/panel_p1_stage_b_diagnostics_df_addendum.md`. That split is normative compatibility behavior: legacy public covariance/coefficient inference keeps the Stage-A residual-df convention, while Stage-B standardized diagnostics use nuisance-rank/component-aware df. It is therefore not changed by Round 7.

## Hosted execution note

The seven permanent pull-request workflows created directly for the bot-authored technical commit `ad42ed423...` all ended as `action_required` with no jobs because the triggering actor was `github-actions[bot]`; this is a GitHub Actions authorization state, not a test failure. This user-authored checkpoint commit is intended to provide a normal exact-head pull-request trigger. Hosted-green status must be based only on the permanent workflows for the checkpoint exact head, not on the bot-triggered `action_required` runs.

## Physical-evidence boundary

The last accepted Tesla P100 measurement `a99726e19c535dfcd0a94711bbc8be6aac437584` remains immutable historical evidence for the preceding numerical tree only. Round 6 changed production numerical/prediction behavior and the physical correctness runner; Round 7 changes refit failure-state/API behavior. Current acceptance still requires a fresh exact-clean P100 correctness + synchronized performance run on the eventual hosted-clean exact head.

The historical v4 parser/source identities remain frozen to `a99726e1...`; fresh physical evidence must use a new parser/source identity and immutable promotion.

## Pre-hosted verdict

- CRITICAL: 0 open
- HIGH: 0 open
- relevant MEDIUM: 0 open
- temporary Round-7 workflows in technical candidate: 0
- physical GPU evidence: pending fresh Tesla P100 rerun

Provisional hard exit before this checkpoint's permanent workflows complete:

`LOCAL_REVIEW_CLEAN / HOSTED_PENDING / PHYSICAL_REMOTE_PENDING`

PR #126 must remain Draft until fresh physical evidence is audited/promoted and the post-evidence review loop is clean. No Ready-for-review transition or merge is authorized by this checkpoint.
