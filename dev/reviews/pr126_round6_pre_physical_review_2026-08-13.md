# PR #126 — Round 6 pre-physical review checkpoint

Date: 2026-08-13

This checkpoint records the terminal **local/focused** state of the additional Round-6 `.claude/skills/code-review.md` review/fix loop requested after Round 5. It is deliberately a pre-hosted/pre-physical record: the commit that adds this file must itself pass the permanent hosted workflows before the PR body or a submitted exact-head review may claim hosted-green status.

## Reviewed technical candidate

Technical candidate before this checkpoint:

`e6dad766e3e329ccacab1e3fd76764f54992005f`

Temporary guarded review workflows were used only to apply exact, scope-checked fixes and run focused regressions. They are absent from the technical candidate tree.

## Round-6 findings closed

The additional independent review found and fixed the following material issues:

1. **CRITICAL — fixed-effect level prediction omitted the grand mean.**
   Known one-way/two-way fixed-effect predictions now restore the common level component exactly once and match the corresponding dummy/joint least-squares projection. Fully unknown fixed-effect labels retain the documented linear-only fallback.

2. **HIGH — formula-enabled fixed effects leaked across refits.**
   Effective entity/time effects are rebuilt from constructor configuration plus the current formula request on every fit. A formula FE fit no longer mutates the configuration used by a later array/no-FE refit.

3. **HIGH — formulas with more than two fixed-effect variables silently dropped extra effects.**
   High-dimensional FE formulas now fail closed. Effects-only FE formulas without a non-intercept regressor also have an explicit unsupported failure mode.

4. **HIGH — no-FE PanelOLS formula/intercept and fit-statistics semantics were inconsistent.**
   Patsy/R default intercepts are retained for no-FE level regression, `0 +`/`-1` remains no-intercept, explicit level constants are reflected in diagnostic degrees of freedom, centered TSS, adjusted R-squared, and classical model F semantics. Pinned statsmodels baselines cover both intercept and no-intercept level OLS behavior.

5. **HIGH — failed refits could expose stale fitted/inference state.**
   PanelOLS invalidates derived fit state at the start of every fit. Failed refits leave the estimator unfitted and clear coefficients, covariance/inference sidecars, formula metadata, backend/index provenance, and fixed-effect maps instead of mixing old parameters with new request metadata.

6. **CRITICAL — connected two-way prediction exposed unidentified one-sided fixed effects.**
   The normalization ambiguity of individual entity/time effects exists even on a connected incidence graph. Any two-way prediction that uses stored effects now requires both entity and time labels to be known and in the same incidence component. One-sided, known-plus-unknown, and cross-component combinations fail closed; both-unseen remains the explicit linear-only fallback.

7. **CRITICAL — no-intercept `rsquared_within` used centered TSS.**
   No-FE/no-intercept PanelOLS now uses standard uncentered total sum of squares, consistent with its overall R-squared and pinned statsmodels OLS behavior. FE fits and level-intercept fits retain centered/within semantics.

## Coverage added or strengthened

- focused NumPy regressions for one-way/two-way level prediction and grand-mean restoration;
- formula refit isolation, >2-FE fail-closed, effects-only failure, default-intercept and explicit no-intercept behavior;
- failed-refit transactional state including `_inference_result`, private inference arrays, covariance metadata, and public fitted outputs;
- pinned statsmodels estimator baselines for no-FE PanelOLS with and without an intercept;
- maintained Torch-CPU projection/prediction parity through the existing backend-neutral test path;
- physical correctness runner retains the 35-estimator + 12-public-covariance-primitive matrix and adds auditable prediction/level-regression contracts rather than inflating covariance case counts:
  - one-way fixed-effect prediction;
  - two-way fixed-effect prediction;
  - omitted explicit RandomEffects constant restoration;
  - disconnected two-way prediction guards;
  - connected two-way partial-label normalization guards;
  - no-FE level-constant inference/prediction parity against both same-backend PooledOLS and NumPy reference;
  - requested/executed fit and prediction backend provenance.

## Documentation and evidence boundary

Long-lived EN/CN panel model pages contain stable statistical/API semantics only. Detailed change history remains in changelog/review records.

The accepted Tesla P100 measurement `a99726e19c535dfcd0a94711bbc8be6aac437584` remains immutable historical evidence for the preceding numerical tree only. Round 6 changes production prediction/formula/inference behavior and the physical correctness runner, so current acceptance requires a fresh exact-clean P100 correctness + synchronized performance run on the eventual hosted-clean exact head.

The historical v4 parser/source identities remain frozen to `a99726e1...`; fresh evidence must use a new parser/source identity.

## Local/focused review verdict before checkpoint commit

- CRITICAL: 0 open
- HIGH: 0 open
- relevant MEDIUM: 0 open
- actionable inline review threads: 0
- temporary Round-6 workflows in technical candidate: 0

Provisional hard exit before hosted execution: `LOCAL_REVIEW_CLEAN / HOSTED_PENDING / PHYSICAL_REMOTE_PENDING`.

After this checkpoint commit, require the permanent hosted gate on the new exact head and then perform a fresh read-only `.claude/skills/code-review.md` review. Only if both are clean may the status advance to `HOSTED_GREEN / REVIEW_CLEAN / PARTIAL_REMOTE_PENDING`. PR #126 must remain Draft until fresh physical evidence is audited/promoted and the post-evidence review loop is clean.
