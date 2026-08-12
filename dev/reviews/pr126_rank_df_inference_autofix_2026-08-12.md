# PR #126 rank-deficient df / inference auto-fix review — 2026-08-12

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Current verdict

**PARTIAL_REMOTE_PENDING / LOCAL REVIEW CLEAN / HOSTED CHECKPOINT PENDING / NOT MERGE-READY**

Active axes: public API/presentation, inference, NumPy/CuPy/Torch backend behavior, formula/panel metadata, benchmark/performance, and docs/artifacts. Loss, penalty, generic solver framework, and CV remain inactive.

The current local source/docs/runner candidate before this checkpoint is `1a6d75e0140bc26d17a0425c4226d4a81a9c102c`. The prior hosted checkpoint exposed two maintained-contract gaps; both are fixed there without reverting the rank-aware production behavior. This connector-authored review-record-only commit is the new exact hosted-checkpoint head; it does not modify production source, tests, physical runners, benchmark parsers/manifests, or generated benchmark assets.

## Findings reopened by independent review

### [CRITICAL][BUG][fixed] Rank-deficient residual and Swamy-Arora df used raw column count

The supported minimum-norm rank-deficient paths already used explicit numerical rank for coefficient solves, but RandomEffects auxiliary/final df and adjacent Between/FirstDifference/Panel legacy-df paths still counted redundant columns. Equivalent identified column spaces could therefore produce different variance components, theta, scales, HC1 corrections, or inference.

Fix:

- historical full-rank formulas are preserved exactly;
- rank-deficient PanelOLS, BetweenOLS, FirstDifferenceOLS, and RandomEffects residual df use identified fit-space rank;
- RandomEffects within/between auxiliary df use `rank_within`/`rank_between` with the original nuisance-effect parameterization;
- column-space invariance tests compare `[x]` with `[x,2x]` under nonrobust and HC1/robust, including unbalanced RE and entity-effect PanelOLS.

### [HIGH][INFER][fixed] Negative-variance validity depended on measurement units

The prior `4096*eps*max(1, |V_jj|)` guard contained an absolute unit-dependent floor. A row/column covariance scale was considered during the first fix but rejected on re-review because off-diagonal covariance entries transform differently under regressor reparameterization.

Final fix: strict inference rejects every strictly negative final covariance diagonal. IEEE `-0.0` compares equal to zero and may be normalized to zero. Tests cover extreme outcome scaling, large off-diagonal entries, and signed zero.

### [HIGH][INFER][fixed] Non-finite covariance could bypass the strict diagonal guard

`NaN` and positive infinity are not `< 0`, so a diagonal-only sign check could still publish non-finite BSE/CI if finite inputs overflowed during covariance algebra. Strict inference now fails closed unless the entire final covariance matrix is finite before the negative-diagonal check. Maintained mock-covariance regressions cover `NaN`, `+Inf`, and `-Inf`.

### [HIGH][MATRIX][fixed locally / needs remote GPU] Physical matrix did not exercise the repaired df branches

The physical runner now adds eight exact-collinearity estimator integrations: PanelOLS(entity FE), BetweenOLS, FirstDifferenceOLS, and RandomEffects under both nonrobust and historical HC1/robust. Each case persists `fit_rank` and `parameter_count`; fresh GPU audit must verify `fit_rank < parameter_count` and requested/executed backend identity.

Fresh target: **47/47 per backend = 35 estimator integrations + 12 public primitives**. Performance remains 58 rows. The timing runner now discovers CUDA-suffixed CuPy distribution versions.

### [HIGH][TEST][fixed] Maintained BetweenOLS zero-df test encoded the obsolete raw-column-count rule

The permanent regression matrix still required `BetweenOLS` to raise whenever the number of entity means did not exceed the raw parameter count. That test contradicted the newly supported identified-rank extension. It is now split into two executable contracts: an exact-collinearity case with `groups <= raw k` but positive `groups-rank` must fit with `df_resid=1`, while a full-rank three-group design with `rank == groups` must fail for genuinely zero identified residual degrees of freedom.

### [MEDIUM][DOC][fixed] Long-lived model pages used a PR lifecycle token

The documentation contract correctly rejected `PARTIAL_REMOTE_PENDING` in EN/CN model pages as a stale global release-status marker. The model pages now state only durable evidence facts: the old P100 source is historical, the post-fix implementation has not yet been promoted from fresh physical evidence, and the maintained matrix is 47/47 per backend plus 58 performance rows. Exact PR lifecycle status remains in `dev/reviews/` and PR metadata.

## Validation completed before this checkpoint

- first focused source/external matrix: **86 passed**;
- strict variance/entity-FE re-review matrix: success;
- expanded runner + maintained Torch contract matrix: **45 passed**;
- non-finite covariance focused inference/covariance matrix: success;
- docs/plan/current-acceptance consistency validation: success;
- hosted-failure follow-up regression matrix: success;
- maintained documentation contracts and bilingual link check after follow-up: success;
- syntax / compile checks: success;
- `git diff --check`: success.

## Independent re-review result

The post-fix source review rechecked rank/df formulas, full-rank backward compatibility, backend-native fit/covariance paths, formula chronology/alignment, physical runner provenance, performance runner provenance, strict finite/negative covariance failure semantics, maintained legacy tests, and current docs/artifact status. No unresolved CRITICAL, HIGH, or relevant MEDIUM finding remains locally at this checkpoint.

The old `3dc7df19...` P100 evidence and all v1/v2 canonical identities remain immutable historical evidence and are not current acceptance evidence because production numerical behavior changed afterward.

## Remaining hard gate

1. This exact connector-authored checkpoint must pass all seven permanent hosted workflows, including maintained Torch CPU and R `plm`/`sandwich` alignment.
2. A final read-only strict review must find no new local blocker.
3. Then exit locally as `PARTIAL_REMOTE_PENDING` and collect fresh exact-head Tesla P100 correctness/performance evidence: **47/47 per backend** plus **58 performance rows**.
4. Fresh evidence must be audited and promoted under new immutable parser/source identities; historical v1/v2 identities must never be overwritten.

PR #126 remains Draft, open, mergeable, and unmerged. Ready-for-review and merge remain explicit user lifecycle actions only.
