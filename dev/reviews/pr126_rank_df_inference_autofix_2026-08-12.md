# PR #126 rank-deficient df / inference auto-fix review — 2026-08-12

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Current verdict

**PARTIAL_REMOTE_PENDING / LOCAL REVIEW IN PROGRESS / NOT MERGE-READY**

Active axes: public API/presentation, inference, NumPy/CuPy/Torch backend behavior, formula/panel metadata, benchmark/performance, and docs/artifacts. Loss, penalty, generic solver framework, and CV remain inactive.

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

### [HIGH][MATRIX][fixed locally / needs remote GPU] Physical matrix did not exercise the repaired df branches

The physical runner now adds eight exact-collinearity estimator integrations: PanelOLS(entity FE), BetweenOLS, FirstDifferenceOLS, and RandomEffects under both nonrobust and historical HC1/robust. Each case persists `fit_rank` and `parameter_count`; fresh GPU audit must verify `fit_rank < parameter_count` and requested/executed backend identity.

Fresh target: **47/47 per backend = 35 estimator integrations + 12 public primitives**. Performance remains 58 rows. The timing runner now discovers CUDA-suffixed CuPy distribution versions.

## Validation completed so far

- first focused source/external matrix: **86 passed**;
- strict variance/entity-FE re-review matrix: success;
- expanded runner + maintained Torch contract matrix: **45 passed**;
- syntax / compile checks: success;
- `git diff --check`: success.

## Remaining gate

After docs/status synchronization, run the seven permanent hosted workflows on a connector-authored local-review checkpoint and perform another independent strict review. If no local blocker remains, exit as `PARTIAL_REMOTE_PENDING` and collect fresh exact-head P100 correctness/performance artifacts. Old `3dc7df19...` and all v1/v2 canonical identities remain immutable historical evidence.
