# PR #126 — Round 8 pre-physical review checkpoint

Date: 2026-08-13

This checkpoint records the independent Formula Compatibility review/fix loop performed after the Round-7 checkpoint. It is a pre-physical acceptance record: the user-authored commit adding this file must pass the permanent hosted workflows before exact-head hosted-green status is claimed.

## Reviewed technical candidate

Round-8 technical candidate:

`1492c95a879ddc30715aed0119c40dd0c0cf9fcb`

Relative to the Round-7 checkpoint `2f6b00f6759d2be7d16494fb9ac71587ef28f59d`, the final Round-8 tree changes only:

- `statgpu/panel/_formula.py`;
- `dev/tests/test_panel_formula.py`;
- `docs/en/models/panel.md`;
- `docs/cn/models/panel.md`.

No temporary guarded workflow remains in the final technical tree.

## Round-8 findings closed

### CRITICAL — magic fixed-effect tokens could silently rewrite legitimate regressor names

The legacy token parser used ordinary substring detection/replacement for `EntityEffects`, `TimeEffects`, and `FixedEffects`. A valid formula such as `y ~ EntityEffectsScore` could therefore be rewritten as `y ~ Score` while also enabling an entity fixed effect. If both columns existed, the model could silently fit a different specification from the one requested.

The parser now recognizes magic tokens only as complete identifiers at the top level of the main Patsy RHS. Regressor names such as `EntityEffectsScore`, `TimeEffectsTrend`, and `FixedEffectsWeight` remain ordinary variables, and occurrences inside transforms such as `C(EntityEffects)` are not rewritten.

### HIGH — linearmodels token syntax and fixest pipe syntax could be silently mixed

Previously token stripping occurred before pipe parsing, so formulas such as `y ~ x1 + EntityEffects | time` could be reinterpreted rather than rejected. The two fixed-effect grammars did not have a documented combined meaning.

The parser now isolates the pipe specification first, applies linearmodels-style tokens only to the main Patsy formula, and fails closed if token-based and pipe-based fixed-effect syntax are combined in one formula. Pure token syntax and pure pipe syntax remain supported.

### HIGH — quoted response names containing `~` could corrupt token scanning

The first Round-8 scanner revision located the RHS with `formula.find("~")`. Patsy legitimately supports quoted names such as `Q("y~EntityEffects")`; in that case the first `~` belongs to the quoted response name rather than the formula separator, allowing text from the LHS to be misclassified as a fixed-effect token.

The final parser locates the formula separator with a quote/bracket-aware top-level scan. An actual `PanelOLS` regression verifies that `Q("y~EntityEffects") ~ x1` is fitted as an ordinary no-FE level regression and matches the equivalent explicit design matrix.

### MEDIUM — post-token predictor validation used a different separator rule

After the quote-aware token scanner was fixed, the post-strip “predictors remain” guard still used `clean.split("~", 1)`. A quoted response containing `~` could therefore bypass the explicit effects-only failure path.

A shared `_top_level_formula_rhs_start()` helper now defines the top-level formula separator for both token scanning and post-strip predictor validation. `Q("y~value") ~ EntityEffects` consistently fails through the explicit no-predictor/token failure rather than leaking into a later parser-specific error.

## Focused evidence

Successful guarded workflows:

- `31679136252` — complete-token boundaries, mixed-syntax fail-closed behavior, actual `EntityEffectsScore` regression, EN/CN formula-contract docs; focused formula and final-review regressions PASS; temporary workflows removed;
- `31679503338` — quote-aware response-separator handling and actual `Q("y~EntityEffects") ~ x1` PanelOLS regression PASS; temporary workflow removed;
- `31679912402` — shared quote-aware RHS helper and quoted-response effects-only validation PASS; both temporary RHS workflows removed.

Two intermediate automation-only failures did not modify the production tree:

- `31678827075` — malformed temporary workflow YAML, zero jobs;
- `31679732324` — exact patch matcher typo before tests/commit.

They are not validation failures of the technical candidate.

## Final independent re-review

A fresh read-only review of the final Round-8 technical candidate found no additional open CRITICAL, HIGH, or relevant MEDIUM issue within the active formula/API scope.

The supported fixed-effect formula contract is now:

- ordinary Patsy/R formula behavior is preserved;
- linearmodels-style additive tokens are recognized only as complete top-level RHS identifiers;
- fixest pipe fixed effects remain supported;
- token and pipe fixed-effect syntaxes are alternatives and cannot be mixed;
- transforms/quoted names are not subject to substring token rewriting;
- unsupported effects-only or over-two-FE requests fail closed.

Non-additive uses of the reserved magic tokens are not advertised as supported grammar and are left to normal Patsy rejection rather than extending the project-specific parser language.

## Physical-evidence boundary

The accepted Tesla P100 measurement `a99726e19c535dfcd0a94711bbc8be6aac437584` remains immutable historical evidence for the preceding numerical tree only. Round 6 already changed production numerical/prediction behavior and the physical correctness runner; Round 7 changed transactional refit behavior; Round 8 changes formula parsing/API behavior. The old P100 artifacts are therefore not current exact-head acceptance evidence.

Current acceptance still requires a fresh exact-clean Tesla P100 correctness + synchronized performance run after the exact hosted-clean checkpoint is established. The historical v4 parser/source identities remain frozen to `a99726e1...`; fresh physical evidence must use a new parser/source identity and immutable promotion.

## Pre-hosted verdict

- CRITICAL: 0 open
- HIGH: 0 open
- relevant MEDIUM: 0 open
- unresolved actionable review threads: 0 at the preceding checkpoint; recheck after hosted completion
- temporary Round-8 workflows in technical candidate: 0
- physical GPU evidence: pending fresh Tesla P100 rerun

Provisional hard exit before this checkpoint's permanent workflows complete:

`LOCAL_REVIEW_CLEAN / HOSTED_PENDING / PHYSICAL_REMOTE_PENDING`

PR #126 remains Draft. No Ready-for-review transition or merge is authorized by this checkpoint.
