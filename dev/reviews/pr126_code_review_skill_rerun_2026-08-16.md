# PR126 `.claude/skills/code-review.md` rerun — 2026-08-16

## Scope and active gates

Exact review baseline before this fix series: `36a14ae0f38aec5bdcdd9eec0b45b7da2975bc05`.

Active axes:

- `BACKEND`: NumPy / CuPy / Torch, explicit-device no-fallback contract.
- `INFER`: Fama-MacBeth coefficient-series covariance and retained-period identification.
- `FORMULA`: default intercept, explicit no-intercept rejection, missing-row side-array alignment.
- `MATRIX`: retained-period rank policy and rank-revealing solve.
- `PERF`: per-period decomposition/synchronization cost.
- `TEST`: maintained regression and physical-runner contracts.
- `DOC`: root/EN/CN validation-state consistency.
- `ARTIFACT`: physical evidence reproducibility and benchmark frontend staleness triggers.

Inactive axes with rationale:

- `CV`: Fama-MacBeth is non-tunable in this API.
- `LOSS`, `PENALTY`, `SOLVER`: no regularized optimization capability is touched.

## Findings and fixes

### [MEDIUM][DOC][fixed] stale exact-head physical-validation status

The detailed EN/CN changelogs still described the post-`a99726e1` physical gate as awaiting a rerun even though the later Fama-MacBeth correctness source `464b587e83b234d78b5449666488d7f2f8ad367c` had already been validated on Tesla P100.

Fix:

- synchronized `CHANGELOG.md`, `docs/en/changelog.md`, and `docs/cn/changelog.md`;
- preserved older P100 runs as immutable historical evidence;
- recorded the accepted `464b587e...` Stage-C 35+12 and focused Fama-MacBeth evidence;
- after the single-factorization numerical optimization below, explicitly reopened exact-head physical acceptance rather than incorrectly carrying the older evidence forward.

### [MEDIUM][TEST/ARTIFACT][fixed] benchmark staleness trigger did not match catalog scan root

The benchmark source catalog recursively scans `results/**/*.json`, while Benchmark Frontend CI previously enumerated only selected PR/result directories. A newly tracked evidence JSON in another `results/` subtree could therefore change deterministic inventory/catalog output without triggering the staleness job.

Fix:

- both `push.paths` and `pull_request.paths` now include `results/**/*.json`;
- removed dependence on per-PR result-directory allowlists.

### [MEDIUM][PERF][fixed] retained-period rank guard duplicated matrix decomposition

The Fama-MacBeth correctness fix first called `panel_matrix_rank(X_t, xp)` (SVD) and then solved normal equations, so every retained period paid for two decompositions plus the rank synchronization.

Fix:

- `FamaMacBeth.fit()` now calls `panel_lstsq(X_t, y_t, xp)` once;
- the shared rank-revealing SVD supplies both the minimum-norm least-squares coefficient and the exact shared rank decision;
- the full-rank requirement is unchanged and still fails closed before averaging/inference;
- maintained regression monkeypatches the shared SVD primitive and asserts exactly one call per retained period.

### [LOW][ARTIFACT][fixed] focused Fama-MacBeth P100 evidence lacked a committed runner

Fix:

- added `dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py`;
- runner requires an exact expected SHA and clean working tree;
- checks ordered-categorical chronology, formula/missing-row alignment, lexical negative control, rank rejection, both no-intercept spellings, requested/executed backend identity, and a realistic multi-period numerical parity workload;
- timing uses backend synchronization and records raw samples/median only; it makes no speedup claim;
- maintained CPU regression imports the runner and executes its timing workload with the NumPy backend so syntax/schema/basic execution cannot silently rot.

## Validation boundary

Hosted checks must be rerun on the final executable tree because production source, tests, workflow, benchmark runner, and docs changed.

The numerical Fama-MacBeth implementation changed after accepted physical source `464b587e...`, so the physical gate is reopened. Both physical runners require a clean worktree at startup; therefore their outputs must first be written outside the repository. Run both against one clean final SHA, then copy the successful artifacts into `results/`:

```bash
FULL_SHA="$(git rev-parse HEAD)"
SHORT_SHA="$(git rev-parse --short=8 HEAD)"
test -z "$(git status --porcelain)" || exit 1

python dev/benchmarks/validate_panel_stage_c_gpu.py \
  --out "/tmp/panel_stage_c_correctness_${SHORT_SHA}.json" \
  --expected-sha "${FULL_SHA}"

test -z "$(git status --porcelain)" || exit 1

python dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py \
  --out "/tmp/fama_macbeth_review_fix_${SHORT_SHA}.json" \
  --expected-sha "${FULL_SHA}"

test -z "$(git status --porcelain)" || exit 1

mkdir -p results/pr126_p100_fama_fix
cp "/tmp/panel_stage_c_correctness_${SHORT_SHA}.json" \
  "results/pr126_p100_fama_fix/panel_stage_c_correctness_${SHORT_SHA}.json"
cp "/tmp/fama_macbeth_review_fix_${SHORT_SHA}.json" \
  "results/pr126_p100_fama_fix/fama_macbeth_review_fix_${SHORT_SHA}.json"
```

Both default to physical CuPy + Torch validation. The focused runner additionally records synchronized timing samples and NumPy numerical parity for the single-factorization path. Adding the evidence JSONs intentionally changes the benchmark source inventory; regenerate deterministic frontend/docs benchmark assets before the evidence commit is finalized so the staleness gate remains current.

## Current exit state

Until the current executable-tree hosted workflows and the two physical commands above pass:

`PARTIAL_REMOTE_PENDING`

No merge action is part of this review/fix loop.
