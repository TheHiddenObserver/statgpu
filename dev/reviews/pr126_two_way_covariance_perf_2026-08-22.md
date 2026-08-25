# PR #126 two-way covariance performance optimization — 2026-08-22

Review standard: `.claude/skills/code-review.md`, `auto-fix` mode.

This addendum records the performance optimization that closed the
`benchmark_panel_stage_c_covariance.py` gate, which could not complete after
the PR's own `050a1ba1 perf: scope two-way precision fallback` introduced the
row-level exact accumulation fallback.

## Problem

`two_way_clustered_covariance` fires `_component_row_reduction_needs_expansion`
when any component column satisfies

`min_nonzero / max_abs < sqrt(16 * n * eps)`.

For ordinary normally distributed influence scores the empirical min/max ratio
decays like `1/n` while the threshold grows like `sqrt(n)`, so the two cross at
roughly `n > 6500`: every balanced panel above ~6.5k rows took the exact
per-row dyadic two-sum chain (`~61560 * ~52` scalar terms for 10k rows), which
cost ~1000 seconds per fit on CuPy P100. The full Stage-C performance
benchmark therefore could not complete (60+ minute timeouts).

## Why simple batching failed

Three batching attempts were implemented and measured:

1. **Batched row outer products + per-row two-fold + row-axis merge tree** —
   component dimensions have different group counts (e.g. 500 / 20 / 10000),
   so padded "rows" have no cross-dimension correspondence; the exact path's
   cross-row inclusion-exclusion cancellation is a term-ordered compensated
   chain that batching cannot reproduce.
2. **Adjacent-pair merge tree** — same-sign dyadic terms merge exactly
   (multiplication by two is exact), then cancel exactly, dropping the
   residual that the global chain preserves (`5.16e304 - 5.16e304 == 0` vs the
   true residual `4.95e211`).
3. **Two-level (row then cross-row) compensation** — each row's grow-expansion
   already rounds at `eps^2` of the row's dominant term (`~1e272` at 1e304
   scale), which is larger than the true cross-row residual (`~1e211`), so the
   restored covariance overflows.

Item-level exact summation is inherently sequential (Rump/Neumaier chains),
so the exact path cannot be vectorized without changing its numerical result.

## Fix

`needs_row_expansion=True` now computes the **vectorized** Gram first and
accepts it when the largest tier residual is negligible:

- relative to the dominant component within the retiering floor
  `sqrt(16 n eps) * 16` (the same floor `_retier_component_for_safe_gram`
  already uses to guarantee each retained tier is independently Gram-safe);
- and its cross/self Gram contribution is within that floor of the
  vectorized result scale.

Deep-cancellation designs (whose vectorized result collapses toward zero)
still reject the check and keep the exact per-row path, so the numerical
contracts in `test_panel_stage_c_covariance.py` (exact-zero multiscale
fixtures, third-magnitude components, unsafe cross cancellation) are
unchanged.

The first attempt used an `eps * 1024` bound, ~1e7 tighter than the retiering
floor, which rejected every retiered residual and therefore never accepted the
vectorized result; the bound was aligned to `sqrt(16 n eps) * 16`.

## Validation

- Local: `test_panel_stage_c_covariance.py` 60/60; full suite 3008 passed
  (5 pre-existing unrelated failures reproduced on the pre-fix head).
- Remote Tesla P100 (CuPy 13.6.0 / Torch 2.0.0+cu117), clean detached
  worktree at `528d967e`:
  - `pooled_cluster_two_way` fit 10k rows: **1018s -> 1.28s** (CuPy),
    0.21s (Torch); 100k rows: 0.41s / 0.10s.
  - Full `benchmark_panel_stage_c_covariance.py` completes in ~40s
    (60 rows: base 54 + high_t_qs 4 + two_way_unbalanced 2), all medians
    0.01–0.83s.
  - All 12 physical runners PASS at the same head (stage_c 35 cases/backend,
    fama optimized, hac chronology, t2 tail, device affinity, fama scaling,
    rhs cancellation x2, rank precedence x2, intercept cancellation x2).
- Generated frontend/docs benchmark assets refreshed deterministically from a
  clean worktree (`ffc848fc`) so the CI `staleness` gate passes.

## Files

- `statgpu/panel/_covariance.py`: `_row_expansion_residual_acceptable` +
  vectorized-first fallback in `two_way_clustered_covariance`.
- `frontend/public/data/*.json`, `docs/assets/benchmarks/data/*.json`:
  deterministic regeneration.

## Review exit

- actionable CRITICAL: 0 open
- actionable HIGH: 0 open
- locally actionable MEDIUM: 0 open
- remaining hard exit: final exact-head hosted CI on the new head