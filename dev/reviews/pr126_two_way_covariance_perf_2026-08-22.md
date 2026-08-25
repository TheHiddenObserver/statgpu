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

## Review-fix follow-up (same day)

A follow-up review of the acceptance check found and fixed two issues:

1. **Per-dimension residual bound** (`5068da3f`).  The first follow-up attempt
   scaled the cross contribution by the maximum row count across cluster
   dimensions, but Gram cross terms only exist *inside* one dimension (between
   its own tiers) with that dimension's own row count.  The maximum-row bound
   created phantom cross terms and rejected ordinary balanced panels again
   (10k-row fit went back to the exact path).  Each residual is now bounded
   against its own set's dominant tier and row count; 10k-row CuPy fits stay
   at ~1.3s and 100k-row at ~0.4s.
2. **Shared vectorized terms builder** (`dfb1db2f`) — the vectorized-first and
   certified-vectorized branches now share one
   `_append_vectorized_component_terms` implementation.

### Independent re-review (`4a7c1ba9`)

A further pass over the remaining PR-touched files (formula, reference
distributions, fixed effects, random effects, between, first difference)
found no CRITICAL/HIGH issues; the fixes recorded here:

1. **Student-t(1) extreme tail** — the Cauchy survival function subtracts
   `1 - (0.5 + atan(x)/pi)` and collapses to zero already around |t| ~ 1e15,
   although the true tail stays representable.  The df=1 branch now evaluates
   `2 atan(1/x)/pi` (well conditioned for large x), keeping the representable
   tail at |t| = 1e154 (`6.37e-155`, exact to the `2/(pi x)` reference), with
   zero statistics floored to `tiny` so no divide warning fires.
2. **Formula side-array alignment** — a side array longer than the original
   design rows was silently indexed by the retained-row positions; it is now
   validated to match `positions.max() + 1` and fails closed otherwise.
3. **CuPy `add.at` documented as safe** — atomic accumulation (unlike the
   non-atomic `maximum.at`/`scatter_max` corruption fixed earlier), so the
   additive scatter sites are retained with an explanatory comment.
4. **Residual-acceptance unit tests** — lock the ordinary/deep-cancellation/
   oversized/empty decisions of `_row_expansion_residual_acceptable`, plus a
   defensive empty-set check.

Deferred (documented, not blocking): union-find and predict-guard Python
loops in `_fixed_effects.py` for very large panels, and the host-transfer
cost of the CuPy `maximum.at` fallback under tiered reduction (correctness
first, gated by the risk classification).


### Codex review findings (10bbdaa0)

An independent review comment round on the PR produced six findings; all are
resolved:

1. **P1 — formula side-array alignment** (introduced by the earlier review):
   positions.max() + 1 is not the original row count when Patsy drops
   trailing rows, so valid full-length side arrays were rejected.  Over-long
   arrays are now aligned by the retained positions and only arrays too short
   to cover the last retained row fail closed; the test was updated.
2. **P1 — Driscoll-Kraay ordered-categorical chronology**: confirmed the DK
   path shares actorize_panel_metadata, which preserves declared category
   order; a test now locks DK ordered == numeric equivalence and a lexical
   negative control.
3. **P1 — CuPy grouped min/max host round trips**: the sequential host
   fallback was unconditional, copying the full score matrix to host in every
   clustered/DK reduction.  Ordinary magnitudes (<= 1e6, one decade below the
   observed corruption window) keep the native GPU scatter; only risky
   magnitudes fall back.  Verified on P100: clustered fits stay ~1.3s at 10k
   rows and the extreme 1e308 cancellation path still returns the exact
   residual (0.25).
4. **P2 — docs**: the Fama-MacBeth pages (en/cn) now record the fresh
   exact-head P100 acceptance (12/12 runners) instead of the stale
   'validation required' wording.
5. **P2 — runner backend provenance**: already resolved — the runners validate
   the fit-persisted _backend_name, not a device-derived guess.
6. **P2 — demean convergence scale**: the alternating loop checks both entity
   and time group-mean residuals every iteration, so a reintroduced entity
   mean after the time projection is caught; no change needed.


### Final independent re-review (e7d4d37a)

A closing pass re-examined the previously-touched code plus the remaining core
panel paths (_fama_macbeth.py fit/covariance/inference, _pooled.py fit and
covariance dispatch, formula side-array boundaries, reference-distribution
edge cases, and the CuPy grouped min/max condition gate).  No new
CRITICAL/HIGH/MEDIUM issues were found; the only follow-up was a test
correction (
etained-length side arrays pass through unchanged, only
too-short arrays fail closed).  CI is fully green and the exact-head P100
matrix (12/12 physical runners) passes.

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