# PR #80 Exact-Source Review/Fix Cycle — 2026-08-04

## Scope and active gates

Starting head: `e4a25bccb2a47df099085e26f31244df48f80038`.

Active axes:

| Axis | Decision |
|---|---|
| Backend | Three-backend behavior required for CoxPHCV and group-penalty promotion suites |
| CV | Supported; custom-grid ordering, staged safety, cache semantics, refit, and reusable custom splits are active |
| Inference | Unchanged by this cycle |
| Formula | Not formula-facing in this cycle |
| Benchmark | Exact-source physical CuPy and Torch evidence required |
| Docs | Public staged-fallback, changelog, and evidence contracts synchronized with the promotion lifecycle |

## Findings and fixes

[CRITICAL][ARTIFACT][fixed] `dev/benchmarks/*gpu_suite.py` — checkout hashes did not prove that Python imported the same checkout.

Impact: an editable install or site-package from another checkout could supply the runtime implementation while the JSON recorded the current Git commit and current-tree hashes.

Fix:
- added `dev/benchmarks/_exact_source_runtime.py`;
- every canonical suite now prepends the exact Git root to `PYTHONPATH`, sets `PYTHONNOUSERSITE=1`, and passes the same environment and checkout working directory to every child/sub-runner;
- a fresh subprocess imports requested modules, verifies every `__file__` is under the checkout, and hashes the files actually imported;
- child reports must expose passing `runtime_import_provenance`;
- negative hosted coverage injects a conflicting `PYTHONPATH` package and proves the checkout wins.

Evidence:
- `dev/tests/test_pr80_exact_source_runtime_provenance.py`;
- canonical child suites retain their schema-2 formats;
- the final aggregation report now emits machine schema 3 and records complete child reports.

[HIGH][PERF/BACKEND][fixed] `statgpu/survival/_cox_cv_staged_safety_contract.py` — explicit CuPy requests retained staged machinery with every candidate expanded to every stage, causing a second full-precision full-grid pass.

Impact: the safety fallback could approximately double candidate fitting on CuPy while CPU and Torch used one exhaustive pass.

Fix:
- staged and successive-halving flags are now disabled inside the selector on every backend;
- the raw selector is called exactly once;
- `staged_safety_strategy="single_pass_exhaustive"` is public;
- the physical runner sets the retained fold cache limit to zero and requires fold preparation count to equal the effective fold count, so a repeated pass fails the gate.

Evidence:
- `dev/tests/test_pr80_cox_cv_staged_safety_contract.py`;
- `dev/tests/test_pr80_target_transfer_overflow_cache.py`;
- `dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py`;
- synchronized EN/CN staged-safety guides and primary CoxPH model pages.

[MEDIUM][CV/API][fixed] `CoxPHCV.cv_splits` — a one-shot generator was exhausted after one fit and was not safe for cloning or serialization.

Impact: repeated fit could fail with no folds, and legacy sklearn clone/deepcopy could fail before reconstruction.

Fix:
- added `statgpu/survival/_cox_cv_split_lifecycle_contract.py`;
- one-shot split iterators are materialized once into a private reusable snapshot;
- repeated fit temporarily uses the snapshot without rewriting the public `cv_splits` attribute;
- `get_params()` exports the reusable equivalent needed by legacy sklearn cloning;
- modern clone and pickle use reusable fold sequences;
- `set_params(cv_splits=...)` invalidates the old snapshot.

Evidence:
- `dev/tests/test_pr80_cox_cv_split_lifecycle_contract.py`, including repeated public CPU fit, clone, legacy parameter reconstruction, pickle, and setter invalidation.

[MEDIUM][DOC/ARTIFACT][fixed] staged safety, current changelogs, durable artifact access, schema naming, and archive maintenance.

Fix:
- synchronized the EN/CN staged-safety guide and primary CoxPH pages with the one-pass contract;
- synchronized root, EN, and CN changelog entry points for 2026-08-04;
- preserved all earlier changelog content in date-labelled archive files and returned `.markdown` archives to the maintained link/content/mirror checks;
- published the unchanged exact-head JSON as a durable Gist with a recorded SHA-256;
- changed the final outer report from machine schema 2 to machine schema 3 and added a hosted structural assertion;
- distinguished evidence for the validated runtime commit from later documentation/schema commits that require their own exact-head promotion.

## Review-fix iterations and hosted validation

1. The first hosted run exposed two existing tests that still treated repeated staged fold preparation as correct when the retained fold cache was disabled. Those tests were updated to the new safety contract: independent of cache limit, requested staged execution prepares each effective fold once and reports `fold_state_cache_enabled=False`.
2. Re-review found that historical Group sub-runners do not publish `source_clean_after`. The canonical Group suite now checks the actual checkout status immediately after every sub-runner rather than trusting a missing nested field.
3. Re-review found the revised single-pass cache regression absent from the staged exact-source manifests. Both the inner runner and canonical suite now hash that test.
4. GitHub Actions workflow #946 passed on exact head `a726937a39eb0ed5a370dd03362884b63a9e9818`:
   - full CPU: 1879 passed, 662 skipped, 15 warnings;
   - static contracts: passed;
   - documentation contracts: passed;
   - Python 3.9, 3.10, 3.11, and 3.12 regression matrices: passed.
5. The physical final suite passed on the same clean exact head with 134/134 checks, identical commits at outer/child/nested levels, zero return codes, empty gate-failure arrays, passing runtime import provenance, Group CuPy/Torch 24/24 cases, passing Cox order/cache and staged-safety suites, all eight staged candidates evaluated at full precision, and one fold preparation per effective fold.
6. The unchanged raw artifact is durably published at
   `https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139`.
   Its SHA-256 is
   `e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb`
   and its size is 86,315 bytes.
7. The subsequent documentation/schema-maintenance commits intentionally create a new head. They do not invalidate the artifact as evidence for `a726937...`, but they do reopen the repository's exact-head promotion gate for the final PR head.

## Current audit conclusion

The runtime implementation at `a726937a39eb0ed5a370dd03362884b63a9e9818` completed both hosted and physical promotion. No coefficient, score, selected-penalty, backend-parity, lifecycle, or provenance counterexample remains open for that commit.

The current follow-up fixes only artifact-schema and documentation-contract defects. Because repository policy requires the physical artifact to bind the exact final head, the new head must receive one final clean CuPy/Torch run after hosted CI passes.

## Exit status

`PARTIAL_REMOTE_PENDING`

Run from a clean checkout of the final follow-up head:

```bash
python dev/benchmarks/benchmark_pr80_final_gpu_suite.py \
  --output results/benchmark_frontend_sources/pr80_final_gpu_suite_schema3.json
```

Promotion requires:
- outer `schema_version=3`;
- exact identical commit at outer, child, and nested levels;
- `runtime_import_provenance.passed=true` in all canonical suites;
- actual imported module paths and hashes under the checkout;
- clean before and after;
- zero return codes;
- Group, Cox custom-grid-order, and Cox staged-safety suites passing on CuPy and Torch;
- staged strategy `single_pass_exhaustive`;
- fold preparation count equal to effective folds on both GPU backends;
- every `gate_failures` array empty.

No commit may be added after the new passing physical artifact without rerunning this exact-head gate.
