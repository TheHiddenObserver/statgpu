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
| Docs | Public staged-fallback, changelog, and evidence contracts synchronized before the final physical run |

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
- schema version 2 for the final, group, custom-grid-order, and staged-safety canonical suites.

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
- synchronized EN/CN staged-safety guides.

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

[MEDIUM][DOC][fixed] staged safety, current changelogs, and exact-source evidence behavior.

Fix:
- synchronized the EN/CN staged-safety guide with the one-pass contract;
- synchronized root, EN, and CN changelog entry points for 2026-08-04;
- preserved all earlier changelog content in date-labelled archive files linked from the current entry points;
- canonical reports now carry runtime import provenance;
- this review/fix report records the remaining exact-head physical promotion gate without publishing a premature PASS claim.

## Review-fix iterations and hosted validation

1. The first hosted run exposed two existing tests that still treated repeated staged fold preparation as correct when the retained fold cache was disabled. Those tests were updated to the new safety contract: independent of cache limit, requested staged execution prepares each effective fold once and reports `fold_state_cache_enabled=False`.
2. Re-review found that historical Group sub-runners do not publish `source_clean_after`. The canonical Group suite now checks the actual checkout status immediately after every sub-runner rather than trusting a missing nested field.
3. Re-review found the revised single-pass cache regression absent from the staged exact-source manifests. Both the inner runner and canonical suite now hash that test.
4. GitHub Actions workflow #943 passed on implementation head `4c8f9493ee08e7ecf6ec88c7296c02070547cda2`:
   - full CPU: 1879 passed, 662 skipped, 15 warnings;
   - static contracts: passed;
   - documentation contracts: passed;
   - Python 3.9, 3.10, 3.11, and 3.12 regression matrices: passed.
5. The final documentation-only synchronization commits must receive the same hosted gates before physical promotion.

## Current audit conclusion

A third independent delta review found no remaining locally reproducible CRITICAL, HIGH, or actionable MEDIUM issue in the runtime, lifecycle, provenance, runner, test, or documentation changes covered by this cycle.

## Exit status

`PARTIAL_REMOTE_PENDING`

A clean exact-head physical run is required after the final hosted workflow passes:

```bash
python dev/benchmarks/benchmark_pr80_final_gpu_suite.py \
  --output results/benchmark_frontend_sources/pr80_final_gpu_suite_schema3.json
```

Promotion requires:
- exact identical commit at outer, child, and nested levels;
- `runtime_import_provenance.passed=true` in all canonical suites;
- actual imported module paths and hashes under the checkout;
- clean before and after;
- zero return codes;
- Group, Cox custom-grid-order, and Cox staged-safety suites passing on CuPy and Torch;
- staged strategy `single_pass_exhaustive`;
- fold preparation count equal to effective folds on both GPU backends;
- every `gate_failures` array empty.

No commit may be added after a passing physical artifact without rerunning this exact-head gate.
